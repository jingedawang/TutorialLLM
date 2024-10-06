import json
import random
import torch
import torch.nn as nn
from torch.nn import functional as F


class Dataset():
    def __init__(self, input_path: str = 'data.json', batch_size: int = 16, block_size: int = 256, device: str = 'cpu'):
        self.batch_size = batch_size
        self.block_size = block_size
        self.device = device

        # Load poems data
        poems = json.load(open(input_path, 'r', encoding='utf-8'))
        random.shuffle(poems)
        # Split the data into 5:3:2 for pretrain, instruction finetune, and alignment
        pretrain_poems = poems[:int(len(poems)*0.5)]
        finetune_poems = poems[int(len(poems)*0.5):int(len(poems)*0.8)]
        alignment_poems = poems[int(len(poems)*0.8):]

        # Reformat pretrain data
        pretrain_text = []
        for poetry in pretrain_poems:
            paragraphs = '\n'.join(poetry['paragraphs'])
            pretrain_text.append(f'{poetry['title']}\n{paragraphs}')
        pretrain_text = '\n\n'.join(pretrain_text)
        print('The whole pretrain data is a long text with all poems concatenated together. Here are the first 100 characters:')
        print(pretrain_text[:100])

        # Reformat instruction finetune data
        finetune_texts = []
        instruction = '請用以下題目寫一首詩'
        instruction_label = '<INS>'
        input_label = '<INP>'
        response_label = '<RES>'
        for poetry in finetune_poems:
            paragraphs = '\n'.join(poetry['paragraphs'])
            content = f'{instruction_label}{instruction}{input_label}{poetry["title"]}{response_label}{paragraphs}'
            finetune_texts.append(content)
        print('The instruction finetune data is a list of formatted texts. Here is the first item:')
        print(finetune_texts[0])

        # Create a vocabulary from all the characters appeared in the dataset
        all_text = f'{pretrain_text}{"".join(finetune_texts)}\0'
        characters = sorted(list(set(all_text)))
        self.vocabulary_size = len(characters)
        print(f'Dataset length: {len(all_text)}, vocabulary size: {self.vocabulary_size}')
        # Create a mapping from characters to indices and vice versa
        character_to_index = { character: index for index, character in enumerate(characters) }
        index_to_character = { index: character for index, character in enumerate(characters) }
        # Encode method to convert a text to a list of indices
        self.encode = lambda text: [character_to_index[character] for character in text]
        # Decode method to convert a list of indices back to a text
        self.decode = lambda index_list: ''.join([index_to_character[index] for index in index_list])

        # Train and test splits for pretrain data
        pretrain_data = torch.tensor(self.encode(pretrain_text), dtype=torch.long)
        # Split the data into 90% train and 10% evaluate
        self.pretrain_train_data = pretrain_data[:int(0.9 * len(pretrain_data))]
        self.pretrain_evaluate_data = pretrain_data[int(0.9 * len(pretrain_data)):]

        # Train and test splits for instruction finetune data
        finetune_data = [torch.tensor(self.encode(finetune_text), dtype=torch.long) for finetune_text in finetune_texts]
        # Split the data into 90% train and 10% evaluate
        print(len(finetune_data))
        self.finetune_train_data = finetune_data[:int(0.9 * len(finetune_data))]
        self.finetune_evaluate_data = finetune_data[int(0.9 * len(finetune_data)):]

    def get_batch_pretrain(self, split: str):
        """
        Generate a batch of pretrain data

        Args:
        - split: 'train' or 'evaluate'
        """
        # Choose train or evaluate split
        data = self.pretrain_train_data if split == 'train' else self.pretrain_evaluate_data
        # Randomly choose the starting index of each item in the batch
        start_indices = torch.randint(len(data) - self.block_size, (self.batch_size,))
        # The input texts are all the characters in interval [start_index, start_index + block_size) for each item in the batch
        inputs = torch.stack([data[index:index+self.block_size] for index in start_indices])
        # The label texts are all the characters in interval [start_index + 1, start_index + block_size + 1) for each item in the batch.
        # So, the label texts are the same as the input texts, but shifted by 1 character to the right.
        labels = torch.stack([data[index+1:index+self.block_size+1] for index in start_indices])
        # Move the tensors to the device and return
        return inputs.to(self.device), labels.to(self.device)

    def generate_batch_finetune(self, split: str):
        """
        Generate a batch of instruction finetune data

        Args:
        - split: 'train' or 'evaluate'
        """
        # Choose train or evaluate split
        data = self.finetune_train_data if split == 'train' else self.finetune_evaluate_data

        def process_batch(batch: list):
            # All the inputs and labels are initialized to zeros of largest length
            inputs = torch.zeros(len(batch), self.block_size, dtype=torch.long)
            labels = torch.zeros(len(batch), self.block_size, dtype=torch.long)
            for i, item in enumerate(batch):
                # Assign the real values to the zeros-initialized tensors
                available_length = len(item) if len(item) < self.block_size else self.block_size
                inputs[i, :available_length] = item[:available_length]
                labels[i, :available_length-1] = item[1:available_length]

                # Mask all the remaining zeros by setting them to -100 (the loss function will ignore these tokens)
                mask = labels[i] == 0
                indices = torch.nonzero(mask).squeeze()
                if indices.numel() > 1:
                    # Exclude the first zero because it marks the end of the text
                    labels[i, indices[1:]] = -100
            return inputs, labels

        # Initialize an empty list to store the batch
        batch = []
        for item in data:
            batch.append(item)
            # If the batch is full, process it and yield
            if len(batch) >= self.batch_size:
                inputs, labels = process_batch(batch)
                batch = []
                yield inputs.to(self.device), labels.to(self.device)
        # If there are still remaining items, process them and yield
        if len(batch) > 0:
            inputs, labels = process_batch(batch)
            yield inputs.to(self.device), labels.to(self.device)