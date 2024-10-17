import json
import random
import torch


class Dataset():
    """
    Dataset class to load and process the poems data. It provides methods to generate batches of pretrain, instruction finetune, and alignment data.

    The data is represented as lists of token ids, where each token is a character in the vocabulary.
    The token id and character mapping is stored in the `encode` and `decode` methods.
    """

    def __init__(self, input_path: str = 'data.json', batch_size: int = 16, max_length: int = 256, device: str = 'cpu'):
        """
        Initialize the dataset with a JSON file or poems.

        The input JSON file contains a list of poems, each of which has a title and a list of paragraphs.
        All the data will be split into pretrain, instruction finetune, and alignment data with a ratio of 5:3:2.
        + For pretrain data, poems are put together to form a long text.
        + For instruction finetune data, each poem is formatted as an instruction-response pair.
            The instruction is a fixed string '請用以下題目寫一首詩' and a title, while the response is the paragraphs of the poem.
        + For alignment data, each item contains a positive-negative pair of poems. The positive pair is the original poem,
            while the negative pair has at least one paragraph replaced by a random paragraph from other poems.

        Data in each category will be further split into train and evaluate sets.
        All the data will be tokenized into a token id sequence, where each token is a character in the vocabulary.
        This is necessary for the model to process the data.

        Args:
            input_path: The path to the JSON file containing the poems data.
            batch_size: The number of items in a batch.
            max_length: The maximum length of a text to be processed.
            device: The device to run the model on ('cpu' or 'cuda').
        """
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = device

        # Load poems JSON file
        poems = json.load(open(input_path, 'r', encoding='utf-8'))
        # Shuffle the poems to make it random
        random.seed(2024)
        random.shuffle(poems)
        # Split the data into 5:3:2 for pretrain, instruction finetune, and alignment
        pretrain_poems = poems[:int(len(poems)*0.5)]
        finetune_poems = poems[int(len(poems)*0.5):int(len(poems)*0.8)]
        alignment_poems = poems[int(len(poems)*0.8):]

        # Reformat pretrain data. All poems are concatenated directly to form a long text.
        # We don't care about the format in pretrain stage. This data is just used to make
        # the model learn how the poem text looks like.
        pretrain_texts = []
        for poetry in pretrain_poems:
            positive_paragraphs = '\n'.join(poetry['paragraphs'])
            pretrain_texts.append(f'{poetry["title"]}\n{positive_paragraphs}')
        pretrain_text = '\n\n'.join(pretrain_texts)
        print('The whole pretrain data is a long text with all poems concatenated together. Here are the first 100 characters:')
        print(pretrain_text[:100])

        # Reformat instruction finetune data. Each poem is formatted as an instruction-response pair.
        # The target of this stage is to fix the format of the poems generate by the model.
        finetune_texts = []
        instruction = '請用以下題目寫一首詩'
        instruction_label = '<INS>'
        input_label = '<INP>'
        response_label = '<RES>'
        for poetry in finetune_poems:
            positive_paragraphs = '\n'.join(poetry['paragraphs'])
            content = f'{instruction_label}{instruction}{input_label}{poetry["title"]}{response_label}{positive_paragraphs}'
            finetune_texts.append(content)
        print('The instruction finetune data is a list of formatted texts. Here is the first item:')
        print(finetune_texts[0])

        # Reformat alignment data.
        # The end character of each paragraph will be replaced by a random character from other paragraphs as the negative sample.
        # This stage is used to let the model learn the phonetic structure of the poem.
        alignment_texts = []
        for poetry in alignment_poems:
            positive_paragraphs = poetry['paragraphs']
            negative_paragraphs = positive_paragraphs.copy()
            # Randomly choose a paragraph to replace the last character with a random character from other poems
            random_index = random.randint(0, len(negative_paragraphs)-1)
            random_paragraph = random.choice([paragraph for paragraph in random.choice(alignment_poems)['paragraphs']])
            negative_paragraphs[random_index] = negative_paragraphs[random_index][:-2] + random_paragraph[-2] + negative_paragraphs[random_index][-1]
            positive_paragraphs = '\n'.join(positive_paragraphs)
            negative_paragraphs = '\n'.join(negative_paragraphs)
            alignment_texts.append((positive_paragraphs, negative_paragraphs))
        print('The alignment data is a list of positive-negative pairs. Here is the first pair:')
        print(alignment_texts[0])

        # Create a vocabulary from all the characters appeared in the dataset
        # Note that we add a special character '\0' in the end, which is used as an end-of-text token.
        # An end-of-text token is useful to let the model know when to stop generating text.
        all_text = f'{pretrain_text}{"".join(finetune_texts)}{"".join([pair[0] + pair[1] for pair in alignment_texts])}\0'
        # Get a sorted list of unique characters
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

        # Train and evaluate splits for pretrain data
        pretrain_data = torch.tensor(self.encode(pretrain_text), dtype=torch.long)
        # Split the data into 90% train and 10% evaluate
        self.pretrain_train_data = pretrain_data[:int(0.9 * len(pretrain_data))]
        self.pretrain_evaluate_data = pretrain_data[int(0.9 * len(pretrain_data)):]

        # Train and evaluate splits for instruction finetune data
        finetune_data = [torch.tensor(self.encode(finetune_text), dtype=torch.long) for finetune_text in finetune_texts]
        # Split the data into 90% train and 10% evaluate
        self.finetune_train_data = finetune_data[:int(0.9 * len(finetune_data))]
        self.finetune_evaluate_data = finetune_data[int(0.9 * len(finetune_data)):]

        # Train and evaluate splits for alignment data
        alignment_data = [(torch.tensor(self.encode(pair[0]), dtype=torch.long), torch.tensor(self.encode(pair[1]), dtype=torch.long)) for pair in alignment_texts]
        # Split the data into 90% train and 10% evaluate
        self.alignment_train_data = alignment_data[:int(0.9 * len(alignment_data))]
        self.alignment_evaluate_data = alignment_data[int(0.9 * len(alignment_data)):]

    def get_batch_pretrain(self, split: str):
        """
        Generate a batch of pretrain data.

        Each batch is a random block of text with the length of `max_length`.
        So there is no epoch boundary in the pretrain data. The batches is always unique.

        Args:
            split: Indicate whether to generate a batch for training or evaluation ('train' or 'evaluate').

        Returns:
            Two tensors of shape (`batch_size`, `max_length`), where the first tensor is the input tokens and the second tensor is the label tokens.
            The second dimension is the length of the text. We formed each label by shifting the input by one character to the right.
        """
        # Choose train or evaluate split
        data = self.pretrain_train_data if split == 'train' else self.pretrain_evaluate_data
        # Randomly choose the starting index of each item in the batch
        start_indices = torch.randint(len(data) - self.max_length, (self.batch_size,))
        # The input texts are all the characters in interval [start_index, start_index + max_length) for each item in the batch
        inputs = torch.stack([data[index:index+self.max_length] for index in start_indices])
        # The label texts are all the characters in interval [start_index + 1, start_index + max_length + 1) for each item in the batch.
        # So, the label texts are the same as the input texts, but shifted by 1 character to the right.
        # This forms `max_length` number of training examples for a single input-label pair.
        # For each subsequence from `start_index` to `start_index + i`, where i = 1, 2, ..., `max_length`, the label is `start_index + i + 1`, which denotes the next character.
        labels = torch.stack([data[index+1:index+self.max_length+1] for index in start_indices])
        # Move the tensors to the device and return
        return inputs.to(self.device), labels.to(self.device)

    def get_batch_generator_finetune(self, split: str):
        """
        Get a generator to yield batches of instruction finetune data.

        Data is consumed in a streaming fashion, so the generator will keep yielding batches to form an epoch.
        This is useful to train the model multiple epochs without loading all the data into memory.

        Args:
            split: Indicate whether to generate a batch for training or evaluation ('train' or 'evaluate').

        Yields:
            Two tensors of shape (batch_size, T), where the first tensor is the input tokens and the second tensor is the label tokens, T <= `max_length`.
            The second dimension is the length of the text. We formed each label by shifting the input by one character to the right.
        """
        # Choose train or evaluate split
        data = self.finetune_train_data if split == 'train' else self.finetune_evaluate_data

        # Initialize an empty list to store the batch
        batch = []
        for item in data:
            batch.append(item)
            # If the batch is full, process it and yield
            if len(batch) >= self.batch_size:
                inputs, labels = self.process_batch(batch)
                # Reset the batch for the next iteration
                batch = []
                # Return a batch of inputs and labels to the caller
                yield inputs.to(self.device), labels.to(self.device)
        # If there are still remaining items, process them and yield
        if len(batch) > 0:
            inputs, labels = self.process_batch(batch)
            yield inputs.to(self.device), labels.to(self.device)

    def process_batch(self, batch: list):
        """
        Process a batch of instruction finetune data.

        Args:
            batch: A list of token id lists, where each list is a poem represented by token ids.

        Returns:
            A batch of input token id lists and label token ids. The label refer to the next character of each input sequence
        """
        # All the inputs and labels are initialized to zeros of largest length
        inputs = torch.zeros(len(batch), self.max_length, dtype=torch.long)
        labels = torch.zeros(len(batch), self.max_length, dtype=torch.long)
        for i, item in enumerate(batch):
            # Assign the actual values to the zeros-initialized tensors
            available_length = len(item) if len(item) < self.max_length else self.max_length
            inputs[i, :available_length] = item[:available_length]
            # The same format as pretrain data, the label is the next character of the input
            labels[i, :available_length-1] = item[1:available_length]

            # Mask all the remaining zeros by setting them to -100 (the loss function will ignore these tokens)
            mask = labels[i] == 0
            indices = torch.nonzero(mask).squeeze()
            # Check if there are more than one zeros in the label
            if indices.numel() > 1:
                # Exclude the first zero because it marks the end of the text
                labels[i, indices[1:]] = -100
        return inputs, labels

    def get_batch_generator_alignment(self, split: str):
        """
        Get a generator to yield batches of alignment data.

        Data is consumed in a streaming fashion, so the generator will keep yielding batches to form an epoch.

        Args:
            split: Indicate whether to generate a batch for training or evaluation ('train' or 'evaluate').

        Returns:
            Two group of tensors of shape (batch_size, T) for positive batch and negative batch.
            Each group contains the input tokens and the label tokens.
        """
        # All the inputs and labels are initialized to zeros of largest length
        positive_inputs = torch.zeros(self.batch_size, self.max_length, dtype=torch.long)
        positive_labels = torch.zeros(self.batch_size, self.max_length, dtype=torch.long)
        negative_inputs = torch.zeros(self.batch_size, self.max_length, dtype=torch.long)
        negative_labels = torch.zeros(self.batch_size, self.max_length, dtype=torch.long)

        # Choose train or evaluate split
        data = self.alignment_train_data if split == 'train' else self.alignment_evaluate_data

        # Initialize an empty list to store the batch
        batch = []
        for positive_item, negative_item in data:
            batch.append((positive_item, negative_item))
            # If the batch is full, process it and yield
            if len(batch) >= self.batch_size:
                positive_inputs, positive_labels = self.process_batch([item[0] for item in batch])
                negative_inputs, negative_labels = self.process_batch([item[1] for item in batch])
                # Reset the batch for the next iteration
                batch = []
                # Return a batch of inputs and labels to the caller
                yield positive_inputs.to(self.device), positive_labels.to(self.device), negative_inputs.to(self.device), negative_labels.to(self.device)
        # If there are still remaining items, process them and yield
        if len(batch) > 0:
            positive_inputs, positive_labels = self.process_batch([item[0] for item in batch])
            negative_inputs, negative_labels = self.process_batch([item[1] for item in batch])
            yield positive_inputs.to(self.device), positive_labels.to(self.device), negative_inputs.to(self.device), negative_labels.to(self.device)