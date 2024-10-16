import math
import torch
from torch.nn import functional as F

from dataset import Dataset
from evaluator import Evaluator
from model import DpoWrapper, TutorialLLM


class Trainer():

    def __init__(self, model: TutorialLLM, dataset: Dataset, evaluator: Evaluator, device: str) -> None:
        self.model = model
        self.dataset = dataset
        self.evaluator = evaluator
        self.device = device

    def pretrain(self, iterations):
        # Reset the evaluator to clear the loss history
        self.evaluator.reset()
        # Initialize an optimizer with learning rate 1e-3
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3)

        for i in range(iterations):
            # Get a batch of pretrain data
            inputs, labels = self.dataset.get_batch_pretrain('train')
            # Forward pass and calculate the loss
            _, loss = self.model(inputs, labels)

            # Evaluate the model performance
            self.evaluator.evaluate_pretrain(self.model, i, loss.item())

            # Backward pass and update the model
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        print('Save the pretrained model...')
        torch.save(self.model, 'model_pretrain.pth')

    def finetune(self, epochs):
        # Initialize an optimizer with learning rate 1e-3
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3)
        
        for epoch in range(epochs):
            # Reset the evaluator to clear the loss history for each epoch
            self.evaluator.reset()
            for i, (inputs, labels) in enumerate(self.dataset.get_batch_generator_finetune('train')):
                # Forward pass and calculate the loss
                _, loss = self.model(inputs, labels)

                # Evaluate the model performance
                self.evaluator.evaluate_finetune(self.model, epoch, i, loss.item())

                # Backward pass and update the model
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        print('Save the finetuned model...')
        torch.save(self.model, 'model_finetune.pth')

    def align(self, epochs):
        # The alignment needs a reference model for DPO, we use a DpoWrapper to manage the 2 models
        dpo_wrapper = DpoWrapper(self.model)
        # Initialize an optimizer with learning rate 1e-5
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-5)
        for epoch in range(epochs):
            # Reset the evaluator to clear the loss history for each epoch
            self.evaluator.reset()
            for i, (positive_inputs, positive_labels, negative_inputs, negative_labels) in enumerate(self.dataset.get_batch_generator_alignment('train')):
                loss, reward_margin = dpo_wrapper.forward(positive_inputs, positive_labels, negative_inputs, negative_labels)

                # Evaluate the model every evaluation_interval iterations
                self.evaluator.evaluate_alignment(dpo_wrapper, epoch, i, loss.item(), reward_margin.item())

                # Backward pass and update the model
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        print('Save the aligned model...')
        torch.save(self.model, 'model_aligned.pth')