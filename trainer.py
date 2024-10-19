import torch

from dataset import Dataset
from evaluator import Evaluator
from model import DpoWrapper, TutorialLLM


class Trainer():
    """
    Trainer for the model.

    This module provides methods to pretrain, finetune, and align the model.
    """

    def __init__(self, model: TutorialLLM, dataset: Dataset, evaluator: Evaluator, device: str) -> None:
        """
        Initialize the trainer with the model, dataset, evaluator, and device.

        Args:
            model: The model to be trained.
            dataset: The dataset to provide training data.
            evaluator: The evaluator to evaluate the model performance.
            device: The device to run the model on ('cpu' or 'cuda').
        """
        self.model = model
        self.dataset = dataset
        self.evaluator = evaluator
        self.device = device

    def pretrain(self, iterations: int) -> None:
        """
        Pretrain the model for a certain number of iterations.

        For each iteration, a batch of pretrain data is used to train the model.

        Args:
            iterations: The number of iterations to pretrain the model.
        """
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

    def finetune(self, epochs) -> None:
        """
        Finetune the model for a certain number of epochs.

        For each epoch, a batch of finetune data is used to train the model.

        Args:
            epochs: The number of epochs to finetune the model.
        """
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

    def align(self, epochs) -> None:
        """
        Align the model with our preference for a certain number of epochs.

        For each epoch, a batch of alignment data is used to train the model.

        Args:
            epochs: The number of epochs to align the model with our preference.
        """
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