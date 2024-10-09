import math
import torch
from torch.nn import functional as F

from dataset import Dataset
from evaluator import Evaluator
from model import TutorialLLM


class Trainer():

    def __init__(self, model: TutorialLLM, dataset: Dataset, evaluator: Evaluator, device: str) -> None:
        self.model = model
        self.dataset = dataset
        self.evaluator = evaluator
        self.device = device

        self.learning_rate = 1e-3

    def pretrain(self, iterations):
        # Reset the evaluator to clear the loss history
        self.evaluator.reset()
        # Use AdamW optimizer
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)

        for i in range(iterations):
            # Get a batch of pretrain data
            inputs, labels = self.dataset.get_batch_pretrain('train')
            # Forward pass and calculate the loss
            logits, loss = self.model(inputs, labels)

            # Evaluate the model performance
            self.evaluator.evaluate_pretrain(self.model, i, loss.item())

            # Backward pass and update the model
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        print('Save the pretrained model...')
        # Save model to file
        torch.save(self.model, 'model_pretrain.pth')

    def finetune(self, epochs):
        # Reset the optimizer
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        
        for epoch in range(epochs):
            # Reset the evaluator to clear the loss history for each epoch
            self.evaluator.reset()
            for i, (inputs, labels) in enumerate(self.dataset.get_batch_generator_finetune('train')):
                # Forward pass and calculate the loss
                logits, loss = self.model(inputs, labels)

                # Evaluate the model performance
                self.evaluator.evaluate_finetune(self.model, epoch, i, loss.item())

                # Backward pass and update the model
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        print('Save the finetuned model...')
        # Save model to file
        torch.save(self.model, 'model_finetune.pth')

    def align(self, epochs):
        # Load an extra finetuned model as reference model. The reference model is fixed during alignment.
        reference_model = torch.load('model_finetune.pth')
        # Reset the optimizer
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        # A hyperparameter to control the strength of the alignment loss, larger beta means stronger alignment
        beta = 0.1
        for epoch in range(epochs):
            # Reset the evaluator to clear the loss history for each epoch
            self.evaluator.reset()
            for i, (positive_inputs, positive_labels, negative_inputs, negative_labels) in enumerate(self.dataset.get_batch_generator_alignment('train')):
                # Forward pass the positive and negative samples on model and reference model
                positive_logits, positive_loss = self.model(positive_inputs, positive_labels, False)
                negative_logits, negative_loss = self.model(negative_inputs, negative_labels, False)
                with torch.no_grad():
                    reference_positive_logits, reference_positive_loss = reference_model(positive_inputs, positive_labels, False)
                    reference_negative_logits, reference_negative_loss = reference_model(negative_inputs, negative_labels, False)

                # Implement the DPO(Direct Preference Optimiazation) loss
                positive_distance = (positive_loss - reference_positive_loss)
                negative_distance = (negative_loss - reference_negative_loss)
                reward_margin = negative_distance - positive_distance
                loss = - F.logsigmoid(beta * reward_margin).mean()
                reward_margin = reward_margin.mean()

                # Evaluate the model every evaluation_interval iterations
                self.evaluator.evaluate_alignment(self.model, epoch, i, loss.item(), reward_margin.item(), reference_model)

                # Backward pass and update the model
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()