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

    def pretrain(self, iterations, interval_to_evaluate, iterations_to_evaluate):
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
            self.evaluator.evaluate_pretrain(i, loss)

            # Backward pass and update the model
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        print('Save the pretrained model...')
        # Save model to file
        torch.save(self.model, 'model_pretrain.pth')

    def finetune(self, epochs, interval_to_evaluate):
        # Reset the optimizer
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        
        for epoch in range(epochs):
            # Reset the evaluator to clear the loss history for each epoch
            self.evaluator.reset()
            for i, (inputs, labels) in enumerate(self.dataset.get_batch_generator_finetune('train')):
                # Forward pass and calculate the loss
                logits, loss = self.model(inputs, labels)

                # Evaluate the model performance
                self.evaluator.evaluate_finetune(epoch, i, loss)

                # Backward pass and update the model
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        print('Save the finetuned model...')
        # Save model to file
        torch.save(self.model, 'model_finetune.pth')

    def align(self, epochs, interval_to_evaluate):
        # Clone the model as a reference model and switch it to evaluation mode
        reference_model = self.model.copy().eval()
        # Reset the optimizer
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        # A hyperparameter to control the strength of the alignment loss, larger beta means stronger alignment
        beta = 0.1
        for epoch in range(epochs):
            loss_sum = positive_distance_sum = negative_distance_sum = math.nan
            for i, (positive_inputs, positive_labels, negative_inputs, negative_labels) in enumerate(self.dataset.get_batch_generator_alignment('train')):
                # Evaluate the model every evaluation_interval iterations
                if i % interval_to_evaluate == 0:
                    # Calculate the average loss for this interval
                    mean_loss_train = loss_sum / interval_to_evaluate
                    mean_reward_margin = (positive_distance_sum - negative_distance_sum) / interval_to_evaluate
                    loss_sum = positive_distance_sum = negative_distance_sum = 0
                    loss = self.model.estimate_loss_eval(self.dataset, 'alignment')
                    print(f"Epoch {epoch}, step {i}, train loss {mean_loss_train:.4f}, evaluate loss {loss:.4f}")

                # Forward pass the positive and negative samples on model and reference model
                positive_logits, positive_loss = self.model(positive_inputs, positive_labels, False)
                negative_logits, negative_loss = self.model(negative_inputs, negative_labels, False)
                reference_positive_logits, reference_positive_loss = reference_model(positive_inputs, positive_labels)
                reference_negative_logits, reference_negative_loss = reference_model(negative_inputs, negative_labels)

                # Implement the DPO(Direct Preference Optimiazation) loss
                loss = (positive_loss - reference_positive_loss) - (negative_loss - reference_negative_loss)
                loss = F.logsigmoid(beta * loss).mean()
                loss_sum += loss.item()

                # Calculate the distance between the model and the reference model
                positive_distance = (positive_loss - reference_positive_loss).mean().detach()
                negative_distance = (negative_loss - reference_negative_loss).mean().detach()
                positive_distance_sum += positive_distance
                negative_distance_sum += negative_distance

                # Backward pass and update the model
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()