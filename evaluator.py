import torch

from dataset import Dataset
from model import DpoWrapper, TutorialLLM

class Evaluator():
    """
    Evaluator for the model.

    This module provides methods to evaluate the model performance during training.
    """

    def __init__(self, dataset: Dataset, device: str, iterations_to_evaluate_pretrain: int, interval_to_evaluate_pretrain: int, interval_to_evaluate_finetune: int, interval_to_evaluate_alignment: int) -> None:
        """
        Initialize the evaluator with the dataset, device, and evaluation intervals.

        Args:
            dataset: The dataset to provide evaluation data.
            device: The device to run the model on ('cpu' or 'cuda').
            iterations_to_evaluate_pretrain: The number of iterations to evaluate the pretrain process.
            interval_to_evaluate_pretrain: The interval of iterations to evaluate the pretrain process.
            interval_to_evaluate_finetune: The interval of iterations to evaluate the finetune process.
            interval_to_evaluate_alignment: The interval of iterations to evaluate the alignment process.
        """
        self.dataset = dataset
        self.device = device
        self.iterations_to_evaluate_pretrain = iterations_to_evaluate_pretrain
        self.interval_to_evaluate_pretrain = interval_to_evaluate_pretrain
        self.interval_to_evaluate_finetune = interval_to_evaluate_finetune
        self.interval_to_evaluate_alignment = interval_to_evaluate_alignment

        self.test_input = '<INS>請用以下題目寫一首詩<INP>春夜喜雨<RES>'

        self.reset()
    
    def reset(self) -> None:
        """
        Reset the loss and reward margin accumulators.
        """
        self.train_loss_sum = 0
        self.train_reward_margin_sum = 0

    @torch.inference_mode()
    def evaluate_pretrain(self, model: TutorialLLM, iteration: int, train_loss: float) -> None:
        """
        Evaluate the model performance during the pretrain process.

        This method should be called every iteration during training.
        The train loss and evaluate loss will be printed every `interval_to_evaluate_pretrain` iterations.
        A poem starting with the title "春夜喜雨" will be generated to see how the model is doing.

        Args:
            model: The model to evaluate.
            iteration: The current iteration number.
            train_loss: The training loss of the current iteration.
        """
        if iteration % self.interval_to_evaluate_pretrain == 0:
            # Get average train loss and evaluate loss
            mean_loss_train = self.train_loss_sum / self.interval_to_evaluate_pretrain
            self.reset()
            evaluate_loss = self.evaluate_pretrain_loss(model, self.iterations_to_evaluate_pretrain)
            print(f"Step {iteration}, train loss {mean_loss_train:.4f}, evaluate loss {evaluate_loss:.4f}")

            # Let's generate a poem starting with the title '春夜喜雨' to see how the model is doing
            test_tokens = torch.tensor(self.dataset.encode('春夜喜雨'), dtype=torch.long, device=self.device).unsqueeze(0)
            print('Generate first 100 characters of poems starting with 春夜喜雨:')
            print(self.dataset.decode(model.generate(test_tokens, max_new_tokens=100)[0].tolist()))
        
        # Accumulate the training loss
        self.train_loss_sum += train_loss
    
    @torch.inference_mode()
    def evaluate_pretrain_loss(self, model: TutorialLLM, iterations: int) -> float:
        """
        Evaluate the model loss during the pretrain process.

        Args:
            model: The model to evaluate.
            iterations: The number of iterations to evaluate the model.

        Returns:
            The average loss of the model in the evaluation.
        """
        losses = torch.zeros(iterations)
        # Evaluate the model `iterations` times
        for k in range(iterations):
            # Get a batch of pretrain data and compute the loss
            inputs, labels = self.dataset.get_batch_pretrain('evaluate')
            _, loss = model(inputs, labels)
            losses[k] = loss.item()
        loss = losses.mean()
        return loss
    
    @torch.inference_mode()
    def evaluate_finetune(self, model: TutorialLLM, epoch: int, iteration: int, train_loss: float) -> None:
        """
        Evaluate the model performance during the finetune process.

        This method should be called every iteration during training.
        The train loss and evaluate loss will be printed every `interval_to_evaluate_finetune` iterations.
        A poem starting with the title "春夜喜雨" will be generated to see how the model is doing.

        Args:
            model: The model to evaluate.
            epoch: The current epoch number.
            iteration: The current iteration number.
            train_loss: The training loss of the current iteration.
        """
        if iteration % self.interval_to_evaluate_finetune == 0:
            # Get average train loss and evaluate loss
            mean_loss_train = self.train_loss_sum / self.interval_to_evaluate_finetune
            self.reset()
            evaluate_loss = self.evaluate_finetune_loss(model)
            print(f"Epoch {epoch}, step {iteration}, train loss {mean_loss_train:.4f}, evaluate loss {evaluate_loss:.4f}")

            # Let's generate a poem with a given title to see how the model is doing
            test_tokens = torch.tensor(self.dataset.encode(self.test_input), dtype=torch.long, device=self.device).unsqueeze(0)
            output = self.dataset.decode(model.generate(test_tokens, max_new_tokens=100)[0].tolist())
            # Truncate the output to the end-of-text character '\0'
            output = output[:output.find('\0')]
            print('Generate a complete poem for title 春夜喜雨:')
            print(output[len(self.test_input):])
        
        # Accumulate the training loss
        self.train_loss_sum += train_loss

    @torch.inference_mode()
    def evaluate_finetune_loss(self, model: TutorialLLM) -> float:
        """
        Evaluate the model loss during the finetune process.

        Args:
            model: The model to evaluate.

        Returns:
            The average loss of the model in the evaluation.
        """
        loss_sum = 0
        # Get a batch generator of finetune data
        batch_generator = self.dataset.get_batch_generator_finetune('evaluate')
        # Evaluate the model by processing all batches generated by the generator
        for k, batch in enumerate(batch_generator):
            inputs, labels = batch
            _, loss = model(inputs, labels)
            loss_sum += loss.item()
        loss = loss_sum / (k + 1)
        return loss

    @torch.inference_mode()
    def evaluate_alignment(self, dpo_wrapper: DpoWrapper, epoch: int, iteration: int, train_loss: float, train_reward_margin: float) -> None:
        """
        Evaluate the model performance during the alignment process.

        This method should be called every iteration during training.
        The train loss and evaluate loss will be printed every `interval_to_evaluate_alignment` iterations.
        Poems generated by the aligned model and the reference model will be printed to compare the two models.

        Args:
            dpo_wrapper: The DPO wrapper to evaluate.
            epoch: The current epoch number.
            iteration: The current iteration number.
            train_loss: The training loss of the current iteration.
            train_reward_margin: The training reward margin of the current iteration.
        """
        if iteration % self.interval_to_evaluate_alignment == 0:
            # Calculate the average loss for this interval
            mean_loss_train = self.train_loss_sum / self.interval_to_evaluate_alignment
            mean_reward_margin_train = self.train_reward_margin_sum / self.interval_to_evaluate_alignment
            self.reset()
            evaluate_loss, evaluate_reward_margin = self.evaluate_alignment_loss(dpo_wrapper)
            print(f"Epoch {epoch}, step {iteration}, train loss {mean_loss_train:.4f}, evaluate loss {evaluate_loss:.4f}, train reward margin {mean_reward_margin_train:.4f}, evaluate reward margin {evaluate_reward_margin:.4f}")

            # Let's ask the two models to generate a poem respectively
            test_tokens = torch.tensor(self.dataset.encode(self.test_input), dtype=torch.long, device=self.device).unsqueeze(0)
            aligned_output = self.dataset.decode(dpo_wrapper.aligned_model.generate(test_tokens, max_new_tokens=100)[0].tolist())
            reference_output = self.dataset.decode(dpo_wrapper.reference_model.generate(test_tokens, max_new_tokens=100)[0].tolist())
            # Truncate the output to the end-of-text character '\0'
            aligned_output = aligned_output[:aligned_output.find('\0')]
            reference_output = reference_output[:reference_output.find('\0')]
            print('Generate a complete poem for title 春夜喜雨:')
            print('Aligned model:')
            print(aligned_output[len(self.test_input):])
            print('Reference model:')
            print(reference_output[len(self.test_input):])

        # Accumulate the training loss and reward margin
        self.train_loss_sum += train_loss
        self.train_reward_margin_sum += train_reward_margin

    @torch.inference_mode()
    def evaluate_alignment_loss(self, dpo_wrapper: DpoWrapper) -> tuple[float, float]:
        """
        Evaluate the model loss during the alignment process.

        Args:
            dpo_wrapper: The DPO wrapper to evaluate.

        Returns:
            The average loss and reward margin of the model in the evaluation.
        """
        loss_sum = 0
        reward_margin_sum = 0
        # Get a batch generator of alignment data
        batch_generator = self.dataset.get_batch_generator_alignment('evaluate')
        # Evaluate the model by processing all batches generated by the generator
        for k, (positive_inputs, positive_labels, negative_inputs, negative_labels) in enumerate(batch_generator):
            loss, reward_margin = dpo_wrapper.forward(positive_inputs, positive_labels, negative_inputs, negative_labels)
            loss_sum += loss.item()
            reward_margin_sum += reward_margin.item()
        loss = loss_sum / (k + 1)
        reward_margin = reward_margin_sum / (k + 1)
        return loss, reward_margin