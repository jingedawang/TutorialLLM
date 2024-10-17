import torch

from dataset import Dataset
from evaluator import Evaluator
from model import TutorialLLM
from trainer import Trainer


print(f'{"-"*50}\nSTAGE 1: PREPARE THE DATA')
# The number of parallel items to process, known as the batch size
batch_size = 16
# The maximum length of a text to be processed
max_length = 256
# Run the model on GPU(cuda) if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# Set a seed for reproducibility
torch.manual_seed(2024)
dataset = Dataset('data.json', batch_size, max_length, device)
print('Check a batch of pretrain data:')
print(dataset.get_batch_pretrain('train'))
print('Check a batch of finetune data:')
print(next(dataset.get_batch_generator_finetune('train')))

print(f'{"-"*50}\nSTAGE 2: TRAINING CONFIGURATION')
# The dimension of the embedding vector in the transformer
dim_embedding = 64
# The number of heads in the multi-head attention
num_head = 4
# The number of layers in the transformer
num_layer = 4
# Create a TutorialLLM instance
model = TutorialLLM(dataset.vocabulary_size, dim_embedding, max_length, num_head, num_layer, device)
# Switch the model to training mode
model.train()
model.to(device)
# Show the model size
print(f'Our model has {sum(parameter.numel() for parameter in model.parameters())/1e6} M parameters')
# The number of iterations to evaluate the pretrain process (each iteration processes a batch)
iterations_to_evaluate_pretrain = 100
# The interval of iterations to evaluate the pretrain process
interval_to_evaluate_pretrain = 50
# The interval of iterations to evaluate the finetune process
interval_to_evaluate_finetune = 50
# The interval of iterations to evaluate the alignment process
interval_to_evaluate_alignment = 50
# Create an Evaluator instance to evaluate the performance during training
evaluator = Evaluator(dataset, device, iterations_to_evaluate_pretrain, interval_to_evaluate_pretrain, interval_to_evaluate_finetune, interval_to_evaluate_alignment)
# Create a Trainer instance for susequent training
trainer = Trainer(model, dataset, evaluator, device)

print(f'{"-"*50}\nSTAGE 3: PRETRAIN')
# The number of iterations for pretrain (each iteration processes a batch)
iterations_for_pretrain = 5000
# Pretrain the model
trainer.pretrain(iterations_for_pretrain)

print(f'{"-"*50}\nSTAGE 4: FINETUNE')
# The number of epochs to finetune the model
epochs_for_finetune = 10
# Finetune the model
trainer.finetune(epochs_for_finetune)

print(f'{"-"*50}\nSTAGE 5: ALIGN PREFERENCE')
# The number of epochs to align the model
epochs_for_alignment = 5
# Align the model with human preference
trainer.align(epochs_for_alignment)