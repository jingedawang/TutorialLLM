import os
import sys
import warnings
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from dataset import Dataset
from evaluator import Evaluator
from model import TutorialLLM
from trainer import Trainer


def test_forward_reduce_loss_false_returns_unreduced_loss_without_warning():
    """
    Test the unreduced loss path used by DPO without relying on dataset loading.
    """
    torch.manual_seed(2024)
    model = TutorialLLM(vocabulary_size=17, dim_embed=8, max_length=4, num_head=1, num_layer=1, device='cpu')
    token_ids = torch.randint(0, 17, (2, 4))
    labels = torch.randint(0, 17, (2, 4))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        logits, loss = model(token_ids, labels, reduce_loss=False)

    assert logits.shape == (8, 17)
    assert loss.shape == (8,)
    assert not any('size_average and reduce args will be deprecated' in str(w.message) for w in caught)



def test_run():
    """
    Test the overal pipeline runs without error.
    """
    batch_size = 1
    max_length = 32
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(2024)
    dataset = Dataset('data.json', batch_size, max_length, device)
    dataset.finetune_train_data = dataset.finetune_train_data[:10]
    dataset.finetune_evaluate_data = dataset.finetune_evaluate_data[:10]
    dataset.alignment_train_data = dataset.alignment_train_data[:10]
    dataset.alignment_evaluate_data = dataset.alignment_evaluate_data[:10]

    dim_embedding = 8
    num_head = 1
    num_layer = 2
    model = TutorialLLM(dataset.vocabulary_size, dim_embedding, max_length, num_head, num_layer, device)
    model.train()
    model.to(device)
    iterations_to_evaluate_pretrain = 10
    interval_to_evaluate_pretrain = 10
    interval_to_evaluate_finetune = 10
    interval_to_evaluate_alignment = 10
    evaluator = Evaluator(dataset, device, iterations_to_evaluate_pretrain, interval_to_evaluate_pretrain, interval_to_evaluate_finetune, interval_to_evaluate_alignment)
    trainer = Trainer(model, dataset, evaluator, device)

    iterations_for_pretrain = 10
    trainer.pretrain(iterations_for_pretrain)

    epochs_for_finetune = 1
    trainer.finetune(epochs_for_finetune)

    epochs_for_alignment = 1
    trainer.align(epochs_for_alignment)