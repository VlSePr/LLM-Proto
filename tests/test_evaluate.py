import torch
import torch.nn.functional as F

from src.evaluate import compute_val_metrics
from src.model import IGNORE_INDEX, TransformerLM


def test_val_loss_is_token_weighted(tiny_cfg, seed):
    model = TransformerLM(tiny_cfg).eval()
    b1 = torch.randint(0, tiny_cfg.vocab_size, (2, 8))
    t1 = b1.clone()
    t1[0, 5:] = IGNORE_INDEX          # batch 1: 13 valid tokens
    b2 = torch.randint(0, tiny_cfg.vocab_size, (1, 8))
    t2 = b2.clone()                                      # batch 2: 8 valid tokens
    loader = [(b1, t1), (b2, t2)]

    metrics = compute_val_metrics(model, loader, torch.device("cpu"), max_batches=10)

    with torch.no_grad():
        l1 = model(b1)["logits"].reshape(-1, tiny_cfg.vocab_size)
        l2 = model(b2)["logits"].reshape(-1, tiny_cfg.vocab_size)
        total = F.cross_entropy(l1, t1.reshape(-1), ignore_index=IGNORE_INDEX, reduction="sum") \
              + F.cross_entropy(l2, t2.reshape(-1), ignore_index=IGNORE_INDEX, reduction="sum")
    assert metrics["tokens_evaluated"] == 21
    assert metrics["batches"] == 2
    assert abs(metrics["loss"] - total.item() / 21) < 1e-5


def test_max_batches_and_train_mode_restored(tiny_cfg, seed):
    model = TransformerLM(tiny_cfg).train()
    b = torch.randint(0, tiny_cfg.vocab_size, (1, 8))
    loader = [(b, b)] * 5
    metrics = compute_val_metrics(model, loader, torch.device("cpu"), max_batches=2)
    assert metrics["batches"] == 2
    assert model.training
