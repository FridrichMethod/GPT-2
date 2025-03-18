"""Paraphrase detection for GPT starter code.

Consider:
 - ParaphraseGPT: Your implementation of the GPT-2 classification model.
 - train: Training procedure for ParaphraseGPT on the Quora paraphrase detection dataset.
 - test: Test procedure. This function generates the required files for your submission.

Running:
    >>> python paraphrase_detection.py --use_gpu
trains and evaluates your ParaphraseGPT model and writes the required submission files.
"""

import argparse
import random
import time
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

# from torch.optim import AdamW
from torch.quantization import (
    FakeQuantize,
    MinMaxObserver,
    QConfig,
    default_weight_observer,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import (
    ParaphraseDetectionDataset,
    ParaphraseDetectionTestDataset,
    load_paraphrase_data,
)
from evaluation import model_eval_paraphrase, model_test_paraphrase
from models.gpt2 import GPT2Model
from optimizer import AdamW

TQDM_DISABLE = False


# Fix the random seed.
def seed_everything(seed=11711):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_custom_qconfig(bit_width: int) -> QConfig:
    """Get a custom QConfig for quantization-aware training (QAT) with the specified bit width."""
    if bit_width != 8:
        raise ValueError("Only 8-bit quantization is supported in this example.")
    quant_min = -(2 ** (bit_width - 1))
    quant_max = 2 ** (bit_width - 1) - 1

    act_fake_quant = FakeQuantize.with_args(
        observer=MinMaxObserver,
        dtype=torch.qint8,
        quant_min=quant_min,
        quant_max=quant_max,
    )

    weight_fake_quant = FakeQuantize.with_args(
        observer=default_weight_observer,
        dtype=torch.qint8,
        quant_min=quant_min,
        quant_max=quant_max,
    )

    return QConfig(activation=act_fake_quant, weight=weight_fake_quant)


def prepare_model_for_qat(model: nn.Module, qconfig: Optional[QConfig] = None) -> None:
    """Prepare the given model for quantization-aware training (QAT).

    This sets the qconfig and inserts fake quantization modules.
    """

    # Here we use the default QAT configuration with 'fbgemm'
    if qconfig is None:
        model.qconfig = torch.quantization.get_default_qat_qconfig("fbgemm")
    else:
        model.qconfig = qconfig

    for module in model.modules():
        if isinstance(module, nn.Embedding):
            module.qconfig = None

    # Prepare the model in-place for QAT
    torch.quantization.prepare_qat(model, inplace=True)
    print("Model prepared for quantization-aware training.")


class ParaphraseGPT(nn.Module):
    """Your GPT-2 Model designed for paraphrase detection."""

    def __init__(self, args):
        super().__init__()
        if args.use_lora:
            self.gpt = GPT2Model.from_pretrained(
                model=args.model_size,
                d=args.d,
                l=args.l,
                num_heads=args.num_heads,
                use_lora=args.use_lora,
                lora_r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
            )
        else:
            self.gpt = GPT2Model.from_pretrained(
                model=args.model_size,
                d=args.d,
                l=args.l,
                num_heads=args.num_heads,
                use_lora=args.use_lora,
            )

        if args.use_lora:
            for name, param in self.gpt.named_parameters():
                if "lora_" not in name:
                    param.requires_grad = False
        else:
            for param in self.gpt.parameters():
                param.requires_grad = True

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Predict the label of the token using the paraphrase_detection_head Linear layer.

        We structure the input as:

          'Is "{s1}" a paraphrase of "{s2}"? Answer "yes" or "no": '

        So you want to find the prediction for the next token at the end of this sentence. Optimistically, it will be the
        token "yes" (byte pair encoding index of 8505) for examples that are paraphrases or "no" (byte pair encoding index
        of 3919) for examples that are not paraphrases.

        Takes a batch of sentences and produces embeddings for them.
        """

        last_token = self.gpt(input_ids, attention_mask)["last_token"]
        logits = self.gpt.hidden_state_to_token(last_token)

        return logits


def save_model(
    model: ParaphraseGPT, optimizer: AdamW, args: argparse.Namespace, filepath: str
) -> None:
    save_info = {
        "model": model.state_dict(),
        "optim": optimizer.state_dict(),
        "args": args,
        "system_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "torch_rng": torch.random.get_rng_state(),
    }

    torch.save(save_info, filepath)
    print(f"save the model to {filepath}")


def train(args: argparse.Namespace) -> None:
    """Train GPT-2 for paraphrase detection on the Quora dataset."""
    device = torch.device("cuda") if args.use_gpu else torch.device("cpu")
    # Create the data and its corresponding datasets and dataloader.
    para_train_data = load_paraphrase_data(args.para_train)
    para_dev_data = load_paraphrase_data(args.para_dev)

    para_train_data = ParaphraseDetectionDataset(para_train_data, args)
    para_dev_data = ParaphraseDetectionDataset(para_dev_data, args)

    para_train_dataloader = DataLoader(
        para_train_data,
        shuffle=True,
        batch_size=args.batch_size,
        collate_fn=para_train_data.collate_fn,
    )
    para_dev_dataloader = DataLoader(
        para_dev_data,
        shuffle=False,
        batch_size=args.batch_size,
        collate_fn=para_dev_data.collate_fn,
    )

    args = add_arguments(args)
    model = ParaphraseGPT(args)
    model = model.to(device)

    lr = args.lr
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.0, eps=1e-6)
    best_dev_acc = 0

    # Run for the specified number of epochs.
    for epoch in range(args.epochs):
        model.train()
        if args.use_quantization:
            print("Enabling quantization-aware training for GPT-2 backbone.")
            if args.bit_width is None:
                prepare_model_for_qat(model)
            else:
                prepare_model_for_qat(model, get_custom_qconfig(args.bit_width))
        train_loss = 0
        num_batches = 0
        t0 = time.perf_counter()
        for batch in tqdm(
            para_train_dataloader,
            desc=f"train-{epoch}",
            disable=TQDM_DISABLE,
        ):
            # Get the input and move it to the gpu (I do not recommend training this model on CPU).
            b_ids, b_mask, labels = (
                batch["token_ids"],
                batch["attention_mask"],
                batch["labels"].flatten(),
            )
            b_ids = b_ids.to(device)
            b_mask = b_mask.to(device)
            labels = labels.to(device)

            # Compute the loss, gradients, and update the model's parameters.
            optimizer.zero_grad()
            logits = model(b_ids, b_mask)
            loss = F.cross_entropy(logits, labels, reduction="mean")
            loss.backward()
            optimizer.step()

            dt = time.perf_counter() - t0

            train_loss += loss.item()
            num_batches += 1

            if num_batches % 50 == 0:
                print(f"Estimate speed: {num_batches / dt:.4f} it/s", flush=True)
                print(
                    f"Batch memory allocated: {torch.cuda.memory_allocated()/1024**2:.2f} MB",
                    flush=True,
                )
                print(
                    f"Batch memory reserved: {torch.cuda.memory_reserved()/1024**2:.2f} MB",
                    flush=True,
                )

        train_loss = train_loss / num_batches

        model.eval()
        if args.use_quantization:
            torch.quantization.convert(model, inplace=True)
        dev_acc, *_ = model_eval_paraphrase(para_dev_dataloader, model, device)

        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            save_model(model, optimizer, args, args.filepath)

        print(
            f"Epoch {epoch}: train loss :: {train_loss :.3f}, dev acc :: {dev_acc :.3f}"
        )


@torch.no_grad()
def test(args: argparse.Namespace) -> None:
    """Evaluate your model on the dev and test datasets; save the predictions to disk."""
    device = torch.device("cuda") if args.use_gpu else torch.device("cpu")
    saved = torch.load(args.filepath, weights_only=False)

    model = ParaphraseGPT(saved["args"])
    model.load_state_dict(saved["model"])
    model = model.to(device)
    model.eval()
    if args.use_quantization:
        torch.quantization.convert(model, inplace=True)
    print(f"Loaded model to test from {args.filepath}")

    para_dev_data = load_paraphrase_data(args.para_dev)
    para_test_data = load_paraphrase_data(args.para_test, split="test")

    para_dev_data = ParaphraseDetectionDataset(para_dev_data, args)
    para_test_data = ParaphraseDetectionTestDataset(para_test_data, args)

    para_dev_dataloader = DataLoader(
        para_dev_data,
        shuffle=False,
        batch_size=args.batch_size,
        collate_fn=para_dev_data.collate_fn,
    )
    para_test_dataloader = DataLoader(
        para_test_data,
        shuffle=True,
        batch_size=args.batch_size,
        collate_fn=para_test_data.collate_fn,
    )

    dev_para_acc, _, dev_para_y_pred, _, dev_para_sent_ids = model_eval_paraphrase(
        para_dev_dataloader, model, device
    )
    print(f"dev paraphrase acc :: {dev_para_acc :.3f}")
    test_para_y_pred, test_para_sent_ids = model_test_paraphrase(
        para_test_dataloader, model, device
    )

    with open(args.para_dev_out, "w+") as f:
        f.write(f"id \t Predicted_Is_Paraphrase \n")
        for p, s in zip(dev_para_sent_ids, dev_para_y_pred):
            f.write(f"{p}, {s} \n")

    with open(args.para_test_out, "w+") as f:
        f.write(f"id \t Predicted_Is_Paraphrase \n")
        for p, s in zip(test_para_sent_ids, test_para_y_pred):
            f.write(f"{p}, {s} \n")


def get_args() -> argparse.Namespace:
    """Get the arguments for the script."""
    parser = argparse.ArgumentParser()

    parser.add_argument("--para_train", type=str, default="data/quora-train.csv")
    parser.add_argument("--para_dev", type=str, default="data/quora-dev.csv")
    parser.add_argument("--para_test", type=str, default="data/quora-test-student.csv")
    parser.add_argument(
        "--para_dev_out", type=str, default="predictions/para-dev-output.csv"
    )
    parser.add_argument(
        "--para_test_out", type=str, default="predictions/para-test-output.csv"
    )

    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--use_gpu", action="store_true")

    parser.add_argument(
        "--use_lora",
        action="store_true",
        help="Use LoRA for parameter-efficient fine-tuning.",
    )

    parser.add_argument(
        "--batch_size",
        help="sst: 64, cfimdb: 8 can fit a 12GB GPU",
        type=int,
        default=8,
    )
    parser.add_argument("--lr", type=float, help="learning rate", default=1e-5)
    parser.add_argument(
        "--model_size",
        type=str,
        help="The model size as specified on hugging face. DO NOT use the xl model.",
        choices=["gpt2", "gpt2-medium", "gpt2-large"],
        default="gpt2",
    )

    parser.add_argument(
        "--use_quantization",
        action="store_true",
        help="Enable quantization-aware training (QAT)",
    )

    args = parser.parse_args()

    return args


def add_arguments(args: argparse.Namespace) -> argparse.Namespace:
    """Add arguments that are deterministic on model size."""
    if args.model_size == "gpt2":
        args.d = 768
        args.l = 12
        args.num_heads = 12
    elif args.model_size == "gpt2-medium":
        args.d = 1024
        args.l = 24
        args.num_heads = 16
    elif args.model_size == "gpt2-large":
        args.d = 1280
        args.l = 36
        args.num_heads = 20
    else:
        raise ValueError(f"{args.model_size} is not supported.")

    if args.use_lora:
        args.lora_r = 8
        args.lora_alpha = 32
        args.lora_dropout = 0.1

    if args.use_quantization:
        args.bit_width = 8

    return args


if __name__ == "__main__":
    args = get_args()
    args.filepath = (
        f"{args.model_size}-{args.epochs}-{args.lr}-paraphrase.pt"  # Save path.
    )
    seed_everything(args.seed)  # Fix the seed for reproducibility.
    train(args)
    test(args)
