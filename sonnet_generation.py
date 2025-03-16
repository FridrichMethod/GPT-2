"""Sonnet generation starter code.

Running:
    >>> python sonnet_generation.py --use_gpu [--use_lora]

trains your SonnetGPT model and writes the required submission files.
"""

import argparse
import random

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from datasets import SonnetsDataset
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


class SonnetGPT(nn.Module):
    """Your GPT-2 Model designed for paraphrase detection."""

    def __init__(self, args: argparse.Namespace) -> None:
        """Initialize the SonnetGPT model."""
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
        # self.gpt = GPT2LMHeadModel.from_pretrained(args.model_size)

        # Initialize the tokenizer.
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token

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
        """Computes the logits for every token in the sequence.

        This is similar to the forward for ParaphraseGPT, but we now want to produce a logit for each token in our sequence;
        not just the last token! This will allow our model to learn the natural language distribution that composes sonnets,
        not just the distribution over next tokens for the last token!
        """

        # Pass inputs through GPT2 to obtain hidden states.
        outputs = self.gpt(input_ids, attention_mask)
        sequence_output = outputs[
            "last_hidden_state"
        ]  # shape: [batch_size, seq_len, hidden_size]

        # Convert hidden states to token logits using weight tying.
        logits = self.gpt.hidden_state_to_token(
            sequence_output
        )  # shape: [batch_size, seq_len, vocab_size]
        return logits

    @property
    def device(self):
        """Returns the device of the model parameters."""
        return next(self.gpt.parameters()).device

    @torch.no_grad()
    def generate(
        self,
        encoding: torch.Tensor,
        beam_search: bool = False,
        **kwargs,
    ) -> str:
        """Generates an original sonnet using either nucleus sampling or beam search."""

        if beam_search:
            return self.generate_beam_search(encoding, **kwargs)
        else:
            return self.generate_nucleus(encoding, **kwargs)

    @torch.no_grad()
    def generate_nucleus(
        self,
        encoding: torch.Tensor,
        temperature: float = 1.2,
        top_p: float = 0.9,
        max_length: int = 128,
        **kwargs,
    ) -> str:
        """Generates an original sonnet using top-p sampling and softmax temperature."""

        token_ids = encoding.to(self.device)

        for _ in range(max_length):
            attention_mask = torch.ones_like(
                token_ids, dtype=torch.int64, device=self.device
            )
            # Forward pass to get logits
            logits_sequence = self.forward(token_ids, attention_mask)
            logits_last_token = (
                logits_sequence[:, -1, :] / temperature
            )  # Apply temperature scaling

            # Convert logits to probabilities
            probs = F.softmax(logits_last_token, dim=-1)

            # Top-p (nucleus) sampling
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            top_p_mask = cumulative_probs <= top_p
            top_p_mask[..., 1:] = top_p_mask[
                ..., :-1
            ].clone()  # Shift mask right for proper thresholding
            top_p_mask[..., 0] = True  # Always include the highest probability token
            filtered_probs = sorted_probs * top_p_mask  # Zero out unlikely tokens
            filtered_probs /= filtered_probs.sum(
                dim=-1, keepdim=True
            )  # Normalize probabilities

            # Sample from filtered distribution
            sampled_index = torch.multinomial(filtered_probs, 1)
            sampled_token = torch.gather(sorted_indices, -1, sampled_index)

            # Stop if end-of-sequence token is reached (do not include eos in the output)
            if sampled_token.item() == self.tokenizer.eos_token_id:
                break

            # Append sampled token
            token_ids = torch.cat([token_ids, sampled_token], dim=1)

        decoded_output = self.tokenizer.decode(token_ids[0], skip_special_tokens=True)
        full_sonnet = f"{decoded_output}\n\n"

        return full_sonnet

    @torch.no_grad()
    def generate_beam_search(
        self,
        encoding: torch.Tensor,
        temperature: float = 1.2,
        top_p: float = 0.9,
        max_length: int = 128,
        beam_width: int = 3,
        length_penalty: float = 1.1,
        repetition_penalty: float = 1.1,
        **kwargs,
    ) -> str:
        """Generates an original sonnet using beam search with nucleus filtering."""

        # Beam search with nucleus filtering.
        initial_ids = encoding.to(self.device)
        # Each beam is a tuple: (cumulative_log_prob, token_ids)
        beams = [(0.0, initial_ids)]
        completed_beams = []

        for _ in range(max_length):
            new_beams = []
            for score, seq in beams:
                attention_mask = torch.ones_like(
                    seq, dtype=torch.int64, device=self.device
                )
                logits_sequence = self.forward(seq, attention_mask)
                logits_last_token = logits_sequence[:, -1, :] / temperature
                log_probs = F.log_softmax(
                    logits_last_token, dim=-1
                )  # shape: [1, vocab_size]

                # Sort tokens and apply nucleus filtering.
                sorted_log_probs, sorted_indices = torch.sort(
                    log_probs, descending=True
                )
                cumulative_probs = torch.cumsum(sorted_log_probs.exp(), dim=-1)
                nucleus_mask = cumulative_probs <= top_p
                nucleus_mask[..., 1:] = nucleus_mask[..., :-1].clone()
                nucleus_mask[..., 0] = True
                # Mask tokens outside the nucleus.
                filtered_log_probs = sorted_log_probs.masked_fill(
                    ~nucleus_mask, -float("inf")
                )

                # Apply repetition penalty: modify log_probs for tokens already generated in this beam.
                for token in seq[0]:
                    filtered_log_probs[
                        :, sorted_indices[0] == token
                    ] *= repetition_penalty

                # Convert filtered log probabilities back to probabilities.
                filtered_log_probs = F.log_softmax(filtered_log_probs, dim=-1)
                # Sample beam_width tokens using multinomial sampling.
                sampled_indices_in_sorted = torch.multinomial(
                    filtered_log_probs.exp(), beam_width, replacement=False
                )
                sampled_log_probs = torch.gather(
                    filtered_log_probs, -1, sampled_indices_in_sorted
                )
                sampled_token_ids = torch.gather(
                    sorted_indices, -1, sampled_indices_in_sorted
                )

                # Expand the current beam for each candidate.
                for i in range(beam_width):
                    # Only keep beams with valid scores.
                    token_score = sampled_log_probs[0, i].item()
                    if token_score == -float("inf"):
                        continue
                    # If EOS token is already generated, keep the beam as is.
                    token_id = sampled_token_ids[0, i].view(1, 1)
                    if token_id.item() == self.tokenizer.eos_token_id:
                        completed_beams.append((score, seq))
                        continue
                    new_seq = torch.cat([seq, token_id], dim=1)
                    new_score = score + token_score
                    new_beams.append((new_score, new_seq))

            if not new_beams:
                break

            # Keep only the best beam_width candidates.
            beams = sorted(
                new_beams,
                key=lambda x: x[0] / (x[1].size(1) ** length_penalty),
                reverse=True,
            )[:beam_width]

        # Choose the best complete sequence if available.
        if completed_beams:
            best_seq = max(
                completed_beams, key=lambda x: x[0] / (x[1].size(1) ** length_penalty)
            )[1]
        else:
            best_seq = max(
                beams, key=lambda x: x[0] / (x[1].size(1) ** length_penalty)
            )[1]

        decoded_output = self.tokenizer.decode(best_seq[0], skip_special_tokens=True)
        full_sonnet = f"{decoded_output}\n\n"

        return full_sonnet


def save_model(
    model: SonnetGPT, optimizer: AdamW, args: argparse.Namespace, filepath: str
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
    sonnet_dataset = SonnetsDataset(args.sonnet_path)
    sonnet_dataloader = DataLoader(
        sonnet_dataset,
        shuffle=True,
        batch_size=args.batch_size,
        collate_fn=sonnet_dataset.collate_fn,
    )

    # Create the held-out dataset: these only have the first 3 lines. Your job is to fill in the rest!
    # held_out_sonnet_dev_dataset = SonnetsDataset(args.held_out_sonnet_dev_path)

    args = add_arguments(args)
    model = SonnetGPT(args)
    model = model.to(device)

    lr = args.lr
    optimizer = AdamW(model.parameters(), lr=lr)

    # Early stopping parameters
    best_loss = float("inf")
    epochs_without_improvement = 0
    patience = (
        args.patience
    )  # Number of epochs to wait for improvement before stopping.
    min_delta = (
        args.min_delta
    )  # Minimum change in the loss to qualify as an improvement.

    # Run for the specified number of epochs.
    epoch = 0
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        num_batches = 0.0

        for batch in tqdm(
            sonnet_dataloader, desc=f"train-{epoch}", disable=TQDM_DISABLE
        ):
            # Get the input and move it to the gpu (I do not recommend training this model on CPU).
            b_ids, b_mask = batch["token_ids"], batch["attention_mask"]
            b_ids = b_ids.to(device)
            b_mask = b_mask.to(device)

            # Compute the loss, gradients, and update the model's parameters.
            optimizer.zero_grad()
            logits = model(b_ids, b_mask)
            logits = rearrange(
                logits[:, :-1].contiguous(), "b t d -> (b t) d"
            )  # Ignore the last prediction in the sequence.
            labels = (
                b_ids[:, 1:].contiguous().flatten()
            )  # Ignore the first token to compose the labels.
            loss = F.cross_entropy(logits, labels, reduction="mean")
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            num_batches += 1

        train_loss = train_loss / num_batches
        print(f"Epoch {epoch}: train loss :: {train_loss:.3f}.")

        # Early stopping check based on training loss improvement.
        if best_loss - train_loss > min_delta:
            best_loss = train_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            print(
                f"No significant improvement for {epochs_without_improvement} epoch(s)."
            )
            if epochs_without_improvement >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs.")
                break

        # Generate outputs on the held-out dataset (for qualitative monitoring).
        # model.eval()
        # print("Generating several output sonnets for dev...\n\n")
        # for batch in held_out_sonnet_dev_dataset:
        #     encoding = model.tokenizer(
        #         batch[1], return_tensors="pt", padding=True, truncation=True
        #     ).to(device)
        #     output = model.generate(
        #         encoding["input_ids"],
        #         beam_search=args.beam_search,
        #         temperature=args.temperature,
        #         top_p=args.top_p,
        #         max_length=128,
        #         beam_width=args.beam_width,
        #         length_penalty=args.length_penalty,
        #         repetition_penalty=args.repetition_penalty,
        #     )
        #     print(output)

    save_model(model, optimizer, args, f"{args.model_size}-{args.filepath}")
    print("\n\n")


@torch.no_grad()
def generate_submission_sonnets(args: argparse.Namespace) -> None:
    """Generate sonnets for the held-out dataset."""
    device = torch.device("cuda") if args.use_gpu else torch.device("cpu")

    # Load the model from the last epoch.
    saved = torch.load(f"{args.model_size}-{args.filepath}", weights_only=False)

    model = SonnetGPT(saved["args"])
    model.load_state_dict(saved["model"])
    model = model.to(device)
    model.eval()

    # Create the held-out dataset: these only have the first 3 lines. Your job is to fill in the rest!
    held_out_sonnet_dev_dataset = SonnetsDataset(args.held_out_sonnet_dev_path)
    held_out_sonnet_dataset = SonnetsDataset(args.held_out_sonnet_path)

    generated_sonnets_dev = []
    for batch in held_out_sonnet_dev_dataset:
        sonnet_id = batch[0]
        encoding = model.tokenizer(
            batch[1], return_tensors="pt", padding=False, truncation=True
        ).to(device)
        output = model.generate(
            encoding["input_ids"],
            beam_search=args.beam_search,
            temperature=args.temperature,
            top_p=args.top_p,
            max_length=128,
            beam_width=args.beam_width,
            length_penalty=args.length_penalty,
            repetition_penalty=args.repetition_penalty,
        )
        generated_sonnets_dev.append((sonnet_id, output))
        print(output)

    with open(args.sonnet_dev_out, "w+") as f:
        f.write(f"--Generated Sonnets-- \n\n")
        for sonnet in generated_sonnets_dev:
            f.write(f"\n{sonnet[0]}\n")
            f.write(sonnet[1])

    generated_sonnets = []
    for batch in held_out_sonnet_dataset:
        sonnet_id = batch[0]
        encoding = model.tokenizer(
            batch[1], return_tensors="pt", padding=False, truncation=True
        ).to(device)
        output = model.generate(
            encoding["input_ids"],
            beam_search=args.beam_search,
            temperature=args.temperature,
            top_p=args.top_p,
            max_length=128,
            beam_width=args.beam_width,
            length_penalty=args.length_penalty,
            repetition_penalty=args.repetition_penalty,
        )
        generated_sonnets.append((sonnet_id, output))
        print(output)

    with open(args.sonnet_out, "w+") as f:
        f.write(f"--Generated Sonnets-- \n\n")
        for sonnet in generated_sonnets:
            f.write(f"\n{sonnet[0]}\n")
            f.write(sonnet[1])


def get_args() -> argparse.Namespace:
    """Get the command line arguments."""
    parser = argparse.ArgumentParser()

    parser.add_argument("--sonnet_path", type=str, default="data/sonnets.txt")
    parser.add_argument(
        "--held_out_sonnet_path", type=str, default="data/sonnets_held_out.txt"
    )
    parser.add_argument(
        "--held_out_sonnet_dev_path", type=str, default="data/sonnets_held_out_dev.txt"
    )
    parser.add_argument(
        "--sonnet_dev_out", type=str, default="predictions/generated_sonnets_dev.txt"
    )
    parser.add_argument(
        "--sonnet_out", type=str, default="predictions/generated_sonnets.txt"
    )

    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--use_gpu", action="store_true")

    parser.add_argument(
        "--use_lora",
        action="store_true",
        help="Use LoRA for parameter-efficient fine-tuning.",
    )

    # Generation parameters.
    parser.add_argument(
        "--temperature", type=float, help="softmax temperature.", default=1.2
    )
    parser.add_argument(
        "--top_p",
        type=float,
        help="Cumulative probability distribution for nucleus sampling.",
        default=0.9,
    )
    parser.add_argument(
        "--beam_search",
        action="store_true",
        help="Use beam search instead of sampling.",
    )
    parser.add_argument(
        "--beam_width", type=int, help="Beam width for beam search.", default=3
    )
    parser.add_argument(
        "--length_penalty",
        type=float,
        help="Length penalty for beam search.",
        default=1.1,
    )
    parser.add_argument(
        "--repetition_penalty",
        type=float,
        help="Repetition penalty for beam search.",
        default=1.1,
    )

    parser.add_argument(
        "--batch_size", help="The training batch size.", type=int, default=8
    )
    parser.add_argument("--lr", type=float, help="learning rate", default=1e-5)
    parser.add_argument(
        "--model_size",
        type=str,
        help="The model size as specified on hugging face.",
        choices=["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"],
        default="gpt2",
    )

    parser.add_argument(
        "--patience", type=int, default=3, help="Early stopping patience."
    )
    parser.add_argument(
        "--min_delta",
        type=float,
        default=0.005,
        help="Minimum change in the monitored quantity to qualify as an improvement.",
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

    return args


if __name__ == "__main__":
    args = get_args()
    args.filepath = f"{args.epochs}-{args.lr}-sonnet.pt"  # Save path.
    seed_everything(args.seed)  # Fix the seed for reproducibility.
    train(args)
    generate_submission_sonnets(args)
