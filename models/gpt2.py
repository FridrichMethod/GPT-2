from typing import Dict

import torch
from torch import nn
from transformers import GPT2Model as OpenAIGPT2Model

from config import GPT2Config
from models.base_gpt import GPTPreTrainedModel
from modules.gpt2_layer import GPT2Layer
from utils import get_extended_attention_mask


class GPT2Model(GPTPreTrainedModel):
    """The GPT model returns the final embeddings for each token in a sentence.

    The model consists of:
    1. Embedding layers (used in self.embed).
    2. A stack of n GPT layers (used in self.encode).
    3. A linear transformation layer for the [CLS] token (used in self.forward, as given).
    """

    def __init__(self, config: GPT2Config) -> None:
        super().__init__(config)
        self.config = config

        # Embedding layers.
        self.word_embedding = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id
        )
        self.pos_embedding = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )
        self.embed_dropout = nn.Dropout(config.hidden_dropout_prob)

        # Register position_ids (1, len position emb) to buffer because it is a constant.
        position_ids = torch.arange(config.max_position_embeddings).unsqueeze(0)
        self.register_buffer("position_ids", position_ids)

        # GPT-2 layers.
        self.gpt_layers = nn.ModuleList(
            [GPT2Layer(config) for _ in range(config.num_hidden_layers)]
        )

        # Final layer norm.
        self.final_layer_norm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )

        self.init_weights()

    def embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Get the embedding for each input token."""
        input_shape = input_ids.size()
        seq_length = input_shape[1]

        inputs_embeds = self.word_embedding(input_ids)

        pos_ids = self.position_ids[:, :seq_length]
        pos_embeds = self.pos_embedding(pos_ids)

        # Add the two embeddings together.
        hidden_states = inputs_embeds + pos_embeds

        # Apply dropout.
        hidden_states = self.embed_dropout(hidden_states)

        return hidden_states

    def encode(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Feed the hidden states through the GPT layers.

        Args:
        ----
        hidden_states: torch.Tensor [batch_size, seq_len, hidden_size]
            The output from the embedding layer.
        attention_mask: torch.Tensor [batch_size, seq_len]
            The attention mask for the input tokens.

        Returns:
        --------
        hidden_states: torch.Tensor [batch_size, seq_len, hidden_size]
            The output from the GPT layers.
        """

        # Get the extended attention mask for self-attention.
        # Returns extended_attention_mask of size [batch_size, 1, 1, seq_len].
        # Distinguishes between non-padding tokens (with a value of 0) and padding tokens
        # (with a value of a large negative number).
        extended_attention_mask = get_extended_attention_mask(
            attention_mask, self.dtype
        )

        # Pass the hidden states through the encoder layers.
        for layer_module in self.gpt_layers:
            # Feed the encoding from the last bert_layer to the next.
            hidden_states = layer_module(hidden_states, extended_attention_mask)

        return hidden_states

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Get the final embeddings for each token in the sentence.

        Args:
        ----
        input_ids: torch.Tensor [batch_size, seq_len]
            The input token ids.
        attention_mask: torch.Tensor [batch_size, seq_len]
            The attention mask for the input tokens.

        Returns:
        --------
        output_dict: Dict[str, torch.Tensor]
            The final embeddings for each token in the sentence.
        """

        # Get the embedding for each input token.
        embedding_output = self.embed(input_ids)

        # Feed to a transformer (a stack of GPTLayers).
        sequence_output = self.encode(embedding_output, attention_mask)
        sequence_output = self.final_layer_norm(sequence_output)

        # Get the hidden state of the final token.
        last_non_pad_idx = attention_mask.sum(dim=1) - 1  # Subtract 1 to get last index
        last_token = sequence_output[
            torch.arange(sequence_output.shape[0]), last_non_pad_idx
        ]

        return {"last_hidden_state": sequence_output, "last_token": last_token}

    def hidden_state_to_token(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """Convert hidden states to token logits using weight tying.

        GPT-2 uses weight tying with the input word embeddings. The logits are the dot product between output hidden states
        and the word embedding weights:

        >>> return hidden_state(s) * E^T
        """

        return torch.matmul(hidden_state, self.word_embedding.weight.T)

    @classmethod
    def from_pretrained(
        cls,
        model: str = "gpt2",
        d: int = 768,
        l: int = 12,
        num_heads: int = 12,
        use_lora: bool = False,
        lora_r: int = 8,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
    ) -> "GPT2Model":
        """Load the GPT-2 model from the Hugging Face library and remap the weights to our model."""
        # Load the original Hugging Face GPT-2 model.
        gpt_model = OpenAIGPT2Model.from_pretrained(model).eval()

        # Create our configuration. We add extra attributes for LoRA.
        config = GPT2Config(
            hidden_size=d,
            num_hidden_layers=l,
            num_attention_heads=num_heads,
            intermediate_size=d * 3,
        )
        config.use_lora = use_lora
        if use_lora:
            config.lora_r = lora_r
            config.lora_alpha = lora_alpha
            config.lora_dropout = lora_dropout

        our_model = GPT2Model(config).eval()

        # Load word and positional embeddings.
        our_model.word_embedding.load_state_dict(gpt_model.wte.state_dict())
        our_model.pos_embedding.load_state_dict(gpt_model.wpe.state_dict())

        for i in range(l):
            layer = our_model.gpt_layers[i]
            # Remap the Q, K, V weights.
            if config.use_lora:
                # For LoRA, the base linear layer is stored in the attribute 'linear'
                layer.self_attention.query.linear.weight.data = gpt_model.state_dict()[
                    f"h.{i}.attn.c_attn.weight"
                ][:, :d].T
                layer.self_attention.query.linear.bias.data = gpt_model.state_dict()[
                    f"h.{i}.attn.c_attn.bias"
                ][:d]
                layer.self_attention.key.linear.weight.data = gpt_model.state_dict()[
                    f"h.{i}.attn.c_attn.weight"
                ][:, d : d * 2].T
                layer.self_attention.key.linear.bias.data = gpt_model.state_dict()[
                    f"h.{i}.attn.c_attn.bias"
                ][d : d * 2]
                layer.self_attention.value.linear.weight.data = gpt_model.state_dict()[
                    f"h.{i}.attn.c_attn.weight"
                ][:, d * 2 :].T
                layer.self_attention.value.linear.bias.data = gpt_model.state_dict()[
                    f"h.{i}.attn.c_attn.bias"
                ][d * 2 :]
            else:
                layer.self_attention.query.weight.data = gpt_model.state_dict()[
                    f"h.{i}.attn.c_attn.weight"
                ][:, :d].T
                layer.self_attention.query.bias.data = gpt_model.state_dict()[
                    f"h.{i}.attn.c_attn.bias"
                ][:d]
                layer.self_attention.key.weight.data = gpt_model.state_dict()[
                    f"h.{i}.attn.c_attn.weight"
                ][:, d : d * 2].T
                layer.self_attention.key.bias.data = gpt_model.state_dict()[
                    f"h.{i}.attn.c_attn.bias"
                ][d : d * 2]
                layer.self_attention.value.weight.data = gpt_model.state_dict()[
                    f"h.{i}.attn.c_attn.weight"
                ][:, d * 2 :].T
                layer.self_attention.value.bias.data = gpt_model.state_dict()[
                    f"h.{i}.attn.c_attn.bias"
                ][d * 2 :]

            # Remap final dense layer in MHA.
            layer.attention_dense.weight.data = gpt_model.state_dict()[
                f"h.{i}.attn.c_proj.weight"
            ].T
            layer.attention_dense.bias.data = gpt_model.state_dict()[
                f"h.{i}.attn.c_proj.bias"
            ]

            # Remap attention layer norm.
            layer.attention_layer_norm.weight.data = gpt_model.state_dict()[
                f"h.{i}.ln_1.weight"
            ]
            layer.attention_layer_norm.bias.data = gpt_model.state_dict()[
                f"h.{i}.ln_1.bias"
            ]

            # Remap post-attention MLP layers.
            layer.interm_dense.weight.data = gpt_model.state_dict()[
                f"h.{i}.mlp.c_fc.weight"
            ].T
            layer.interm_dense.bias.data = gpt_model.state_dict()[
                f"h.{i}.mlp.c_fc.bias"
            ]
            layer.out_dense.weight.data = gpt_model.state_dict()[
                f"h.{i}.mlp.c_proj.weight"
            ].T
            layer.out_dense.bias.data = gpt_model.state_dict()[f"h.{i}.mlp.c_proj.bias"]

            # Remap second layer norm weights.
            layer.out_layer_norm.weight.data = gpt_model.state_dict()[
                f"h.{i}.ln_2.weight"
            ]
            layer.out_layer_norm.bias.data = gpt_model.state_dict()[f"h.{i}.ln_2.bias"]

        # Remap the final layer norm values.
        our_model.final_layer_norm.weight.data = gpt_model.state_dict()["ln_f.weight"]
        our_model.final_layer_norm.bias.data = gpt_model.state_dict()["ln_f.bias"]

        return our_model
