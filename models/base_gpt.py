from torch import dtype, nn

from config import GPT2Config
from utils import get_parameter_dtype


class GPTPreTrainedModel(nn.Module):

    def __init__(self, config: GPT2Config) -> None:
        super().__init__()
        self.config = config
        self.name_or_path = config.name_or_path

    def init_weights(self) -> None:
        """Initialize the weights"""
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    @property
    def dtype(self) -> dtype:
        """Return the dtype of the model."""
        return get_parameter_dtype(self)
