import math
from typing import Callable, Iterable, Optional, Tuple, overload

import torch
from torch import nn
from torch.optim.optimizer import Optimizer


class AdamW(Optimizer):
    """Implements AdamW algorithm."""

    def __init__(
        self,
        params: Iterable[torch.nn.parameter.Parameter],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-6,
        weight_decay: float = 0.0,
        correct_bias: bool = True,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr} - should be >= 0.0")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(
                f"Invalid beta parameter: {betas[0]} - should be in [0.0, 1.0]"
            )
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(
                f"Invalid beta parameter: {betas[1]} - should be in [0.0, 1.0]"
            )
        if eps < 0.0:
            raise ValueError("Invalid epsilon value: {eps} - should be >= 0.0")

        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
            "correct_bias": correct_bias,
        }
        super().__init__(params, defaults)

    @overload
    def step(self, closure: None = ...) -> None: ...

    @overload
    def step(self, closure: Callable[[], float]) -> float: ...

    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        """Performs a single optimization step."""
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            correct_bias = group["correct_bias"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError(
                        "AdamW does not support sparse gradients, "
                        "please consider SparseAdam instead"
                    )

                # State initialization
                state = self.state[p]
                if "step" not in state:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(p.data)
                state["step"] += 1
                t = state["step"]
                exp_avg: torch.Tensor = state["exp_avg"]
                exp_avg_sq: torch.Tensor = state["exp_avg_sq"]

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                if correct_bias:
                    # Bias correction terms
                    bias_correction1 = 1 - beta1**t
                    bias_correction2 = 1 - beta2**t
                    # Efficient Adam update uses a single scaled step_size
                    step_size = lr * math.sqrt(bias_correction2) / bias_correction1
                else:
                    step_size = lr

                denom = exp_avg_sq.sqrt().add_(eps)
                p.data.addcdiv_(exp_avg, denom, value=-step_size)

                if weight_decay != 0:
                    p.data.add_(p.data, alpha=-lr * weight_decay)

        if closure is not None:
            return closure()

        return None
