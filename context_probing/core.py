from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from transformers import PreTrainedTokenizerBase
from transformers.modeling_outputs import CausalLMOutput

from .utils import (
    columns_to_diagonals,
    diagonals_to_columns,
    get_windows,
    rows_to_diagonals,
)


def get_logprobs(
    model: Callable[..., CausalLMOutput],
    inputs: Dict[str, Any],
    label_probs: bool = False,
    batch_size: int = 8,
) -> torch.Tensor:
    logprobs = []
    num_items = len(inputs["input_ids"])
    for i in range(0, num_items, batch_size):
        batch = {k: v[i : i + batch_size] for k, v in inputs.items()}
        batch_logprobs = model(**batch).logits.log_softmax(dim=-1).to(torch.float16)
        if label_probs:
            batch_logprobs = torch.gather(
                batch_logprobs, dim=-1, index=batch["labels"][..., None]
            )
        logprobs.append(batch_logprobs)
    return torch.cat(logprobs, dim=0)


def nll_score(
    logprobs: torch.Tensor, labels: torch.Tensor, allow_overwrite: bool = False
) -> torch.Tensor:
    if logprobs.shape[-1] == 1:
        return -logprobs.squeeze(-1)
    else:
        return -logprobs[:, torch.arange(len(labels)), labels]


def kl_div_score(
    logprobs: torch.Tensor, labels: torch.Tensor, allow_overwrite: bool = False
) -> torch.Tensor:
    del labels

    if not allow_overwrite:
        logprobs = logprobs.clone()

    log_p = logprobs[
        torch.arange(logprobs.shape[1]).clamp(max=logprobs.shape[0] - 1),
        torch.arange(logprobs.shape[1]),
    ]
    # Compute things in place as much as possible
    log_p_minus_log_q = logprobs
    del logprobs
    log_p_minus_log_q *= -1
    log_p_minus_log_q += log_p

    if log_p.dtype == torch.float16 and log_p.device.type == "cpu":
        # Use np.exp because torch.exp is not implemented for float16 on CPU
        p_np = log_p.numpy()
        del log_p
        np.exp(p_np, out=p_np)
        p = torch.as_tensor(p_np)
    else:
        p = log_p.exp_()
        del log_p

    result = log_p_minus_log_q
    result *= p

    return result.sum(dim=-1)


METRIC_FUNCTIONS = {
    "xent": nll_score,
    "nll": nll_score,
    "kl_div": kl_div_score,
}


@torch.inference_mode()
def run_probing(
    model: Callable[..., CausalLMOutput],
    inputs: Dict[str, Any],
    window_len: Optional[int] = None,
    metrics: Optional[List[str]] = None,
    tokenizer: Optional[PreTrainedTokenizerBase] = None,
    eos_id: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    if metrics is None:
        metrics = ["kl_div", "xent"]

    if eos_id is None:
        if tokenizer is not None:
            eos_id = tokenizer.eos_token_id
        else:
            eos_id = 0

    if len(inputs["input_ids"]) < 1 or not isinstance(
        inputs["input_ids"][0], (list, torch.Tensor)
    ):
        inputs = {k: [v] for k, v in inputs.items()}

    [input_ids] = inputs["input_ids"]
    if "labels" in inputs:
        [label_ids] = inputs["labels"]
    else:
        label_ids = list(input_ids)[1:] + [eos_id]

    if window_len is None:
        if tokenizer is None:
            window_len = np.inf
        else:
            window_len = tokenizer.model_max_length
    window_len = min(window_len, len(input_ids))
    inputs_sliding = get_windows(
        inputs, window_len=window_len, pad_id=eos_id, return_tensors="pt"
    )
    logprobs = get_logprobs(
        model=model,
        inputs=inputs_sliding,
        label_probs=all(m in ["xent", "nll"] for m in metrics),
    )
    num_tokens = logprobs.shape[0]

    logprobs = logprobs.transpose(0, 1)
    logprobs = columns_to_diagonals(logprobs)
    logprobs = logprobs[:, :num_tokens]

    scores = {}
    for key in metrics:
        scores[key] = METRIC_FUNCTIONS[key](logprobs=logprobs, labels=label_ids)

    return scores


@torch.inference_mode()
def get_delta_scores(
    scores: torch.Tensor, normalize: bool = False, nan_to_zero: bool = True
) -> torch.Tensor:
    num_tokens = scores.size(1)

    scores = F.pad(scores, (0, scores.size(0) - 1), value=torch.nan)
    scores = diagonals_to_columns(scores)
    scores = rows_to_diagonals(scores)
    scores = scores[:num_tokens]

    scores = (-scores).flip(1).diff(1).flip(1)
    scores = F.pad(scores, (0, 1), value=torch.nan)

    if nan_to_zero:
        scores = scores.nan_to_num()
    if normalize:
        eps = torch.finfo(scores.dtype).eps
        scores /= scores.nan_to_num().abs().max(dim=1, keepdim=True).values + eps
    return scores
