from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from transformers import PreTrainedTokenizerBase
from transformers.modeling_outputs import ModelOutput

from .utils import (
    columns_to_diagonals,
    diagonals_to_columns,
    get_windows,
    rows_to_diagonals,
)


def get_logprobs(
    model: Callable[..., Union[ModelOutput, torch.Tensor]],
    inputs: Dict[str, Any],
    labels_only: bool = False,
    batch_size: int = 8,
) -> torch.Tensor:
    logprobs = []
    num_items = len(inputs["input_ids"])
    for i in range(0, num_items, batch_size):
        batch = {k: v[i : i + batch_size] for k, v in inputs.items()}
        batch_output = model(**batch)
        if isinstance(batch_output, torch.Tensor):
            batch_logits = batch_output
        else:
            batch_logits = batch_output.logits
        batch_logprobs = batch_logits.log_softmax(dim=-1).to(torch.float16)
        if labels_only:
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
        torch.arange(1, logprobs.shape[1] + 1).clamp(max=logprobs.shape[0] - 1),
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
    model: Callable[..., Union[ModelOutput, torch.Tensor]],
    inputs: Dict[str, Any],
    window_len: Optional[int] = None,
    metrics: Optional[List[str]] = None,
    tokenizer: Optional[PreTrainedTokenizerBase] = None,
    eos_id: Optional[int] = None,
    unigram_logprobs: Optional[torch.Tensor] = None,
    batch_size: int = 8,
) -> Dict[str, torch.Tensor]:
    """Run context length probing with the given model and inputs.

    Args:
        model: A pretrained PyTorch Hugging Face Transformers causal language model, or a callable
            accepting as parameters PyTorch tensors corresponding to a batch of values from
            `inputs` and returning a either a `ModelOutput` with a `logits` attribute or a logits
            PyTorch tensor.
        inputs: A dictionary of inputs to the model, containing at least the key "input_ids". If no
            "labels" key is present, the labels are assumed to be the same as the inputs shifted by
            one and padded with the EOS token. The labels are only useful for computing the "xent"
            ("nll") metric.
        window_len: The maximum length of the context window to consider. If not specified, it will
            be set so that the entire input sequence is considered, without exceeding the maximum
            input length supported by the model.
        metrics: A list of metrics to compute. The available metrics are "kl_div" and "xent" (or
            "nll", which is an alias for "xent"). If not specified, both metrics will be computed.
            Computing only "xent" or "nll" is more memory-efficient.
        tokenizer: A tokenizer for the model. Used only to determine the maximum supported input
            length and the EOS token ID.
        eos_id: The ID of the EOS token. If not specified, it will be determined from the tokenizer
            or set to 0.
        unigram_logprobs: A tensor of unigram (null-context) log-probabilities for all tokens in
            the vocabulary. If not given, they will be set to NaN.
        batch_size: The batch size to use for model inference.

    Returns:
        A dictionary mapping the name of each metric to its values as a PyTorch tensor of shape
        `(window_len + 1, len(inputs["input_ids"]))`. The first dimension corresponds to context
        length (starting from 0) and the second to target token position.
    """
    if metrics is None:
        metrics = ["kl_div", "xent"]
    # If only cross entropy is requested, only store the log-probabilities of the labels
    labels_only = all(m in ["xent", "nll"] for m in metrics)

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
        inputs = {k: v for k, v in inputs.items() if k != "labels"}
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
        labels_only=labels_only,
        batch_size=batch_size
    )
    num_tokens = logprobs.shape[0]

    logprobs = logprobs.transpose(0, 1)
    logprobs = columns_to_diagonals(logprobs)
    logprobs = logprobs[:, :num_tokens]

    if unigram_logprobs is not None:
        unigram_logprobs = unigram_logprobs.clone()
        unigram_logprobs[~torch.isfinite(unigram_logprobs)] = torch.nan
        if labels_only:
            unigram_logprobs = unigram_logprobs[label_ids].unsqueeze(-1)
        else:
            unigram_logprobs = unigram_logprobs.unsqueeze(0).repeat(num_tokens, 1)
    else:
        unigram_logprobs = torch.full_like(logprobs[0], torch.nan)
    logprobs = torch.cat([unigram_logprobs.unsqueeze(0), logprobs], dim=0)

    scores = {}
    for key in metrics:
        scores[key] = METRIC_FUNCTIONS[key](logprobs=logprobs, labels=label_ids)

    return scores


@torch.inference_mode()
def get_delta_scores(
    metric: torch.Tensor, normalize: bool = False, nan_to_zero: bool = True
) -> torch.Tensor:
    """Compute differential importance scores from the metrics returned by `run_probing()`.

    Args:
        metric: One of the metric tensors returned by `run_probing()`.
        normalize: Whether to normalize the scores to the range [-1, 1].
        nan_to_zero: Whether to replace non-finite values with zeros.

    Returns:
        A `scores` tensor of shape `(metric.size(1), metric.size(1))` where `scores[i, j]` is
        the differential importance score of the `j`-th context token for the `i + 1`-st target
        token (starting from 0).
    """
    num_tokens = metric.size(1)

    metric = F.pad(metric, (1, metric.size(0) - 1), value=torch.nan)
    metric = diagonals_to_columns(metric)
    metric = rows_to_diagonals(metric)
    metric = metric[1 : num_tokens + 1]

    scores = (-metric).flip(1).diff(1).flip(1)

    scores_finite = scores.nan_to_num(nan=0., posinf=0., neginf=0.)
    if nan_to_zero:
        scores = scores_finite
    if normalize:
        eps = torch.finfo(scores.dtype).eps
        scores /= scores_finite.abs().max(dim=1, keepdim=True).values + eps
    return scores
