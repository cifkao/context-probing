from typing import Callable, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from transformers import BatchEncoding, PreTrainedTokenizer
from transformers.modeling_outputs import CausalLMOutput


def get_windows(
    examples: BatchEncoding,
    window_len: int,
    start: int = 0,
    stride: int = 1,
    pad_id: int = 0,
) -> BatchEncoding:
    """Get windows of length `window_len` from `examples`.

    Windows are padded with `pad_id` to the right up to `window_len`. The last window starts with
    the last token.
    """
    return BatchEncoding(
        {
            k: [
                t[i][j : j + window_len]
                + [pad_id if k in ["input_ids", "labels"] else 0]
                * (j + window_len - len(t[i]))
                for i in range(len(examples["input_ids"]))
                for j in range(start, len(examples["input_ids"][i]), stride)
            ]
            for k, t in examples.items()
        }
    )


def get_logprobs(
    model: Callable[..., CausalLMOutput],
    inputs: torch.Tensor,
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


def nll_score(logprobs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    if logprobs.shape[-1] == 1:
        return -logprobs.squeeze(-1)
    else:
        return -logprobs[:, torch.arange(len(labels)), labels]


def kl_div_score(logprobs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    del labels

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
    inputs: Dict[str, torch.Tensor],
    window_len: int,
    metrics: Optional[List[str]] = None,
    eos_id: int = 0,
) -> Dict[str, torch.Tensor]:
    if not metrics:
        metrics = ["kl_div", "xent"]

    [input_ids] = inputs["input_ids"]
    if "labels" in inputs:
        [label_ids] = inputs["labels"]
    else:
        label_ids = list(input_ids)[1:] + [eos_id]

    window_len = min(window_len, len(input_ids))
    inputs_sliding = get_windows(
        inputs, window_len=window_len, pad_id=eos_id
    ).convert_to_tensors("pt")
    logprobs = get_logprobs(
        model=model,
        inputs=inputs_sliding,
        label_probs=all(m == "xent" for m in metrics),
    )
    num_tgt_tokens = logprobs.shape[0]

    logprobs = logprobs.permute(1, 0, 2)
    logprobs = F.pad(logprobs, (0, 0, 0, window_len, 0, 0), value=torch.nan)
    logprobs = logprobs.view(-1, logprobs.shape[-1])[:-window_len]
    logprobs = logprobs.view(
        window_len, num_tgt_tokens + window_len - 1, logprobs.shape[-1]
    )

    scores = {}
    for key in metrics:
        scores[key] = METRIC_FUNCTIONS[key](logprobs=logprobs, labels=label_ids)

    return scores


@torch.inference_mode()
def get_importance_scores(scores: torch.Tensor) -> torch.Tensor:
    scores = (-scores).diff(dim=0).transpose(0, 1)
    scores = scores.nan_to_num()
    scores /= scores.abs().max(dim=1, keepdim=True).values + 1e-6
    return scores
