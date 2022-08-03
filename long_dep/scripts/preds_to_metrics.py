import argparse
import collections
import glob

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import transformers

from .predict_sliding import get_data


def _get_saved_shape(path):
    arr = np.load(path, mmap_mode="r")
    return arr.shape


def cross_entropy(logits, labels):
    xents = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), reduction="none"
    )
    return xents.reshape(logits.shape[:2])


def kl_div(log_q, log_p):
    return F.kl_div(log_q, log_p, log_target=True, reduction="none").sum(dim=-1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", type=str)
    parser.add_argument("input_path_pattern", type=str)
    parser.add_argument("output_path", type=str)
    parser.add_argument("--tokenizer-path", type=str, required=True)
    args = parser.parse_args()

    shard_paths = sorted(glob.glob(args.input_path_pattern))
    _, window_len, _ = _get_saved_shape(shard_paths[0])
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.tokenizer_path)
    pad_id = (
        tokenizer.pad_token_id if tokenizer.pad_token_id is not None
        else tokenizer.eos_token_id
    )
    dataset = get_data(
        args.data_path,
        window_len=window_len,
        pad_id=pad_id,
        columns=["input_ids", "attention_mask"]
    )

    labels_all = torch.as_tensor([x[1] for x in dataset["input_ids"]])
    mask_all = torch.as_tensor([x[1] for x in dataset["attention_mask"]])

    logprobs_ctx1, logprobs_full = [], []
    for path in tqdm(shard_paths):
        logits = np.load(path, mmap_mode="r")
        logprobs_ctx1.append(
            torch.log_softmax(
                torch.tensor(logits[:, 0], dtype=torch.float32), dim=-1)
        )
        logprobs_full.append(
            torch.log_softmax(
                torch.tensor(logits[:, -1], dtype=torch.float32), dim=-1)
        )
        del logits
    logprobs_ctx1 = torch.cat(logprobs_ctx1)
    logprobs_full = torch.cat(logprobs_full)

    start_idx = 0
    metrics = {
        k: torch.full((window_len, len(mask_all) + window_len), torch.nan).T
        for k in ["xent", "kl_full", "kl_ctx1"]
    }
    for path in tqdm(shard_paths):
        logits = torch.tensor(np.load(path, mmap_mode="r"), dtype=torch.float32)
        end_idx = start_idx + len(logits)
        logprobs = torch.log_softmax(logits, dim=-1)

        # Compute the global position of the token each prediction corresponds to
        indices = torch.arange(start_idx, end_idx)[:, None] + torch.arange(window_len)

        # Compute metrics
        metrics["xent"][start_idx:end_idx] = cross_entropy(logits, labels_all[indices])
        metrics["kl_full"][start_idx:end_idx] = kl_div(
            logprobs, logprobs_full[torch.clamp(indices - window_len + 1, min=0)]
        )
        metrics["kl_ctx1"][start_idx:end_idx] = kl_div(
            logprobs_ctx1[indices], logprobs
        )

        # Mask out values that correspond to padding
        for val in metrics.values():
            val[start_idx:end_idx][~mask_all[indices]] = torch.nan

        start_idx = end_idx
    
    # "Skew" the results so that they are aligned on token position
    for key in list(metrics.keys()):
        val = metrics[key].T
        val = val.view(-1)[:-window_len]
        val = val.view(len(mask_all) + window_len - 1, window_len)
        assert torch.isnan(val[len(mask_all):]).all()
        val = val[:len(mask_all)]
        metrics[key] = val.T

    torch.save(metrics, args.output_path)


if __name__ == "__main__":
    main()
