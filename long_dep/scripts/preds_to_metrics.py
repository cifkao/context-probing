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
    _, window_len, vocab_size = _get_saved_shape(shard_paths[0])
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

    labels_all = torch.as_tensor(
        [x[1] for x in dataset["input_ids"]] + [pad_id] * window_len
    )
    mask_all = torch.as_tensor(
        [x[1] for x in dataset["attention_mask"]] + [0] * window_len,
        dtype=bool
    )

    def iter_shards(shard_paths):
        start_idx = 0
        for path in shard_paths:
            shard = np.load(path, mmap_mode="r")
            end_idx = start_idx + len(shard)
            yield start_idx, end_idx, shard
            start_idx = end_idx

    logprobs_ctx1 = torch.full((len(mask_all), vocab_size), torch.nan)
    logprobs_full = torch.full((len(mask_all), vocab_size), torch.nan)
    for start_idx, end_idx, logits in iter_shards(tqdm(shard_paths)):
        logprobs_ctx1[start_idx:end_idx] = torch.log_softmax(
            torch.tensor(logits[:, 0], dtype=torch.float32), dim=-1
        )
        logprobs_full[start_idx:end_idx] = torch.log_softmax(
            torch.tensor(logits[:, -1], dtype=torch.float32), dim=-1
        )
        del logits

    metrics = {
        k: torch.full((len(mask_all), window_len), torch.nan)
        for k in ["xent", "kl_full", "kl_ctx1"]
    }
    for start_idx, end_idx, logits in iter_shards(tqdm(shard_paths)):
        logits = torch.tensor(logits, dtype=torch.float32)
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
    
    # "Skew" the results so that they are aligned on token position
    for key in list(metrics.keys()):
        val = metrics[key].T.contiguous()
        val = val.view(-1)[:-window_len]
        val = val.view(window_len, len(mask_all) - 1)
        if not torch.isnan(val[:, -window_len + 1:]).all():
            print(f"{key} trailing padding is not NaN:", val[:, -window_len + 1:], file=sys.stderr)
        val = val[:, :-window_len + 1]
        metrics[key] = val.T

    torch.save(metrics, args.output_path)


if __name__ == "__main__":
    main()
