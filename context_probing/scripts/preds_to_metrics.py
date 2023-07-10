import argparse
import glob
import itertools
import sys

import datasets
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import transformers

from ..utils import get_hf_data


def _get_saved_shape(path):
    arr = np.load(path, mmap_mode="r")
    return arr.shape


def _get_seq_boundary_indices(seq_lengths):
    seq_bounds = np.cumsum([0] + list(seq_lengths))
    seq_start_indices = np.zeros(sum(seq_lengths) + 1, dtype=int)
    seq_start_indices[seq_bounds[1:]] = seq_lengths
    seq_start_indices = torch.as_tensor(
        np.cumsum(seq_start_indices)[:-1], device=seq_lengths.device
    )
    seq_end_indices = np.zeros(sum(seq_lengths), dtype=int)
    seq_end_indices[seq_bounds[:-1]] = seq_lengths
    seq_end_indices = torch.as_tensor(
        np.cumsum(seq_end_indices), device=seq_lengths.device
    )
    return seq_start_indices, seq_end_indices


def cross_entropy(logits, labels):
    xents = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), reduction="none"
    )
    return xents.reshape(logits.shape[:2])


def kl_div(log_q, log_p):
    return F.kl_div(log_q, log_p, log_target=True, reduction="none").sum(dim=-1)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", type=str)
    parser.add_argument("input_path_pattern", type=str)
    parser.add_argument("output_path", type=str)
    parser.add_argument("--tokenizer-path", type=str, required=True)
    parser.add_argument("--topk", type=int, default=None)
    parser.add_argument("--max-ctx", type=int, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    datasets.disable_caching()

    shard_paths = sorted(glob.glob(args.input_path_pattern))
    _, window_len, vocab_size = _get_saved_shape(shard_paths[0])
    if args.max_ctx is not None:
        window_len = args.max_ctx
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.tokenizer_path)
    pad_id = (
        tokenizer.pad_token_id if tokenizer.pad_token_id is not None
        else tokenizer.eos_token_id
    )
    dataset = get_hf_data(
        args.data_path,
        window_len=window_len,
        pad_id=pad_id,
        add_seq_ids=True,
        columns=["input_ids"]
    )

    labels_all = torch.as_tensor(
        [x[1] for x in dataset["input_ids"]],
        device=args.device
    )
    total_len = len(labels_all)

    # For each position, compute the start and end index of the sequence (document)
    # it belongs to
    seq_lengths = torch.as_tensor(
        [
            sum(1 for _ in s)
            for _, s in itertools.groupby(ids[0] for ids in dataset["seq_id"])
        ],
        device=args.device
    )
    assert seq_lengths.sum() == total_len
    seq_start_indices, seq_end_indices = _get_seq_boundary_indices(seq_lengths)

    def iter_shards(shard_paths):
        start_idx = 0
        for path in shard_paths:
            shard = np.load(path, mmap_mode="r")
            shard = shard[:, :window_len]
            end_idx = start_idx + len(shard)
            yield start_idx, end_idx, shard
            start_idx = end_idx

    # Store the distributions for min and max context length for *all* the shards
    logprobs_ctx1 = torch.full(
        (total_len, vocab_size), torch.nan, device=args.device
    )
    logprobs_full = torch.full(
        (total_len, vocab_size), torch.nan, device=args.device
    )
    for start_idx, end_idx, logits in iter_shards(tqdm(shard_paths)):
        logprobs_ctx1[start_idx:end_idx] = torch.log_softmax(
            torch.tensor(logits[:, 0], dtype=torch.float32, device=args.device),
            dim=-1
        )

        # Loop over all sequences that overlap with this shard (should be just 1 or 2)
        for seq_start_idx in seq_start_indices[start_idx:end_idx].unique():
            # If the first window of the sequence is in this shard, use all the
            # positions
            if seq_start_idx in range(start_idx, end_idx):
                logprobs_full[seq_start_idx:seq_start_idx + window_len] = (
                    torch.log_softmax(
                        torch.tensor(
                            logits[seq_start_idx - start_idx],
                            dtype=torch.float32,
                            device=args.device
                        ),
                        dim=-1
                    )
                )
        # For all other windows, use the last position
        idxs = torch.arange(start_idx, end_idx)
        offset = window_len - 1
        # Make sure we stay within the same sequence
        idxs = idxs[idxs + offset < len(seq_start_indices)]
        idxs = idxs[seq_start_indices[idxs + offset] == seq_start_indices[idxs]]
        logprobs_full[idxs + offset] = torch.log_softmax(
            torch.tensor(
                logits[idxs - start_idx, -1], dtype=torch.float32, device=args.device
            ),
            dim=-1
        )

        del logits

    metrics = {
        k: torch.full(
            (total_len + window_len, window_len),
            torch.nan, device=args.device
        )
        for k in ["xent", "kl_full", "kl_ctx1"]
    }
    if args.topk:
        metrics["topk"] = -torch.ones(
            (total_len + window_len, window_len, args.topk),
            dtype=torch.int32, device=args.device
        )
        assert vocab_size < torch.iinfo(metrics["topk"].dtype).max

    for start_idx, end_idx, logits in iter_shards(tqdm(shard_paths)):
        logits = torch.tensor(logits, dtype=torch.float32, device=args.device)
        logprobs = torch.log_softmax(logits, dim=-1)

        # Compute the global position of the token each prediction corresponds to
        indices = (
            torch.arange(start_idx, end_idx, device=args.device)[:, None]
            + torch.arange(window_len, device=args.device)
        )
        # Do not let the indices overflow into neighboring sequences
        indices_clamped = torch.clamp(
            indices, max=seq_end_indices[start_idx:end_idx, None] - 1
        )

        # Compute metrics
        metrics["xent"][start_idx:end_idx] = cross_entropy(
            logits, labels_all[indices_clamped]
        )
        metrics["kl_ctx1"][start_idx:end_idx] = kl_div(
            logprobs_ctx1[indices_clamped], logprobs
        )
        metrics["kl_full"][start_idx:end_idx] = kl_div(
            logprobs, logprobs_full[indices_clamped]
        )
        if args.topk:
            metrics["topk"][start_idx:end_idx] = logits.topk(args.topk).indices

    # "Skew" the results so that they are aligned on token position
    for key in list(metrics.keys()):
        shape_tail = metrics[key].shape[2:]
        val = metrics[key].transpose(0, 1).contiguous().cpu()
        val = val.view(-1, *shape_tail)[:-window_len]
        val = val.view(window_len, total_len + window_len - 1, *shape_tail)
        val = val[:, :-window_len + 1]
        metrics[key] = val.transpose(0, 1)

    torch.save(metrics, args.output_path)


if __name__ == "__main__":
    main()
