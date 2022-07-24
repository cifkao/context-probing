import argparse
import glob
import sys

import dask
import dask.dataframe as dd
import dask.diagnostics
import dask.array as da
import numpy as np
import torch
import torch.functional as F

from .predict_sliding import get_data, get_shard_sizes


def _get_saved_shape_and_dtype(path):
    arr = np.load(path, mmap_mode="r")
    return arr.shape, arr.dtype




def compute_xents(logits, labels):
    logits = torch.as_tensor(logits, dtype=torch.float32)
    labels = torch.as_tensor(labels)
    loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
    xents = loss_fn(logits.view(-1, logits.shape[-1]), labels.view(-1)).reshape(logits.shape[:2])
    return xents.numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", type=str)
    parser.add_argument("input_path_pattern", type=str)
    parser.add_argument("output_path", type=str)
    parser.add_argument("--num-shards", type=int, default=None)
    args = parser.parse_args()

    shard_paths = sorted(glob.glob(args.input_path_pattern))
    (_, window_len, vocab_size), dtype_in = _get_saved_shape_and_dtype(shard_paths[0])
    dataset = get_data(args.data_path, window_len=window_len, columns=["input_ids"])
    shard_sizes = get_shard_sizes(dataset, args.num_shards or len(shard_paths))

    if len(shard_sizes) > len(shard_paths):
        print(
            f"Found {len(shard_paths)} shards instead of {len(shard_sizes)}. "
            "Truncating dataset.",
            file=sys.stderr
        )
        shard_sizes = shard_sizes[:len(shard_paths)]
        dataset = dataset.select(range(sum(shard_sizes)))

    logits = da.concatenate([
        da.from_delayed(
            dask.delayed(np.load)(path),
            shape=(size, window_len, vocab_size), 
            dtype=dtype_in
        )
        for path, size in zip(shard_paths, shard_sizes)
    ])
    logits = logits[:, :-1]

    labels = da.from_array(
        np.asarray(dataset["input_ids"])[:len(logits), 1:],
        chunks=logits.chunks[:-1]
    )

    xents = da.map_blocks(compute_xents, logits, labels[..., None], drop_axis=[-1], dtype=np.float32)

    print("Logits:", logits, file=sys.stderr)
    print("Labels:", labels, file=sys.stderr)
    print("Xents: ", xents, file=sys.stderr)

    with dask.diagnostics.ProgressBar():
        xents_np = xents.compute()

    np.save(args.output_path, xents_np)


if __name__ == "__main__":
    main()
