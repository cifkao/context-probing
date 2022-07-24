import argparse
import glob

import numpy as np
import torch
from tqdm import tqdm

from .predict_sliding import get_data


def _get_saved_shape(path):
    arr = np.load(path, mmap_mode="r")
    return arr.shape


def compute_xents(logits, labels):
    logits = torch.as_tensor(logits, dtype=torch.float32)
    labels = torch.as_tensor(labels)
    loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
    xents = loss_fn(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))
    xents = xents.reshape(logits.shape[:2])
    return xents.numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", type=str)
    parser.add_argument("input_path_pattern", type=str)
    parser.add_argument("output_path", type=str)
    args = parser.parse_args()

    shard_paths = sorted(glob.glob(args.input_path_pattern))
    dataset = get_data(
        args.data_path,
        window_len=_get_saved_shape(shard_paths[0])[1],
        columns=["input_ids"]
    )

    inputs = np.asarray(dataset["input_ids"])

    start_idx = 0
    xents = []
    for path in tqdm(shard_paths):
        logits = np.load(path)[:, :-1]
        end_idx = start_idx + len(logits)
        labels = inputs[start_idx:end_idx, 1:]

        xents.append(compute_xents(logits, labels))

        start_idx = end_idx
    xents = np.concatenate(xents, axis=0)

    np.save(args.output_path, xents)


if __name__ == "__main__":
    main()
