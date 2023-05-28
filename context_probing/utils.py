import datasets
import numpy as np

import torch
import torch.nn.functional as F


BAD_CHAR = chr(0xFFFD)


def _get_windows_batched(examples, window_len, pad_id):
    return {
        k: [
            t[i][j : j + window_len]
            + [pad_id if k == "input_ids" else 0 if type(t[i][0]) == int else None]
            * (j + window_len - len(t[i]))
            for i in range(len(examples["input_ids"]))
            for j in range(len(examples["input_ids"][i]) - 1)
        ]
        for k, t in examples.items()
    }


def _add_labels(example):
    example["labels"] = example["input_ids"].copy()
    return example


def _add_positions(example):
    example["positions"] = np.arange(len(example["input_ids"]))
    return example


def _add_seq_ids(example, idx):
    example["seq_id"] = [idx] * len(example["input_ids"])
    return example


def get_data(
    path,
    pad_id=-1,
    window_len=None,
    columns=None,
    add_positions=False,
    add_seq_ids=False,
    num_proc=16,
):
    dataset = datasets.Dataset.load_from_disk(path)
    if columns is not None:
        dataset = dataset.remove_columns(
            [col for col in dataset.column_names if col not in columns]
        )
    dataset = dataset.map(_add_labels, num_proc=num_proc)
    if add_positions:
        dataset = dataset.map(_add_positions, num_proc=num_proc)
    if add_seq_ids:
        dataset = dataset.map(_add_seq_ids, num_proc=num_proc, with_indices=True)
    if window_len is not None:
        dataset = dataset.map(
            _get_windows_batched,
            fn_kwargs=dict(window_len=window_len, pad_id=pad_id),
            batched=True,
            batch_size=1,
            num_proc=num_proc,
        )
    return dataset


def get_shard_sizes(dataset, total_shards):
    return [
        len(dataset) // total_shards + int(i < len(dataset) % total_shards)
        for i in range(total_shards)
    ]


def ids_to_readable_tokens(tokenizer, ids, strip_whitespace=True):
    cur_ids = []
    result = []
    for idx in ids:
        cur_ids.append(idx)
        decoded = tokenizer.decode(cur_ids)
        if BAD_CHAR not in decoded:
            if strip_whitespace:
                decoded = decoded.strip()
            result.append(decoded)
            del cur_ids[:]
        else:
            result.append("")
    return result


def columns_to_diagonals(tensor: torch.Tensor, fill_value=torch.nan) -> torch.Tensor:
    """Rearrange a tensor so that columns become diagonals; inserted positions are filled with
    `fill_value`.

    The rows and columns are the first two dimensions, respectively; the remaining dimensions are
    left untouched.
    """
    num_rows, num_cols = tensor.shape[:2]
    tensor = F.pad(
        tensor, (*((0,) * 2 * (tensor.ndim - 2)), 0, num_rows), value=fill_value
    )
    tensor = tensor.reshape(-1, *tensor.shape[2:])[:-num_rows]
    tensor = tensor.reshape(num_rows, num_cols + num_rows - 1, *tensor.shape[1:])
    return tensor


def rows_to_diagonals(tensor: torch.Tensor, fill_value=torch.nan) -> torch.Tensor:
    """Rearrange a tensor so that rows become diagonals; inserted positions are filled with
    `fill_value`.

    The rows and columns are the first two dimensions, respectively; the remaining dimensions are
    left untouched.
    """
    return columns_to_diagonals(tensor.transpose(0, 1), fill_value).transpose(0, 1)


def diagonals_to_columns(tensor: torch.Tensor) -> torch.Tensor:
    """Rearrange a tensor so that diagonals become columns. Values on diagonals that are shorter
    than the columns are dropped.

    The rows and columns are the first two dimensions, respectively; the remaining dimensions are
    left untouched.
    """
    num_rows, num_cols = tensor.shape[:2]
    num_cols -= num_rows - 1
    tensor = tensor.reshape(-1, *tensor.shape[2:])
    tensor = F.pad(tensor, (*((0,) * 2 * (tensor.ndim - 1)), 0, num_rows))
    tensor = tensor.reshape(num_rows, num_cols + num_rows, *tensor.shape[1:])
    tensor = tensor[:, :num_cols]
    return tensor
