import datasets
import numpy as np


BAD_CHAR = chr(0xfffd)


def _get_windows_batched(examples, window_len, pad_id):
    return {
        k: [
            t[i][j : j + window_len] + [
                pad_id if k == "input_ids" else
                0 if type(t[i][0]) == int else
                None
            ] * (j + window_len - len(t[i]))
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
    path, pad_id=-1, window_len=None, columns=None, add_positions=False, add_seq_ids=False, num_proc=16
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
            _get_windows_batched, fn_kwargs=dict(window_len=window_len, pad_id=pad_id),
            batched=True, batch_size=1, num_proc=num_proc
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