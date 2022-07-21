import argparse

import datasets
import numpy as np
import transformers


def get_windows_batched(example, window_len):
    return {
        k: [
            t[i][j : j + window_len]
            for i in range(len(example["input_ids"]))
            for j in range(len(example["input_ids"][0]) - window_len + 1)
        ]
        for k, t in example.items()
    }


def add_labels(example):
    example["labels"] = example["input_ids"].copy()
    return example


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", type=str)
    parser.add_argument("output_path_prefix", type=str)
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--window-len", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-proc", type=int, default=16)
    parser.add_argument("--total-shards", type=int, default=1)
    parser.add_argument("--shard-id", type=int, default=0)
    args = parser.parse_args()

    # Prepare the data
    dataset = datasets.Dataset.load_from_disk(args.data_path)
    dataset = dataset.remove_columns(
        [col for col in dataset.column_names if col not in ["input_ids", "attention_mask"]]
    )
    dataset = dataset.map(add_labels, num_proc=args.num_proc)
    dataset = dataset.map(
        get_windows_batched, fn_kwargs=dict(window_len=args.window_len),
        batched=True, batch_size=1, num_proc=args.num_proc
    )

    # Figure out which examples belong in this shard
    if args.shard_id not in range(args.total_shards):
        raise ValueError(f"Invalid shard index {args.shard_id}")
    # There are (len(dataset) % total_shards) shards that have 1 extra example
    shard_sizes = [
        len(dataset) // args.total_shards + int(i < len(dataset) % args.total_shards)
        for i in range(args.shard_id + 1)
    ]
    data_range = range(sum(shard_sizes[:-1]), sum(shard_sizes))
    dataset = dataset.select(data_range)

    # Load and run the model
    model = transformers.AutoModelForCausalLM.from_pretrained(args.model_path)
    trainer = transformers.Trainer(
        args=transformers.TrainingArguments(
            output_dir="/tmp/trainer_output",
            eval_accumulation_steps=1,
            per_device_eval_batch_size=args.batch_size,
            no_cuda=True),
        model=model,
    )
    results = trainer.predict(
        dataset, ignore_keys=["past_key_values", "hidden_states", "attentions"]
    )

    # Save the predictions
    out_path = f"{args.output_path_prefix}-{args.shard_id:05d}-of-{args.total_shards:05d}"
    np.save(out_path, results.predictions, allow_pickle=False)


if __name__ == "__main__":
    main()