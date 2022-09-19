# Context length probing

## Usage

Install the package with with `pip install -e .`, or better (to install exact dependency versions from the lockfile), `pip install poetry && poetry install`.

The following scripts are included:
- `conllu_to_hf` – Converts a Universal Dependencies dataset in the [CoNLL-U](https://universaldependencies.org/format.html) format to a tokenized [HuggingFace dataset](https://github.com/huggingface/datasets) with the annotations included. E.g.:
  ```bash
  python -m context_probing.scripts.conllu_to_hf \
      data/ud-treebanks-v2.10/UD_English-LinES/en_lines-ud-dev.conllu \
      data/en_lines-ud-dev \
      --tokenizer-path gpt2
  ```
- `predict_sliding` – Applies a GPT-style language model along a sliding window and saves the logits as NumPy file(s). E.g.:
  ```bash
  for shard in {0..499}; do
      python -m context_probing.scripts.predict_sliding \
          --model-path EleutherAI/gpt-j-6B --window-len 1024 \
          --total-shards 500 --shard-id $shard \
          --batch-size 8 --num-proc 8 \
          data/en_lines-ud-dev \
          preds/gpt-j-6B/en_lines-ud-dev
  done
  ```
  ⚠️ This will produce 2.1 TB of data (4.2 GB per shard).  
  ⚠️ You may want to parallelize this by submitting each shard to a different compute node. Adjust `--num-proc` and `--batch-size` to the number of available CPU cores and memory.
- `preds_to_metrics` – Reads the predictions produced by `predict_sliding` and computes different metrics (cross entropy, KL divergence, top k predictions). E.g.:
  ```bash
  python -m context_probing.scripts.preds_to_metrics \
      --tokenizer-path EleutherAI/gpt-j-6B --topk 10 \
      data/en_lines-ud-dev \
      'preds/gpt-j-6B/en_lines-ud-dev-0*-of-00500.npy' \
      gpt-j-6B.en_lines-ud-dev.metrics.pth
  ```
