"""Convert a corpus in a CoNLL-U format to a HuggingFace dataset. Each document is encoded using
the given tokenizer and stored as a single example. The new tokenization is aligned to the original
one and the CoNLL-U fields are copied over.
"""

import argparse
import collections

import conllu
import datasets
import transformers

from ..utils import BAD_CHAR


def iter_sentences(paths):
    sent_idx = 0
    for path in paths:
        with open(path) as f:
            doc_id, sent_id = None, None
            for sentence in conllu.parse_incr(f):
                if "newdoc id" in sentence.metadata:
                    doc_id = sentence.metadata["newdoc id"]
                tokens_out = []
                for token in sentence:
                    if sentence.metadata["sent_id"] != sent_id:
                        sent_id = sentence.metadata["sent_id"]
                        sent_idx += 1
                    token["sent_id"] = sent_idx
                    token["doc_id"] = doc_id
                    token["space_after"] = (token["misc"] is None or token["misc"].get("SpaceAfter") != "No")
                    # Skip tokens that span a range of positions
                    if isinstance(token["id"], int):
                        tokens_out.append(token)
                yield tokens_out


def iter_docs(paths):
    doc = []
    for sentence in iter_sentences(paths):
        if doc and sentence[0]["doc_id"] != doc[-1]["doc_id"]:
            yield doc
            doc = []
        doc.extend(sentence)
    yield doc


CONLLU_FIELDS = ["form", "lemma", "upos", "xpos", "deprel", "feats", "id", "head"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_path", type=str, nargs="+")
    parser.add_argument("output_path", type=str)
    parser.add_argument("--tokenizer-path", type=str, required=True)
    parser.add_argument("--feature-prefix", type=str, default="ud_")
    args = parser.parse_args()

    tokenizer = transformers.AutoTokenizer.from_pretrained(args.tokenizer_path)

    dataset = collections.defaultdict(list)
    for doc in iter_docs(args.input_path):
        # Re-tokenize the document
        text = "".join(token["form"] + (" " if token["space_after"] else "") for token in doc)
        text = text.rstrip(" ")
        assert BAD_CHAR not in text
        tokenized = tokenizer(text)
        for key in tokenized.keys():
            dataset[key].append(tokenized[key])
        tokens = tokenizer.convert_ids_to_tokens(tokenized["input_ids"])

        # Map the CoNLL-U annotations to the new tokens in such a way that every new token gets
        # the annotations of the first original token that overlaps with it.
        for field in CONLLU_FIELDS:
            dataset[args.feature_prefix + field].append([])
        dataset["sent_id"].append([])
        queue_src, queue_tgt = collections.deque(doc), collections.deque(tokens)
        token_src, tokens_tgt = None, []
        len_src, len_tgt = 0, 0
        while queue_src or queue_tgt:
            if len_src <= len_tgt:
                if token_src and token_src["space_after"]:
                    len_src += 1
                token_src = queue_src.popleft()
                len_src += len(token_src["form"])
                continue

            # Sometimes a single character is encoded as more than one token, and trying to
            # decode only one of the tokens will produce a U+FFFD (BAD_CHAR), which would mess up
            # the character counters. Therefore we need to accumulate tokens until we have a
            # sequence that can be decoded successfully (with no BAD_CHAR).
            tokens_tgt.append(queue_tgt.popleft())
            tokens_tgt_str = tokenizer.convert_tokens_to_string(tokens_tgt)
            if BAD_CHAR not in tokens_tgt_str:
                len_tgt += len(tokens_tgt_str)
                tokens_tgt = []

                # Sanity check: the beginning of the new token should overlap with the original token
                overlap_max_len = len_src - (len_tgt - len(tokens_tgt_str))
                assert tokens_tgt_str[:overlap_max_len].strip() in token_src["form"]

            # Update the dataset with the aligned annotations
            for field in CONLLU_FIELDS:
                dataset[args.feature_prefix + field][-1].append(token_src[field])
            dataset["sent_id"][-1].append(token_src["sent_id"])

        assert len_src == len_tgt
        assert len(set(len(v) for v in dataset.values())) == 1

    dataset = datasets.Dataset.from_dict(dataset)
    dataset.save_to_disk(args.output_path)


if __name__ == "__main__":
    main()