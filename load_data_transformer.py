import os
import random
import json
import torch
from torch.utils.data import DataLoader, Dataset

import re
from tokenizers import Tokenizer

from typing import List, Any, Tuple

def basic_tokenize(text, level):
    if level=="words":
        return text.split()
    elif level=="character":
        return list(text)
    else:
        raise ValueError(f"Unsupported non-BPE tokenization_level: {level}")


def load_data_transformer(tokenization_level: str):
    """
    Function for loading data for language modeling and WER computation for the transformer models.
    Inputs:
        tokenization_level (str): The level at which to tokenize the input
    """

    def read_lines(path):
        with open(path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]

    lm_dir = "data\lm_data"
    wer_dir = "data\wer_data"

    lm_train_raw = read_lines(os.path.join(lm_dir, "treebank-sentences-train.txt"))
    lm_dev_raw = read_lines(os.path.join(lm_dir, "treebank-sentences-dev.txt"))
    lm_test_raw = read_lines(os.path.join(lm_dir, "treebank-sentences-test.txt"))

    train_tokens = [basic_tokenize(s, tokenization_level) for s in lm_train_raw]

    vocab = {"<bos>": 0, "<eos>": 1, "<unk>": 2}
    for sent in train_tokens:
        for tok in sent:
            if tok not in vocab:
                vocab[tok] = len(vocab)

    vocab["<pad>"] = len(vocab)

    def encode_text(text):
        toks = basic_tokenize(text, tokenization_level)
        ids = [vocab["<bos>"]]
        for tok in toks:
            ids.append(vocab.get(tok, vocab["<unk>"]))
        ids.append(vocab["<eos>"])
        return ids

    lm_train = [encode_text(s) for s in lm_train_raw]
    lm_dev_all = [encode_text(s) for s in lm_dev_raw]
    lm_test = [encode_text(s) for s in lm_test_raw]

    split_idx = len(lm_dev_all) // 2
    if split_idx == 0:
        val_data = lm_dev_all
        dev_data = lm_dev_all
    else:
        val_data = lm_dev_all[:split_idx]
        dev_data = lm_dev_all[split_idx:]
        if len(dev_data) == 0:
            dev_data = val_data

    def load_candidate_json(path):
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        out = []
        for ex_id, item in raw.items():
            raw_sentences = item["sentences"]
            acoustic_scores = item.get("acoustic_scores", item.get("scores", [0.0] * len(raw_sentences)))
            encoded_sentences = [encode_text(s) for s in raw_sentences]

            out.append({
                "id": ex_id,
                "sentences": raw_sentences,
                "encoded_sentences": encoded_sentences,
                "acoustic_scores": acoustic_scores,
            })
        return out

    wer_dev_data = load_candidate_json(os.path.join(wer_dir, "dev_sentences.json"))
    wer_test_data = load_candidate_json(os.path.join(wer_dir, "test_sentences.json"))

    return lm_train, val_data, dev_data, lm_test, wer_dev_data, wer_test_data