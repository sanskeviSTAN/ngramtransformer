import os
import random
import torch
from torch.utils.data import DataLoader, Dataset
import json
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
import re
from tokenizers import Tokenizer
from typing import List, Any, Tuple

def load_data_ngram(tokenization_level: str):


    def read_lines(path):
        with open(path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]

    lm_dir = "data\lm_data"
    wer_dir = "data\wer_data"

    lm_train_raw = read_lines(os.path.join(lm_dir, "treebank-sentences-train.txt"))
    lm_dev_raw = read_lines(os.path.join(lm_dir, "treebank-sentences-dev.txt"))
    lm_test_raw = read_lines(os.path.join(lm_dir, "treebank-sentences-test.txt"))

    hf_tokenizer = None
    if tokenization_level=="subword":
        hf_tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
        hf_tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        trainer = trainers.BpeTrainer(
            vocab_size=2000,
            special_tokens=["<unk>"]
        )
        hf_tokenizer.train_from_iterator(lm_train_raw, trainer=trainer)

    def tokenize_text(text, level, tokenizer=None):
        if level=="character":
            return list(text)
        elif level=="subword":
            return tokenizer.encode(text).tokens
        else:
            raise ValueError(f"Unsupported tokenization_level: {level}")

    lm_train = [tokenize_text(s, tokenization_level, hf_tokenizer) for s in lm_train_raw]

    train_vocab = set()
    for sent in lm_train:
        for tok in sent:
            train_vocab.add(tok)
    train_vocab.add("<unk>")

    def map_unk(tokens):
        return [tok if tok in train_vocab else "<unk>" for tok in tokens]

    lm_dev_all = [map_unk(tokenize_text(s, tokenization_level, hf_tokenizer)) for s in lm_dev_raw]
    lm_test = [map_unk(tokenize_text(s, tokenization_level, hf_tokenizer)) for s in lm_test_raw]

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
            tokenized_sentences = [map_unk(tokenize_text(s, tokenization_level, hf_tokenizer)) for s in raw_sentences]

            out.append({
                "id": ex_id,
                "sentences": raw_sentences,
                "encoded_sentences": tokenized_sentences,
                "acoustic_scores": acoustic_scores,
            })
        return out
    wer_dev_candidates = load_candidate_json(os.path.join(wer_dir, "dev_sentences.json"))
    wer_test_candidates = load_candidate_json(os.path.join(wer_dir, "test_sentences.json"))

    return (
        lm_train,
        val_data,
        dev_data,
        lm_test,
        wer_dev_candidates,
        wer_test_candidates,
    )