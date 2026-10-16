from typing import List, Any, Tuple
import pandas as pd
import numpy as np
import torch
from evaluate import load
def rerank_sentences_for_wer(model: Any, wer_data: List[Any], savepath: str):
    """
    Function to rerank candidate sentences in the HUB dataset. For each set of sentences,
    you must assign each sentence a score in the form of the sentence's acoustic score plus
    the sentence's log probability. You should then save the top scoring sentences in a .csv
    file similar to those found in the results directory.

    Inputs:
        model (Any): An n-gram or Transformer model.
        wer_data (List[Any]): Processed data from the HUB dataset.
        savepath (str): The path to save the csv file pairing sentence set ids and the top ranked sentences.
    """
    rows = []

    for example in wer_data:
        ex_id = example["id"]
        raw_sentences = example["sentences"]
        scored_sentences = example.get("encoded_sentences", raw_sentences)
        acoustic_scores = example.get("acoustic_scores", example.get("scores", [0.0] * len(raw_sentences)))
        best_score = -float("inf")
        best_sentence = None

        for raw_sentence, model_input, acoustic_score in zip(raw_sentences, scored_sentences, acoustic_scores):
            lm_score = model.log_probability(model_input)
            if torch.is_tensor(lm_score):
                lm_score = lm_score.item()

            total_score = float(acoustic_score) + float(lm_score)

            if total_score > best_score:
                best_score = total_score
                best_sentence = raw_sentence

        rows.append({"id": ex_id, "sentences": best_sentence})

    pd.DataFrame(rows).to_csv(savepath, index=False)

def compute_wer(gt_path, model_path):
    # Load the sentences
    ground_truths = pd.read_csv(gt_path)['sentences'].tolist()
    guesses = pd.read_csv(model_path)['sentences'].tolist()

    # Compute wer
    wer = load("wer")
    wer_value = wer.compute(predictions=guesses, references=ground_truths)
    return wer_value
