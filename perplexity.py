from typing import List, Any, Tuple
import torch

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def evaluate_perplexity(model: Any, data: Any):
    """
    Function for computing perplexity on the provided dataset.

    Inputs:
        model (Any): An n-gram or Transformer model.
        data (Any):  Data in the form suitable for the input model.
    """
    m = num_tokens_in_corpus(model, data)
    l = corpus_log_probability(model, data)
    return 2 ** (- l / m)

def num_tokens_in_corpus(model, data):
    """
    Helper function returning the number of tokens in the corpus
    """
    if type(data) == list:
        total = sum([len(dp) for dp in data])
    else:
        padding_idx = model.padding_idx

        total = 0
        for input_tokens, target_tokens in data:
            total += torch.sum(input_tokens != padding_idx).item()
            total += torch.sum(target_tokens[:, -1] != padding_idx).item()
    return total

def corpus_log_probability(model, data):
    """
    Helper function for the total log-probability of the input corpus
    """
    log_p = 0
    if type(data) == list:
        for datapoint in data:
            log_p += model.log_probability(datapoint, base=2)
    else:
        for input_tokens, target_tokens in data:
            input_tokens = input_tokens.to(DEVICE)
            target_tokens = target_tokens.to(DEVICE)
            log_p += torch.sum(model.log_probability(input_tokens, target_tokens, base=2)).item()
    return log_p