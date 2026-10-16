import os
import sys
import argparse
from typing import Dict, List, Any
from tqdm import tqdm
from collections import Counter
import numpy as np

from load_data_ngram import load_data_ngram as load_data
from perplexity import evaluate_perplexity
from wer import rerank_sentences_for_wer, compute_wer

def get_args():
    parser = argparse.ArgumentParser(description='n-gram model')
    parser.add_argument('--tokenization_level', type=str, default='character',
                        help="At what level to tokenize the input data")
    parser.add_argument('--n', type=int, default=6,
                        help="The value of n to use for the n-gram model")
    parser.add_argument('--experiment_name', type=str, default='testing',
                        help="What should we name our experiment?")
    parser.add_argument('--num_samples', type=int, default=10,
                        help="How many samples should we get from our model??")
    parser.add_argument('--max_steps', type=int, default=40,
                        help="What should the maximum output length of our samples be?")

    args = parser.parse_args()
    return args

class NGramLM():
    """
    N-gram language model
    """

    def __init__(self, n: int,k=0.0):
        """
        Initializes the n-gram model.
        Inputs:
            n (int): The value of n to use in the n-gram model
        """
        self.n = n
        self.k = k
        self.ngram_counts = Counter()
        self.context_counts = Counter()
        self.vocab = set()
        self.vocab_size = 0
        self.context_next_tokens = {}
        self.context_next_weights = {}
        self.order_ngram_counts = {}
        self.order_context_counts = {}

    def _events(self, seq: List[Any]):
        """
        Yield (context, token) pairs from a token sequence.
        Assumes any BOS/EOS tokens were added before calling this class.
        """
        if self.n == 1:
            for token in seq:
                yield tuple(), token
        else:
            for i in range(self.n - 1, len(seq)):
                context = tuple(seq[i - self.n + 1:i])
                token = seq[i]
                yield context, token

    def log_probability(self, model_input: List[Any], base=np.e):
        """
        Returns the log-probability of the provided model input.

        Inputs:
            model_input (List[Any]): The list of tokens associated with the input text.
            base (float): The base with which to compute the log-probability
        """
        if len(model_input) == 0:
            return 0.0

        log_p = 0.0

        for i, token in enumerate(model_input):
            if token not in self.vocab:
                token = "<unk>"

            context_len = min(self.n - 1, i)
            if context_len == 0:
                context = tuple()
            else:
                context = tuple(model_input[i - context_len:i])

            order = context_len + 1
            ngram_count = self.order_ngram_counts[order][(context, token)]
            context_count = self.order_context_counts[order][context]

            if self.k == 0.0:
                if ngram_count == 0 or context_count == 0:
                    return -np.inf
                prob = ngram_count / context_count
            else:
                prob = (ngram_count + self.k) / (context_count + self.k * self.vocab_size)

            log_p += np.log(prob) / np.log(base)

        return log_p

    def generate(self, num_samples: int, max_steps: int, results_file: str):
        """
        Generate text by ancestral sampling and save one sample per line.
        """
        directory = os.path.dirname(results_file)
        if directory:
            os.makedirs(directory, exist_ok=True)

        generations = []
        bos_token = "<bos>" if "<bos>" in self.vocab else None

        for _ in range(num_samples):
            if self.n > 1 and bos_token is not None:
                sample = [bos_token] * (self.n - 1)
            else:
                sample = []

            for _ in range(max_steps):
                if self.n == 1:
                    context = tuple()
                else:
                    if len(sample) < self.n - 1:
                        break
                    context = tuple(sample[-(self.n - 1):])

                candidates = self.context_next_tokens.get(context, [])
                weights = self.context_next_weights.get(context, [])

                if len(candidates) == 0:
                    break

                probs = np.array(weights, dtype=float)
                probs = probs / probs.sum()
                next_token = np.random.choice(candidates, p=probs)
                sample.append(next_token)

                if next_token == "<eos>":
                    break

            output_tokens = list(sample)

            while len(output_tokens) > 0 and output_tokens[0] == "<bos>":
                output_tokens = output_tokens[1:]
            if len(output_tokens) > 0 and output_tokens[-1] == "<eos>":
                output_tokens = output_tokens[:-1]

            generations.append(" ".join(str(tok) for tok in output_tokens))

        with open(results_file, "w", encoding="utf-8") as f:
            for line in generations:
                f.write(line + "\n")

    def learn(self, training_data: List[List[Any]]):
        """
        Function for learning n-grams from the provided training data. You may
        add keywords to this function as needed, provided that the default behavior
        is that of an n-gram model without any label smoothing.

        Inputs:
            training_data (List[List[Any]]): A list of model inputs, which should each be lists of input tokens
        """
        self.ngram_counts = Counter()
        self.context_counts = Counter()
        self.vocab = set()
        self.context_next_tokens = {}
        self.context_next_weights = {}

        self.order_ngram_counts = {order: Counter() for order in range(1, self.n + 1)}
        self.order_context_counts = {order: Counter() for order in range(1, self.n + 1)}

        for seq in training_data:
            for tok in seq:
                self.vocab.add(tok)

            for context, token in self._events(seq):
                self.ngram_counts[(context, token)] += 1
                self.context_counts[context] += 1

            for i, token in enumerate(seq):
                max_order = min(self.n, i + 1)
                for order in range(1, max_order + 1):
                    context_len = order - 1
                    if context_len == 0:
                        context = tuple()
                    else:
                        context = tuple(seq[i - context_len:i])

                    self.order_ngram_counts[order][(context, token)] += 1
                    self.order_context_counts[order][context] += 1

        self.vocab.add("<unk>")
        self.vocab_size = len(self.vocab)

        for (context, token), count in self.ngram_counts.items():
            if context not in self.context_next_tokens:
                self.context_next_tokens[context] = []
                self.context_next_weights[context] = []
            self.context_next_tokens[context].append(token)
            self.context_next_weights[context].append(count)
def main():
    # Get key arguments
    args = get_args()

    # Get the data for language-modeling and WER computation
    tokenization_level = args.tokenization_level
    model_type = "n_gram"
    train_data, val_data, dev_data, test_data, dev_wer_data, test_wer_data = load_data(tokenization_level)

    # Initialize and "train" the n-gram model
    n = args.n
    model = NGramLM(n, k=0.1)
    model.learn(train_data)

    # Evaluate model perplexity
    val_perplexity = evaluate_perplexity(model, val_data)
    print(f'Model perplexity on the val set: {val_perplexity}')
    dev_perplexity = evaluate_perplexity(model, dev_data)
    print(f'Model perplexity on the dev set: {dev_perplexity}')
    test_perplexity = evaluate_perplexity(model, test_data)
    print(f'Model perplexity on the test set: {test_perplexity}')

    # Evaluate model WER
    experiment_name = args.experiment_name
    dev_wer_savepath = os.path.join('results', f'{experiment_name}_n_gram_dev_wer_predictions.csv')
    rerank_sentences_for_wer(model, dev_wer_data, dev_wer_savepath)
    dev_wer = compute_wer('data/wer_data/dev_ground_truths.csv', dev_wer_savepath)
    print("Dev set WER was: ", dev_wer)

    test_wer_savepath = os.path.join('results', f'{experiment_name}_n_gram_test_wer_predictions.csv')
    rerank_sentences_for_wer(model, test_wer_data, test_wer_savepath)

    # Generate text from the model
    generation_path = os.path.join('generations', f'{experiment_name}_n_gram_generation_examples.txt')
    num_samples = args.num_samples
    max_steps = args.max_steps
    model.generate(num_samples, max_steps, generation_path)

if __name__ == "__main__":
    main()
    
