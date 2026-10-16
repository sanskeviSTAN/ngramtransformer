import os
import sys
import argparse
from typing import Dict, List
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from load_data_transformer import load_data_transformer as load_data
from perplexity import evaluate_perplexity
from wer import rerank_sentences_for_wer, compute_wer

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class CharacterLevelTransformer(nn.Module):

    def __init__(self, num_layers: int, hidden_dim: int, num_heads: int,
                 ff_dim: int, dropout: float, vocab_size: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embed = nn.Embedding(vocab_size, hidden_dim, padding_idx=vocab_size - 1)
        self.pos_embed = PositionalEncoding(hidden_dim, dropout)
        self.decoder = Decoder(num_layers, hidden_dim, num_heads, ff_dim, dropout)
        self.lm_head = nn.Linear(hidden_dim, vocab_size)
        self.padding_idx = vocab_size - 1

    def log_probability(self, input_tokens_BL: torch.Tensor, target_tokens_BL: torch.Tensor = None, base=np.e):
        """
        Computes the log-probabilities for the inputs in the given minibatch.

        Input:
            input_tokens_BL (torch.Tensor): A tensor of shape (B, L), where B is the
                                         batch-size and L is the input length.
            target_tokens_BL (torch.Tensor): A tensor of shape (B, L). For a given (i, j),
                                          target_tokens[i, j] should be the token following
                                          input_tokens[i, j]
        Output (torch.Tensor): A tensor of shape (B,) containing the log-probability for each
                               example in the minibatch
        """
        if target_tokens_BL is None:
            if isinstance(input_tokens_BL, list):
                if len(input_tokens_BL) <= 1:
                    return 0.0
                input_tokens_BL = torch.tensor(input_tokens_BL[:-1], dtype=torch.long, device=DEVICE).unsqueeze(0)
                target_tokens_BL = torch.tensor(input_tokens_BL.squeeze(0).tolist()[1:] + [], dtype=torch.long,
                                                device=DEVICE)
                target_tokens_BL = torch.tensor(input_tokens_BL.squeeze(0).tolist()[1:] + [], dtype=torch.long,
                                                device=DEVICE)
            elif torch.is_tensor(input_tokens_BL):
                if input_tokens_BL.dim() == 1:
                    if input_tokens_BL.size(0) <= 1:
                        return 0.0
                    seq = input_tokens_BL.to(DEVICE)
                    input_tokens_BL = seq[:-1].unsqueeze(0)
                    target_tokens_BL = seq[1:].unsqueeze(0)
                else:
                    raise ValueError("If target_tokens_BL is None, input_tokens_BL must be a 1D sequence.")
            else:
                raise ValueError("Unsupported input type for log_probability.")

        if isinstance(input_tokens_BL, list):
            input_tokens_BL = torch.tensor(input_tokens_BL, dtype=torch.long, device=DEVICE)
        if isinstance(target_tokens_BL, list):
            target_tokens_BL = torch.tensor(target_tokens_BL, dtype=torch.long, device=DEVICE)

        if input_tokens_BL.dim() == 1:
            input_tokens_BL = input_tokens_BL.unsqueeze(0)
        if target_tokens_BL.dim() == 1:
            target_tokens_BL = target_tokens_BL.unsqueeze(0)

        logits_BLV = self.forward(input_tokens_BL)
        log_probs_BLV = F.log_softmax(logits_BLV, dim=-1)

        target_log_probs_BL = torch.gather(
            log_probs_BLV, dim=-1, index=target_tokens_BL.unsqueeze(-1)
        ).squeeze(-1)

        valid_mask_BL = (target_tokens_BL != self.padding_idx)
        seq_log_probs_B = (target_log_probs_BL * valid_mask_BL).sum(dim=1)

        if base != np.e:
            seq_log_probs_B = seq_log_probs_B / math.log(base)

        if seq_log_probs_B.numel() == 1:
            return seq_log_probs_B.squeeze(0)

        return seq_log_probs_B

    def forward(self, model_input_BL):
        # Perform the embedding
        embeds_BLH = self.embed(model_input_BL) * math.sqrt(self.hidden_dim)
        embeds_BLH = self.pos_embed(embeds_BLH)

        # Pass through the decoder
        mask_1LL = construct_self_attn_mask(model_input_BL)
        decoder_output_BLH = self.decoder(embeds_BLH, mask_1LL)
        output_BLV = self.lm_head(decoder_output_BLH)
        return output_BLV

    def generate(self, num_samples: int, max_steps: int, results_file: str):
        directory = os.path.dirname(results_file)
        if directory:
            os.makedirs(directory, exist_ok=True)

        self.eval()
        generations = []

        start_token = 0
        if start_token == self.padding_idx and self.padding_idx > 0:
            start_token = 1

        with torch.no_grad():
            for _ in range(num_samples):
                generated = [start_token]

                for _ in range(max_steps - 1):
                    input_tokens = torch.tensor(generated, dtype=torch.long, device=DEVICE).unsqueeze(0)
                    logits_BLV = self.forward(input_tokens)
                    next_token = torch.argmax(logits_BLV[0, -1], dim=-1).item()

                    if next_token == self.padding_idx:
                        break

                    generated.append(next_token)

                generations.append(" ".join(str(tok) for tok in generated))

        with open(results_file, "w", encoding="utf-8") as f:
            for line in generations:
                f.write(line + "\n")


def construct_self_attn_mask(x_BL: torch.Tensor):
    """
    Constructs a causal self-attention mask. The output to this function should be a mask of shape
    (1, L, L). Indices that a token can attend to should be
    set to true.

    - Row index (i): Represents the token position.
    - Column index (j): Represents whether token i can attend to token j.
    - A token at position i can only attend to tokens ≤ i.
    """
    L = x_BL.size(1)
    all_ones_LL = torch.ones(L, L, dtype=torch.bool, device=x_BL.device)

    mask_1LL = torch.tril(all_ones_LL)
    mask_1LL = mask_1LL.unsqueeze(0)
    return mask_1LL.to(x_BL.device)


class Decoder(nn.Module):
    def __init__(self, num_layers, hidden_dim, num_heads, ff_dim, dropout):
        super().__init__()
        layers = []
        for i in range(num_layers):
            layers.append(TransformerBlock(num_heads, hidden_dim, ff_dim, dropout).to(DEVICE))
        self.layers = nn.ModuleList(layers)

    def forward(self, x_BLH, mask_1LL):
        for layer in self.layers:
            x_BLH = layer(x_BLH, mask_1LL)
        return x_BLH


class TransformerBlock(nn.Module):
    def __init__(self, num_heads, hidden_dim, ff_dim, dropout):
        super().__init__()

        # Attention block
        self.attn_block = MultiHeadAttention(num_heads, hidden_dim, dropout)
        self.attn_dropout = nn.Dropout(dropout)
        self.attn_norm = nn.LayerNorm(hidden_dim)

        # Feedforward block
        self.mlp_block = TransformerMLP(hidden_dim, ff_dim, dropout)
        self.mlp_dropout = nn.Dropout(dropout)
        self.mlp_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x_BLH, mask_1LL):
        """
        There are two types of errors in this function.
        """
        attn_out_BLH = self.attn_block(x_BLH, mask_1LL)
        x_BLH = self.attn_norm(x_BLH + self.attn_dropout(attn_out_BLH))

        mlp_out_BLH = self.mlp_block(x_BLH)
        x_BLH = self.mlp_norm(x_BLH + self.mlp_dropout(mlp_out_BLH))
        return x_BLH


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, hidden_dim, dropout=0.1):
        super().__init__()
        self.h = num_heads
        self.qkv_dim = hidden_dim // num_heads
        self.dropout = nn.Dropout(dropout)

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    def attention(self, query_BHQL, key_BHQL, value_BHQL, mask_1LL):
        """
        There are three errors in this function to fix.
        """
        dot_products_BHLL = torch.matmul(query_BHQL, key_BHQL.transpose(-2, -1)) / math.sqrt(self.qkv_dim)
        dot_products_BHLL = dot_products_BHLL.masked_fill(~mask_1LL, float("-inf"))
        attn_BHLL = self.dropout(F.softmax(dot_products_BHLL, dim=-1))
        return torch.matmul(attn_BHLL, value_BHQL)

    def forward(self, x_BLH, mask_1LL):
        """
        There are two errors in this function to fix
        """
        mask_B1LL = mask_1LL.unsqueeze(1)
        B = x_BLH.size(0)

        query_BHQL = self.q_proj(x_BLH).view(B, -1, self.h, self.qkv_dim).transpose(1, 2)
        key_BHQL = self.k_proj(x_BLH).view(B, -1, self.h, self.qkv_dim).transpose(1, 2)
        value_BHQL = self.v_proj(x_BLH).view(B, -1, self.h, self.qkv_dim).transpose(1, 2)

        x_BHQL = self.attention(query_BHQL, key_BHQL, value_BHQL, mask_B1LL)

        x_BLH = x_BHQL.transpose(1, 2).contiguous().view(B, -1, self.h * self.qkv_dim)
        return self.out_proj(x_BLH)


class TransformerMLP(nn.Module):
    def __init__(self, hidden_dim, ff_dim, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, ff_dim)
        self.fc2 = nn.Linear(ff_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_BLH):
        """
        There is a single error in this function to fix.
        """
        return self.fc2(self.dropout(F.relu(self.fc1(x_BLH))))


class PositionalEncoding(nn.Module):
    def __init__(self, hidden_dim, dropout, max_len=1000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        positional_encodings_LH = torch.zeros(max_len, hidden_dim)
        position_L1 = torch.arange(0, max_len).unsqueeze(1)
        div_term_H = torch.exp(torch.arange(0, hidden_dim, 2) * (- math.log(10000) / hidden_dim))
        positional_encodings_LH[:, 0::2] = torch.sin(position_L1 * div_term_H)
        positional_encodings_LH[:, 1::2] = torch.cos(position_L1 * div_term_H)
        positional_encodings_1LH = positional_encodings_LH.unsqueeze(0)

        self.register_buffer('positional_encodings', positional_encodings_1LH, persistent=False)

    def forward(self, x_BLH):
        x_BLH = x_BLH + self.positional_encodings[:, :x_BLH.size(1)]
        return self.dropout(x_BLH)


def train(model, train_data, val_data, dev_wer_data, loss_fct, optimizer, max_epochs):
    """
    Training loop for the transformer model. You may change the header as you see fit.
    """
    loss_fct = nn.CrossEntropyLoss(ignore_index=model.padding_idx)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    max_epochs = 5

    def iter_batches(data, batch_size=32):
        if type(data) == list:
            indices = np.arange(len(data))
            np.random.shuffle(indices)

            for start in range(0, len(indices), batch_size):
                batch_indices = indices[start:start + batch_size]
                batch_sequences = [data[i] for i in batch_indices if len(data[i]) >= 2]

                if len(batch_sequences) == 0:
                    continue

                batch_sequences = [
                    seq if torch.is_tensor(seq) else torch.tensor(seq, dtype=torch.long)
                    for seq in batch_sequences
                ]

                input_seqs = [seq[:-1] for seq in batch_sequences]
                target_seqs = [seq[1:] for seq in batch_sequences]

                input_tokens = torch.nn.utils.rnn.pad_sequence(
                    input_seqs, batch_first=True, padding_value=model.padding_idx
                )
                target_tokens = torch.nn.utils.rnn.pad_sequence(
                    target_seqs, batch_first=True, padding_value=model.padding_idx
                )

                yield input_tokens, target_tokens
        else:
            for input_tokens, target_tokens in data:
                yield input_tokens, target_tokens

    best_val_perplexity = float("inf")
    best_state_dict = None

    for epoch in range(max_epochs):
        model.train()
        total_loss = 0.0
        total_tokens = 0

        for input_tokens, target_tokens in tqdm(iter_batches(train_data), desc=f"Epoch {epoch + 1}/{max_epochs}"):
            input_tokens = input_tokens.to(DEVICE)
            target_tokens = target_tokens.to(DEVICE)

            optimizer.zero_grad()
            logits_BLV = model(input_tokens)
            loss = loss_fct(logits_BLV.view(-1, logits_BLV.size(-1)), target_tokens.view(-1))
            loss.backward()
            optimizer.step()

            num_valid_tokens = (target_tokens != model.padding_idx).sum().item()
            total_loss += loss.item() * max(num_valid_tokens, 1)
            total_tokens += num_valid_tokens

        model.eval()
        with torch.no_grad():
            val_perplexity = evaluate_perplexity(model, val_data)

        avg_loss = total_loss /max(total_tokens, 1)
        print(f"Epoch {epoch + 1}: train_loss={avg_loss:.4f} val_perplexity={val_perplexity}")

        if val_perplexity < best_val_perplexity:
            best_val_perplexity = val_perplexity
            best_state_dict = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)


def get_args():

    parser = argparse.ArgumentParser(description='Transformer model')
    parser.add_argument('--num_layers', type=int, default=4,
                        help="How many transformer blocks to use")
    parser.add_argument('--hidden_dim', type=int, default=256,
                        help="What is the transformer hidden dimension")
    parser.add_argument('--num_heads', type=int, default=4,
                        help="How many heads to use for Multihead Attention")
    parser.add_argument('--ff_dim', type=int, default=256,
                        help="What is the intermediate dimension for the feedforward layer")
    parser.add_argument('--dropout_p', type=float, default=0.1,
                        help="The dropout probability to use")

    parser.add_argument('--experiment_name', type=str, default='testing_')
    parser.add_argument('--num_samples', type=int, default=10,
                        help="How many samples should we get from our model?")
    parser.add_argument('--max_steps', type=int, default=40,
                        help="What should the maximum output length be?")

    args = parser.parse_args()
    return args


def main():
    # Get key arguments
    args = get_args()

    # Get the data
    tokenization_level = "character"
    model_type = "transformer"
    train_data, val_data, dev_data, test_data, dev_wer_data, test_wer_data = load_data(tokenization_level)

    # Initialize the transformer and train
    num_layers = args.num_layers
    hidden_dim = args.hidden_dim
    num_heads = args.num_heads
    ff_dim = args.ff_dim
    dropout_p = args.dropout_p

    if type(train_data) == list:
        max_token_id = 0
        for seq in train_data:
            if len(seq) > 0:
                if torch.is_tensor(seq):
                    max_token_id = max(max_token_id, int(seq.max().item()))
                else:
                    max_token_id = max(max_token_id, max(seq))
        vocab_size = max_token_id + 2
    else:
        max_token_id = 0
        for input_tokens, target_tokens in train_data:
            max_token_id = max(
                max_token_id,
                int(torch.max(input_tokens).item()),
                int(torch.max(target_tokens).item())
            )
        vocab_size = max_token_id + 1

    model = CharacterLevelTransformer(num_layers, hidden_dim, num_heads, ff_dim,
                                      dropout_p, vocab_size).to(DEVICE)

    optimizer = None
    loss_fct = None
    max_epochs = None
    train(model, train_data, val_data, dev_wer_data, loss_fct, optimizer, max_epochs)

    # Evaluate model perplexity
    model.eval()
    val_perplexity = evaluate_perplexity(model, val_data)
    print(f'Model perplexity on the val set: {val_perplexity}')
    dev_perplexity = evaluate_perplexity(model, dev_data)
    print(f'Model perplexity on the dev set: {dev_perplexity}')
    test_perplexity = evaluate_perplexity(model, test_data)
    print(f'Model perplexity on the test set: {test_perplexity}')

    # Evaluate model WER
    experiment_name = args.experiment_name
    dev_wer_savepath = os.path.join('results', f'{experiment_name}transformer_dev_wer_predictions.csv')
    rerank_sentences_for_wer(model, dev_wer_data, dev_wer_savepath)
    dev_wer = compute_wer('data/wer_data/dev_ground_truths.csv', dev_wer_savepath)
    print("Dev set WER was: ", dev_wer)

    test_wer_savepath = os.path.join('results', f'{experiment_name}transformer_test_wer_predictions.csv')
    rerank_sentences_for_wer(model, test_wer_data, test_wer_savepath)

    # Generate text from the model
    generation_path = os.path.join('generations', f'{experiment_name}transformer_generation_examples.pkl')
    num_samples = args.num_samples
    max_steps = args.max_steps
    model.generate(num_samples, max_steps, generation_path)


if __name__ == "__main__":
    main()