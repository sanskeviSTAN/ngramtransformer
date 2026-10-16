Example commands:
```
n_gram.py --n=3 --experiment_name=trigram --num_samples=100
transformer.py --num_layers=4 --hidden_dim=256 --experiment_name=transformer
```

Arguments for n_gram:
--tokenization_level (what level to tokenize) (default='character')
--n (value of n for the n-gram model) (default=6)
--experiment_name
--num_samples (default=10)
--max_steps (max output length) (default=40)

Arguments for transformer: 
--num_layers (default=4)
--hidden_dim (default=256)
--num_heads (default=4)
--ff_dim (feedforward layer dim) (default=256)
--dropout_p (dropout probability) (default=0.1)
--experiment_name
--num_samples (default=10)
--max_steps (max output length) (default=40)