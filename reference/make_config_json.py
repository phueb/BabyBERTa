import json

config = {
  "hidden_size": 256,
  "num_attention_heads": 8,
  "num_hidden_layers": 8,
  "vocab_size": 8192,
  "intermediate_size": 1024,
  "max_position_embeddings": 128
}
with open("config.json", 'w') as fp:
    json.dump(config, fp)
