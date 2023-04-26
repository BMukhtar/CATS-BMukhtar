import json
import numpy as np
import config
from tqdm import tqdm
import io

vocab = {}
vectors = []


fin = io.open(config.vec_path_en, 'r', encoding='utf-8', newline='\n', errors='ignore')
n, d = map(int, fin.readline().split())
assert d == config.vecs_dim
counter = 0
for line in tqdm(fin):
    tokens = line.rstrip().split(' ')
    vector = list(map(float, tokens[1:]))
    assert len(vector) == config.vecs_dim
    vectors.append(vector)
    vocab[tokens[0]] = counter
    counter += 1

# add special tokens to vocab pad, unk, sos, eos
tokens = ["<PAD>", "<UNK>", "<SOS>", "<EOS>", "<S/>"]
for token in tokens:
    vocab[token] = len(vocab)
    vectors.append(np.random.rand(config.vecs_dim))

with open(config.vocab_path_en, 'w') as f:
    json.dump(vocab, f, indent=4)

with open(config.vecs_path_en, 'wb') as f:
    np.save(f, vectors)

print("saved")
