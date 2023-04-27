import fasttext
import json
import numpy as np
import config

model = fasttext.load_model(config.bin_path_en)
vocab = {}
vectors = []
max_tokens = 500000
for word in model.words[:max_tokens]:
    vocab[word] = len(vocab)
    vectors.append(model[word])

# add special tokens to vocab pad, unk, sos, eos
tokens = ["<PAD>", "<UNK>", "<SOS>", "<EOS>", "<S/>", "<SS>", "<SSS>"]
for token in tokens:
    vocab[token] = len(vocab)
    vectors.append(np.random.rand(config.vecs_dim))

with open(config.vocab_path_en, 'w') as f:
    json.dump(vocab, f, indent=4)

with open(config.vecs_path_en, 'wb') as f:
    np.save(f, vectors)

print("saved")

#%%
