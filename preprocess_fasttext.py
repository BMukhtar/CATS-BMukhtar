import fasttext
import json
import numpy as np
import config

model = fasttext.load_model(config.bin_path_en)
vocab = {}
vectors = []
for word in model.words:
    vocab[word] = len(vocab)
    vectors.append(model[word])

with open(config.vocab_path_en, 'w') as f:
    json.dump(vocab, f, indent=4)

with open(config.vecs_path_en, 'wb') as f:
    np.save(f, vectors)

print("saved")
