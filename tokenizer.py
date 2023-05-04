from sentence_transformers import SentenceTransformer

model_name = 'all-MiniLM-L6-v2'
model = SentenceTransformer(model_name)

def process(examples):
    res = []
    for example in examples:
        id = example['id']
        sentences = example['sentences']
        embds = model.encode(sentences)
        out = {'id': id, 'embeddings': embds}
        res.append(out)
    return res
