import json
import faiss
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def read_dataset(path, tokenizer):
    dataset = json.load(open(path))
    for sequence in dataset:
        original_tokens, _, _ = tokenizer.split(' '.join(sequence["tokens"]))
        selected_embeddings, embedding_tokens = tokenizer.select_embeddings(original_tokens)
        labels = _fetch_labels(original_tokens, sequence["entities"])

        yield original_tokens, labels, selected_embeddings, embedding_tokens

def read_faiss(path, device=DEVICE):
    index = faiss.read_index(path+'faiss_index')
    if device == "cuda":
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)
         
    with open(path+'faiss_index_table.json', 'r', encoding='utf-8') as json_table:
        index_table = json.load(json_table)

    return index, index_table

def _fetch_labels(tokens, entities, no_entity='O'):
    labels = [set() for token in tokens]
    for entity in entities:
        for labelset in labels[entity["start"]:entity["end"]]:
            labelset.add(entity["type"])
    for labelset in labels:
        if not labelset:
            labelset.add(no_entity) 

    return labels
    


