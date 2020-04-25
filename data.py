import json
import faiss
import utils
import random
import torch
import os
from tqdm import tqdm
from typing import List, Dict, Type
from abc import ABC, abstractmethod

'''
Expected data format is a json with annotated sequences (sentences preferably)
like so:
[
	{
        tokens: [token, token, token],
        entities: [{“type”: type, “start”: start, “end”: end }, ...],
        relations: [{“type”: type, “head”: head, “tail”: tail }, ...],
        orig_id: sentence_hash (for example)
    },
    {
        tokens: [“CNNs”, “are”, “used”, “for”, “computer”, “vision”, “.”],
        entities: [{“type”: MLA, “start”: 0, “end”: 1 }, {“type”: AE, “start”: 4, “end”: 6 }],
		relations: [{“type”:usedFor, “head”: 0, “tail”: 1 }],
		orig_id: -234236432762328423
	}
]
'''


class Datapoint(ABC):
    def __init__(self, embedding, label):
        self.embedding = embedding
        self.label = label

    @abstractmethod
    def to_table_entry(self):
        pass
    
    @abstractmethod
    def calculate_embedding(self):
        pass
    
    @abstractmethod
    def __repr__(self):
        pass
    
    @abstractmethod
    def __str__(self):
        pass


class Token(Datapoint):
    def __init__(self, index: int, label='O', token=str, 
                 sentence_id='', embedding=None, embedding_tokens=[]):
        super().__init__(embedding, label)
        self.index = index
        self.token = token
        self.embedding_tokens = embedding_tokens
        self.sentence_id = sentence_id
        self.id = self.create_id(token, index)

    def create_id(self, token, index):
        return hash(token+str(index))   

    def to_table_entry(self):
        entry = {
            "index": self.index,
            "label": self.label, 
            "string": str(self), 
            "token": self.token,
            "embedding_tokens": self.embedding_tokens,
            "sentence_id": self.sentence_id,
            "id": self.id
        }

        return entry


    def calculate_embedding(self, embeddings, bert_tokens, orig2tok, accumulation_f="abs_max"):
        """
        Here it can be decided which embedding is selected
        to represent all (sub)tokens in the token. Several strategies can be applied.
        E.g. the first subtoken, an average of subtokens, or abs_max.
        """

        def _first(embeddings, bert_tokens, orig2tok):
            first, _ = orig2tok[self.index]
            embedding = embeddings[first]
            tokens = [bert_tokens[first]]

            return embedding, tokens
        
        def _abs_max(embeddings, bert_tokens, orig2tok):
            first, last = orig2tok[self.index]
            selected_embeddings = [embedding for embedding in embeddings[first:last]]
            t = torch.stack(selected_embeddings)
            abs_max_indices = torch.abs(t).argmax(dim=0)
            embedding = t.gather(0, abs_max_indices.view(1,-1)).squeeze()
            tokens = bert_tokens[first:last]

            return embedding, tokens

        def _mean(embeddings, bert_tokens, orig2tok):
            first, last = orig2tok[self.index]
            selected_embeddings = [embedding for embedding in embeddings[first:last]]
            embedding = torch.stack(selected_embeddings).mean(dim=0)
            tokens = bert_tokens[first:last]

            return embedding, tokens

        f_reduce = {"first": _first, "abs_max": _abs_max, "mean": _mean}.get(accumulation_f, _mean)
        self.embedding, self.embedding_tokens = f_reduce(embeddings, bert_tokens, orig2tok)

        return self.embedding

    def __repr__(self):
        return "Token()"

    def __str__(self):
        return "[TOKEN] {} >> {}".format(self.token, self.label)


def prepare_dataset(path, tokenizer, neg_label='O', f_reduce="abs_max"):
    dataset = json.load(open(path))
    for annotation in tqdm(dataset):
        # Get embeddings and token mappings
        string_tokens = adjust_annotation(annotation, tokenizer)   
        bert_tokens, tok2orig, orig2tok = tokenizer.tokenize_with_mapping(string_tokens, "bert")
        embeddings = tokenizer.embed(bert_tokens)[1:-1] # skip special tokens 

        # Create Tokens
        pos_tokens = _create_positive_tokens(string_tokens, annotation["entities"])
        neg_tokens = _create_negative_tokens(string_tokens, pos_tokens, neg_label)
        tokens = pos_tokens + neg_tokens
        for token in tokens: token.calculate_embedding(embeddings, bert_tokens, orig2tok, f_reduce)
        for token in tokens:
            print(token)
        yield tokens


def adjust_annotation(annotation: Dict, tokenizer):
    input_tokens = annotation["tokens"]
    tokens, _, input2split = tokenizer.tokenize_with_mapping(input_tokens, f_tokenize="split")
    for entity in annotation["entities"]:
        positions = input2split[entity["start"]:entity["end"]]
        new_start, new_end = positions[0][0], positions[-1][1]
        entity["start"], entity["end"] = new_start, new_end

    return tokens


def init_faiss(f_reduce: str, f_similarity: str, tokenizer):
    def _l2(size):
        return faiss.IndexFlatL2(size)
    def _ip(size):
        return faiss.IndexFlatIP(size)

    embedding_size = {
        "concat": tokenizer.embedding_size*2,
        "substract": tokenizer.embedding_size,
        "mean": tokenizer.embedding_size 
    }.get(f_reduce, tokenizer.embedding_size)

    index = {
        "L2": _l2,
        "IP": _ip
    }.get(f_similarity, _l2)

    return index(embedding_size)


def save_faiss(index, table, name, save_path="data/save/"):
    print("Saving {} index...".format(name))
    utils.create_dir(save_path)
    index_path = os.path.join(save_path, "{}_index".format(name))
    faiss.write_index(index, index_path)
    table_path = os.path.join(save_path, "{}_table.json".format(name))
    with open(table_path, 'w') as json_file:
        json.dump(table, json_file)
    print("Indexed {} {} with their labels".format(len(table), name))


def load_faiss(path, device, name):
    index_path = os.path.join(path, "{}_index".format(name))
    index = faiss.read_index(index_path)
    if device == "cuda":
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)
    
    table_path = os.path.join(path, "{}_table.json".format(name))
    with open(table_path, 'r', encoding='utf-8') as json_table:
        table = json.load(json_table)

    return index, table


def _create_positive_tokens(string_tokens, annotations) -> List[Token]:
    positive_tokens = []
    sentence_id = hash(' '.join(string_tokens))
    for ann in annotations:
        span_tokens = string_tokens[ann["start"]:ann["end"]]
        span_range = range(ann["start"], ann["end"])
        for token, index in zip(span_tokens, span_range):
            token = Token(index, ann["type"], token, sentence_id)
            positive_tokens.append(token)

    return positive_tokens


def _create_negative_tokens(string_tokens: List[str], pos_tokens: List[Token], neg_label: str) -> List[Token]:
    skip = {token.index for token in pos_tokens}
    negative_tokens = create_tokens(string_tokens, skip, neg_label)

    return negative_tokens
    

def create_tokens(string_tokens: List[str], skip={}, neg_label='O') -> List[Token]:
    tokens = []
    sentence_id = hash(' '.join(string_tokens))
    for i, string_token in enumerate(string_tokens):
        if i not in skip:
            tokens.append(Token(i, neg_label, string_token, sentence_id))

    return tokens
        

def filter_negatives(datapoints: List[Datapoint], neg_label='O'):
    return [d for d in datapoints if d.label!=neg_label]

