import json
import faiss
import utils
import random
import torch
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

class Entity(Datapoint):
    def __init__(self, start: int, end: int, label='O', original_tokens=[], 
                 sentence_id='', embedding=None, embedding_tokens=[]):
        super().__init__(embedding, label)
        self.start = start
        self.end = end
        self.original_tokens = original_tokens
        self.embedding_tokens = embedding_tokens
        self.sentence_id = sentence_id
        self.id = hash('_'.join(original_tokens)+str(start)+str(end))

        

    def to_table_entry(self):
        entry = {
            "start": self.start,
            "end": self.end,
            "label": self.label, 
            "string": str(self), 
            "original_tokens": self.original_tokens,
            "embedding_tokens": self.embedding_tokens,
            "sentence_id": self.sentence_id,
            "id": self.id
        }

        return entry


    def calculate_embedding(self, embeddings, bert_tokens, orig2tok, accumulation_f="abs_max"):
        """
        Here it can be decided which embedding is selected
        to represent all (sub)tokens in the span. Several strategies can be applied.
        E.g. the first (sub)token, an average of (sub)tokens, or a pool.
        """

        def _first(embeddings, bert_tokens, orig2tok):
            first, _ = orig2tok[self.start]
            embedding = embeddings[first]
            tokens = [bert_tokens[first]]

            return embedding, tokens
        
        def _abs_max(embeddings, bert_tokens, orig2tok):
            positions = orig2tok[self.start:self.end]
            first, last = positions[0][0], positions[-1][1]
            selected_embeddings = [embedding for embedding in embeddings[first:last]]
            t = torch.stack(selected_embeddings)
            abs_max_indices = torch.abs(t).argmax(dim=0)
            embedding = t.gather(0, abs_max_indices.view(1,-1)).squeeze()
            tokens = bert_tokens[first:last]

            return embedding, tokens

        def _mean(embeddings, bert_tokens, orig2tok):
            positions = orig2tok[self.start:self.end]
            first, last = positions[0][0], positions[-1][1]
            selected_embeddings = [embedding for embedding in embeddings[first:last]]
            embedding = torch.stack(selected_embeddings).mean(dim=0)
            tokens = bert_tokens[first:last]

            return embedding, tokens

        f_reduce = {"first": _first, "abs_max": _abs_max, "mean": _mean}.get(accumulation_f)
        self.embedding, self.embedding_tokens = f_reduce(embeddings, bert_tokens, orig2tok)

        return self.embedding

    def __repr__(self):
        return "Entity()"

    def __str__(self):
        return "[ENT] {} >> {}".format('_'.join(self.original_tokens), self.label)

class Relation(Datapoint):
    def __init__(self, head: Entity, tail: Entity, head_position, tail_position, label='O', sentence_id='', embedding=None):  
        super().__init__(embedding, label)
        self.head = head
        self.tail = tail
        self.head_position = head_position
        self.tail_position = tail_position
        self.string = str(self)
        self.sentence_id = sentence_id
        self.id = head.id-tail.id


    def to_table_entry(self):
        entry = {
            "head_position": self.head_position,
            "tail_position": self.tail_position,
            "string": str(self),
            "label": self.label,
            "sentence_id": self.sentence_id,
            "id": self.id
        }

        return entry

    def calculate_embedding(self, accumulation_f="substract"):
        """
        Here it can be decided which embedding is selected
        to represent all (sub)tokens in the span. Several strategies can be applied.
        E.g. the first (sub)token, an average of (sub)tokens, or a pool.
        """

        def _concat(head: Entity, tail: Entity):
            embedding = torch.cat([head.embedding, tail.embedding])

            return embedding
        
        def _substract(head: Entity, tail:Entity):
            embedding = head.embedding-tail.embedding

            return embedding

        f_reduce = {"concat": _concat, "substract": _substract} 
        self.embedding = f_reduce[accumulation_f](self.head, self.tail)

        return self.embedding

    def __repr__(self):
        return "Relation()"

    def __str__(self):
        s_head = '_'.join(self.head.original_tokens)
        s_tail = '_'.join(self.tail.original_tokens)
        return "[REL] {} >> {} >> {}".format(s_head, self.label, s_tail)        

def prepare_dataset(path, tokenizer, max_span_length=1, neg_rel=20, neg_ent=30, neg_label='O',
                    f_entity_embedding="abs_max", f_relation_embedding="substract"):
    dataset = json.load(open(path))
    for annotation in tqdm(dataset):
        original_tokens, _, _ = tokenizer.split(' '.join(annotation["tokens"]))
        bert_tokens, tok2orig, orig2tok = tokenizer.tokenize_with_mapping(original_tokens)
        embeddings = tokenizer.embed(bert_tokens)[1:-1] # skip special tokens  
        pos_entities, neg_entities = _create_labeled_entities(original_tokens, annotation["entities"], 
                                                              max_span_length, neg_ent, neg_label)
        entities = pos_entities+neg_entities
        for entity in entities: entity.calculate_embedding(embeddings, bert_tokens, orig2tok, f_entity_embedding)
        relations = _create_labeled_relations(pos_entities, annotation["relations"], neg_rel)
        for relation in relations: relation.calculate_embedding(f_relation_embedding)

        yield entities, relations

def init_faiss(f_embedding, f_similarity, tokenizer):
    def _l2(size):
        return faiss.IndexFlatL2(size)
    def _ip(size):
        return faiss.IndexFlatIP(size)

    embedding_size = {
        "concat": tokenizer.embedding_size*2,
        "substract": tokenizer.embedding_size,
        "abs_max": tokenizer.embedding_size,
        "mean": tokenizer.embedding_size 
    }.get(f_embedding, tokenizer.embedding_size)

    index = {
        "L2": _l2,
        "IP": _ip
    }.get(f_similarity, _l2)

    return index(embedding_size)


def save_faiss(index, table, name, save_path="data/save/"):
    print("Saving {} index...".format(name))
    utils.create_dir(save_path)
    faiss.write_index(index, save_path+"{}_index".format(name))
    with open(save_path+"{}_table.json".format(name), 'w') as json_file:
        json.dump(table, json_file)
    print("Indexed {} {} with their labels".format(len(table), name))


def load_faiss(path, device, name):
    index = faiss.read_index(path+"{}_index".format(name))
    if device == "cuda":
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)
         
    with open(path+"{}_table.json".format(name), 'r', encoding='utf-8') as json_table:
        table = json.load(json_table)

    return index, table


def _create_labeled_entities(tokens, annotations, max_span_length, neg_examples, neg_label):
    pos_entities, neg_entities = [], []
    sentence_id = hash(' '.join(tokens))
    
    # Positive entities
    for ann in annotations:
        original_tokens = tokens[ann["start"]:ann["end"]]
        entity = Entity(ann["start"], ann["end"], ann["type"], original_tokens, sentence_id)
        pos_entities.append(entity)

    # Negative entities
    skip = {e.id for e in pos_entities}
    # negative_entities = create_entities(tokens, max_span_length, skip)
    negative_entities = []
    for sequence in remaining_sequences(tokens, pos_entities):
        negative_entities += create_entities(sequence, max_span_length, skip)
    n_neg = max(min(neg_examples, len(negative_entities)), min(len(tokens), len(negative_entities)))
    for _ in range(n_neg):
        selected = random.randint(0, len(negative_entities)-1)
        entity = negative_entities.pop(selected)
        neg_entities.append(entity)

    return pos_entities, neg_entities


def create_entities(tokens, max_length, skip={}, neg_label='O'):
    entities = []
    sentence_id = hash(' '.join(tokens))
    for l in range(1, max_length+1):
        for i in range(len(tokens)-l+1):
            original_tokens = tokens[i:i+l]
            entity = Entity(i, i+l, neg_label, original_tokens, sentence_id)
            if entity.id not in skip:
                entities.append(entity)

    return entities

def remaining_sequences(tokens, entities):
    entity_tokens = [None for _ in tokens]
    for e in entities:
        entity_tokens[e.start:e.end] = e.label
    remaining_sequence = []
    for token, entity in zip(tokens, entity_tokens):
        if not entity:
            remaining_sequence.append(token)
        elif entity and remaining_sequence:
            yield remaining_sequence
            remaining_sequence = []

def filter_negatives(datapoints: List[Datapoint], neg_label='O'):
    return [d for d in datapoints if d.label!=neg_label]
    
# def filter_entities(entities, tokens, glue_spans=False, neg_label='O'):
#     def which_spans_ending(prev, current):
#         '''Returns which labels of the prev vector were the last in their span'''
#         spans_ending = {label:True for label in prev}
#         for prev_label in prev:
#             if prev_label in current:
#                 spans_ending[prev_label] = False
#         return spans_ending

#     def _glue_spans():
#         filtered_entities =[]

#         # Map entity span labels to token labels
#         mapped_labels = [set() for _ in tokens]
#         for entity in entities:     
#             if entity.label == neg_label:
#                 continue
#             for label_set in mapped_labels[entity.start:entity.end]:
#                 label_set.add(entity.label)
                
#         # Glue individual labels 
#         label_start = {}
#         prev = set()   
#         entity_labels = set([entity.label for entity in entities])
#         entity_labels.remove(neg_label)
#         for i, current in enumerate(mapped_labels):   
#             spans_ending = which_spans_ending(prev, current)
#             # print("PREV / CURRENT / ENDING", i, prev, current, spans_ending)
#             for label in entity_labels:
#                 # span starts
#                 if label in current and label not in prev:
#                     label_start[label] = i
#                 # span ends
#                 elif spans_ending.get(label):
#                     start = label_start.get(label, 0)
#                     e = Entity(start, i, label, tokens[start:i])
#                     filtered_entities.append(e)
#                 # last in sequence
#                 if i==(len(mapped_labels)-1) and label in current:
#                     start = label_start.get(label, 0)
#                     e = Entity(start, i+1, label, tokens[start:i])
#                     filtered_entities.append(e)
#             prev = current
        
#         return filtered_entities

#     if glue_spans:
#         filtered_entities = _glue_spans()
#     else:
#         filtered_entities = [e for e in entities if e.label!=neg_label]

#     return filtered_entities

def _create_labeled_relations(entities, annotations, neg_examples):
    relations = []

    # Positive relations
    for ann in annotations:
        head = entities[ann["head"]]
        tail = entities[ann["tail"]]
        relation = Relation(head, tail, ann["head"], ann["tail"], ann["type"], head.id)
        relations.append(relation)

    # Negative relations
    skip = {r.id for r in relations}
    negative_relations = create_relations(entities, skip)
    for _ in range(min(neg_examples, len(negative_relations))):
        selected = random.randint(0, len(negative_relations)-1)
        relation = negative_relations.pop(selected)
        relations.append(relation)

    return relations


def create_relations(entities, skip={}, neg_label='O', max_rel=1000):
    relations = []
    for i, head in enumerate(entities):
        for j, tail in enumerate(entities):
            relation = Relation(head, tail, i, j, neg_label, head.sentence_id)
            if relation.id not in skip and head != tail:
                relations.append(relation)
            if len(relations) > max_rel:
                # print("[WARNING] Maximum number of relations per sequence reached!")
                return relations

    return relations
        



