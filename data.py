import json
# import faiss
import utils
import random
from typing import List, Dict

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

class Span:
    def __init__(self, start: int, end: int, label='O', token='', embedding=None, sentence_id=''):
        self.start = start
        self.end = end
        self.label = label
        self.token = token
        self.embedding = embedding
        self.sentence_id = sentence_id


    def to_table_entry(self):
        entry = {
            "label":self.label, 
            "token":self.token, 
            "sentence_id":self.sentence_id
        }
        
        return entry


    def calculate_embedding(self, original_tokens, tokenizer, accumulation_f="mean"):
        """
        Here it can be decided which embedding is selected
        to represent all (sub)tokens in the span. Several strategies can be applied.
        E.g. the first (sub)token, an average of (sub)tokens, or a pool.
        """

        def _first(embeddings, tok2orig, orig2tok, bert_tokens):
            first, _ = orig2tok[span[0]]
            embedding = embeddings[first]
            token = bert_tokens[first]

            return embedding, token
        
        def _abs_max(embeddings, tok2orig, orig2tok, bert_tokens):
            pass

        def _mean(span, embeddings, tok2orig, orig2tok, bert_tokens):
            positions = orig2tok[span[0]:span[1]]
            first, last = positions[0][0], positions[-1][1]
            selected_embeddings = [embedding for embedding in embeddings[first:last]]
            embedding = torch.stack(selected_embeddings).mean(dim=0)
            token = '_'.join(bert_tokens[first:last])

            return embedding, token

        f_reduce = {"first": _first, "abs_max": _abs_max, "mean": _mean}

        self.embedding, self.token = f_reduce[accumulation_f](embeddings, tok2orig, orig2tok, bert_tokens)


class Relation:
    def __init__(self, head: Span, tail: Span, label='O', embedding=None, sentence_id=''):  
        self.head = head
        self.tail = tail
        self.label = label
        self.embedding = embedding
        self.sentence_id = sentence_id


    def to_table_entry(self):
        entry = {
            "label":self.label,
            "head":self.head.token,
            "tail":self.tail.token
        }

        return entry

    def calculate_embedding(self, accumulation_f="substract"):
        """
        Here it can be decided which embedding is selected
        to represent all (sub)tokens in the span. Several strategies can be applied.
        E.g. the first (sub)token, an average of (sub)tokens, or a pool.
        """

        def _concat(head: Span, tail: Span):
            embedding = torch.cat([head.embedding, tail.embedding])

            return embedding
        
        def _substract(head: Span, tail:Span):
            embedding = head.embedding-tail.embedding

            return embedding

        f_reduce = {"concat": _concat, "substract": _substract} 
        self.embedding = f_reduce[accumulation_f](self.head, self.tail)



def read_dataset(path, tokenizer, max_span_length, neg_rel=10):
    dataset = json.load(open(path))
    for annotation in dataset:
        original_tokens, _, _ = tokenizer.split(' '.join(annotation["tokens"]))
        bert_tokens, tok2orig, orig2tok = tokenizer.tokenize_with_mapping(original_tokens)
        embeddings = tokenizer.embed(bert_tokens)[1:-1] # skip special tokens  
        l = max_span_length if max_span_length else 1
        pos_spans, neg_spans = create_labeled_spans(original_tokens, annotation["entities"], l)
        for span in spans: span.calculate_embedding()
        tokenizer.add_embedding(original_tokens, neg_spans)
        relations = create_labeled_relations(pos_spans, annotation["relations"], neg_rel)
        add_embedding()

        yield pos_spans+neg_spans, relations

def read_faiss(path, device, index_name, index_table_name):
    index = faiss.read_index(path+index_name)
    if device == "cuda":
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)
         
    with open(path+index_table_name, 'r', encoding='utf-8') as json_table:
        index_table = json.load(json_table)

    return index, index_table

# def _fetch_labels(tokens, entities, no_entity='O'):
#     labels = [set() for token in tokens]
#     for entity in entities:
#         for labelset in labels[entity["start"]:entity["end"]]:
#             labelset.add(entity["type"])
#     for labelset in labels:
#         if not labelset:
#             labelset.add(no_entity) 

#     return labels


def create_labeled_spans(sequence, annotations, max_length, n=None, neg_label='O'):
    labels = []
    spans = []
    
    # Add positive examples
    for entity in entities:
        spans.append((entity["start"], entity["end"]))
        labels.append(set([entity["type"]]))

    # Add negative examples
    skip = set([(entity["start"], entity["end"]) for entity in entities])
    negative_spans = create_spans(sequence, max_length, skip)
    negative_examples = n if n else len(sequence)
    for _ in range(negative_examples):
        selected = random.randint(0, len(negative_spans)-1)
        span = negative_examples.pop(selected)
        spans.append(span)
        labels.append(set([neg_label]))

    return spans, labels

def create_labeled_relations(annotations, neg_rel=10):
    embeddings = []
    labels = []

    return embeddings, labels

def create_relations()

def create_spans(sequence, max_length, skip=set()):
    spans = []
    for l in range(1, max_length+1):
        for i in range(len(sequence)-l+1):
            span = (i, i+l)
            if span not in skip:
                spans.append(span)

    return spans


