import torch
import torch.nn.functional as F
import json
import os
from collections import defaultdict, Counter
from embedders import BertEmbedder
import utils
from tqdm import tqdm
import math
import numpy as np
import time
import faiss
import data
from evaluate import evaluate as eval
from typing import List, Type

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VOTING_F = {'discrete':(lambda s, i: 1), 'rank_weighted':(lambda s, i: 1/(i+1)), 'similarity_weighted':(lambda s, i: s)}

class NearestNeighBERT:
    '''
    A K-nearest neighbor classifier
    Optional parameters can be passed.
    Args:
        k (int): nearest neighbors to cast a vote
        f_voting (str): name of voting function accountable for the weight of a vote
        f_similarity (str): name of function to compare vectors with e.g L2 or IP
        entity_index (faiss index): a faiss index for entities allowing for skipping of training
        entity_table (array): a table which maps faiss entity indices to properties such as label
        relation_index (faiss index): a faiss index for relations allowing for skipping of training
        relation_table (array): a table which maps faiss relation indices to properties such as label
        tokenizer (Embedder): a tokenizer to tokenize and embed text
        f_entity_embedding (str): function name to obtain a single entity embedding from multiple
        f_relation_embedding (str): function name to obtain a single relation embedding
        neg_ent (int): amount of negative entity examples to index in addition to positive examples
        neg_rel (int): amount of negative relation examples to index in addition to positive examples
        neg_label (str): label for a Datapoint that has no type
        max_span_length (int): create spans up to this length
    '''

    def __init__(self, k=10, f_voting="similarity_weighted", f_similarity="L2", entity_index=None, 
                 entity_table=[], relation_index=None, relation_table=[], tokenizer=None,
                 f_entity_embedding="abs_max", f_relation_embedding="substract", neg_ent=30, neg_rel=5,
                 neg_label='O', max_span_length=3, glue_spans=False):
        self.k = k
        self.f_voting = f_voting
        self.f_similarity = f_similarity
        self.entity_index = entity_index
        self.entity_table = entity_table
        self.relation_index = relation_index
        self.relation_table = relation_table
        self.tokenizer = tokenizer
        self.f_entity_embedding = f_entity_embedding
        self.f_relation_embedding = f_relation_embedding
        self.neg_ent = neg_ent
        self.neg_rel = neg_rel
        self.neg_label = neg_label
        self.max_span_length = max_span_length


    def configure(self, path="configs/config_za.json"):
        with open(path, 'r', encoding='utf-8') as json_config:
            config = json.load(json_config)
            self.k = config.get("k", self.k)
            self.f_voting = config.get("voting_f", self.f_voting)
            self.f_entity_embedding = config.get("f_entity_embedding", self.f_entity_embedding)
            self.f_relation_embedding = config.get("f_relation_embedding", self.f_relation_embedding)
            self.neg_ent = config.get("neg_ent", self.neg_ent)
            self.neg_rel = config.get("neg_rel", self.neg_rel)
            self.neg_label = config.get("neg_label", self.neg_label)
            self.max_span_length = config.get("max_span_length", self.max_span_length)
        
        return self

    def train(self, dataset_path, tokenizer_path='scibert-base-uncased', save_path="data/"):   
        self.tokenizer = BertEmbedder(tokenizer_path)
        self.entity_index = data.init_faiss(self.f_entity_embedding, self.f_similarity, self.tokenizer)
        self.relation_index = data.init_faiss(self.f_relation_embedding, self.f_similarity, self.tokenizer)

        data_generator = data.prepare_dataset(dataset_path, self.tokenizer, self.max_span_length, self.neg_rel, 
                                              self.neg_ent, self.neg_label, self.f_entity_embedding, 
                                              self.f_relation_embedding)
        print("Training...")
        for entities, relations in data_generator:

            # Index entities
            entity_embeddings = [e.embedding for e in entities]
            entity_entries = [e.to_table_entry() for e in entities]
            self.entity_table += entity_entries
            self.entity_index.add(torch.stack(entity_embeddings).numpy())

            # Index relations
            relation_embeddings = [r.embedding for r in relations]
            relation_entries = [r.to_table_entry() for r in relations]
            self.relation_table += relation_entries
            self.relation_index.add(torch.stack(relation_embeddings).numpy())
        
        data.save_faiss(self.entity_index, self.entity_table, "entities", save_path)
        data.save_faiss(self.relation_index, self.relation_table, "relations", save_path)

    def ready_inference(self, index_path, tokenizer_path='scibert-base-uncased', device=DEVICE):
        self.tokenizer = BertEmbedder(tokenizer_path)
        self.entity_index, self.entity_table = data.load_faiss(index_path, device, "entities")
        self.relation_index, self.relation_table = data.load_faiss(index_path, device, "relations")

    def infer(self, document):
        inference_document = []
        for sentence in tqdm(document["sentences"]):
            original_tokens, _, token2char, = self.tokenizer.split(sentence)
            bert_tokens, tok2orig, orig2tok = self.tokenizer.tokenize_with_mapping(original_tokens)
            embeddings = self.tokenizer.embed(bert_tokens)[1:-1] # skip special tokens

            # Infer entities
            entities = data.create_entities(original_tokens, self.max_span_length)
            if entities:
                entity_embeddings = [e.calculate_embedding(embeddings, bert_tokens, orig2tok, self.f_entity_embedding) for e in entities]
                q_entities = torch.stack(entity_embeddings).numpy()
                D, I = self.entity_index.search(q_entities, self.k)
                self.vote_labels(entities, self.entity_table, D, I)
            pos_entities = data.filter_negatives(entities, self.neg_label)

            # Infer relations
            relations = data.create_relations(pos_entities, neg_label=self.neg_label)
            if relations:
                relation_embeddings = [r.calculate_embedding(self.f_relation_embedding) for r in relations]
                q_relations = torch.stack(relation_embeddings).numpy()
                D, I = self.relation_index.search(q_relations, self.k)
                self.vote_labels(relations, self.relation_table, D, I)
            pos_relations = data.filter_negatives(relations, self.neg_label)

            prediction = self.convert_prediction(original_tokens, pos_entities, pos_relations)
            inference_document.append(prediction)

        return inference_document

    def infer_(self, document):
        """
        This function alternative assumes document objects also contain embeddings,
        which would significantly speed up the inference process
        """
        # TODO: implement
        pass

    def evaluate(self, evaluation_path, inference_path="data/predictions.json"):
        """
        Evaluate a dataset by doing inference on the data without labels. 
        See data.py for example format of the dataset.
        """
        dataset = json.load(open(evaluation_path))
        sentences = [' '.join(entry["tokens"]) for entry in dataset]
        print("Evaluating...")
        inference_document = self.infer({"sentences": sentences})
        with open(inference_path, 'w', encoding='utf-8') as f:
            json.dump(inference_document, f)
        eval(evaluation_path, inference_path)


    def vote_labels(self, datapoints: List[data.Datapoint], table, distances, indices):
        """
        Given a list of datapoints and their distances and indices,
        assign a label to the datapoints 
        """
        # print(distances)
        a = distances-np.min(distances)
        b = np.max(distances)-np.min(distances)
        normalized_distances = 1-np.divide(a, b, out=np.zeros_like(a), where=b!=0) # 1-similarity for distance
        pred_labels = []
        for i, row in enumerate(indices):
            weight_counter = Counter()
            votes = []
            neighbor_tokens = []
            candidate = datapoints[i] 
            for j, neighbor_index in enumerate(row):
                neighbor = table[neighbor_index]
                vote = neighbor["label"]
                # print(neighbor["string"])
                votes.append(vote)
                weight = VOTING_F[self.f_voting](normalized_distances[i][j], j)
                weight_counter[vote] += weight
            

            pred_label = weight_counter.most_common(1)[0][0]
            candidate.label = pred_label
            # print("---->", candidate)
            # print()

    # def which_spans_ending(self, prev, current):
    #     '''Returns which labels of the prev vector were the last in their span'''
    #     spans_ending = {label:True for label in prev}
    #     for prev_label in prev:
    #         if prev_label in current:
    #             spans_ending[prev_label] = False
                
    #     return spans_ending


    def convert_prediction(self, tokens, entities, relations):
        converted_entities = [{"start":e.start, "end":e.end, "type":e.label} for e in entities]
        converted_relations = [{"head":r.head_position, "tail":r.tail_position, "type":r.label} for r in relations]

        prediction = {
                        "tokens": tokens, 
                        "entities": converted_entities, 
                        "relations": converted_relations, 
                        "orig_id": hash(' '.join(tokens))
                    }

        return prediction     

    # def convert_prediction(self, tokens, entities, relations):
    #     def _glue_spans():
    #         converted_entities, converted_relations = [], []

    #         # Map entity span labels to token labels
    #         mapped_labels = [set() for _ in tokens]
    #         for entity in entities:     
    #             if entity.label == self.neg_label:
    #                 continue
    #             for label_set in mapped_labels[entity.start:entity.end]:
    #                 label_set.add(entity.label)
    #         # label_set.add(self.neg_label)
                    
    #         # Glue individual labels 
    #         label_start = {}
    #         prev = set()   
    #         entity_labels = set([entity.label for entity in entities])
    #         entity_labels.remove(self.neg_label)
    #         for i, current in enumerate(mapped_labels):   
    #             spans_ending = self.which_spans_ending(prev, current)
    #             # print("PREV / CURRENT / ENDING", i, prev, current, spans_ending)
    #             for label in entity_labels:
    #                 # span starts
    #                 if label in current and label not in prev:
    #                     label_start[label] = i
    #                 # span ends
    #                 elif spans_ending.get(label):
    #                     e = {"start": label_start.get(label, 0), "end": i, "type":label}
    #                     converted_entities.append(e)
    #                 # last in sequence
    #                 if i==(len(mapped_labels)-1) and label in current:
    #                     e = {"start": label_start.get(label, 0), "end": i+1, "type":label}
    #                     converted_entities.append(e)
    #             prev = current
            
    #         print("relations to map:", len([r for r in relations if r.label!=self.neg_label]))
    #         # Remap relations to new entity indices
    #         entity_map = {i:None for i, _ in enumerate(tokens)}
    #         for i, e in enumerate(converted_entities):
    #             for j in range(e["start"], e["end"]):
    #                 entity_map[j] = i
    #         converted_relations = []
    #         duplicates = set()
    #         for r in relations:
    #             if r.label == self.neg_label:
    #                 continue
    #             mapped_relation = {"head":entity_map[r.head.start], "tail":entity_map[r.tail.start], "type":r.label}
    #             tuple_= (entity_map[r.head.start],entity_map[r.tail.start],r.label)
    #             if tuple_ not in duplicates:
    #                 converted_relations.append(mapped_relation)
    #                 duplicates.add(tuple_)

    #         print("mapped relations", len(converted_relations))
    #         # for token, label in zip(tokens, mapped_labels):
    #         #     if label:
    #         #         print(token, label)            
    #         for entity in converted_entities:
    #             print(entity, tokens[entity["start"]:entity["end"]])
    #         print(converted_entities)
    #         print(converted_relations)
    #         return converted_entities, converted_relations

    #     if self.glue_spans:
    #         converted_entities, converted_relations = _glue_spans()
    #     else:
    #         converted_entities = [{"start":e.start, "end":e.end, "type":e.label} for e in entities if(
    #                     e.label != self.neg_label)]
    #         converted_relations = [{"head":r.head_position, "tail":r.tail_position, "type":r.label} for r in relations if(
    #                     r.label != self.neg_label)]

    #     prediction = {
    #                     "tokens": tokens, 
    #                     "entities": converted_entities, 
    #                     "relations": converted_relations, 
    #                     "orig_id": hash(' '.join(tokens))
    #                  }

    #     return prediction
    
    def __repr__(self):
        return "NearestNeighBERT()"


    def __str__(self):
        return "NearestNeighBERT"    

if __name__ == "__main__":
    TRAIN_PATH = "../spert/data/datasets/semeval2017_task10/semeval2017_task10_train.json"
    EVAL_PATH = "../spert/data/datasets/semeval2017_task10/semeval2017_task10_dev.json"
    TOKENIZER_PATH = "scibert_scivocab_uncased/"
    CONFIG_PATH = "configs/semeval.json"
    SAVE_PATH = "data/"
    nn = NearestNeighBERT().configure(CONFIG_PATH)
    # nn.train(TRAIN_PATH, TOKENIZER_PATH, SAVE_PATH)
    nn.ready_inference(SAVE_PATH, TOKENIZER_PATH)
    nn.evaluate(EVAL_PATH)