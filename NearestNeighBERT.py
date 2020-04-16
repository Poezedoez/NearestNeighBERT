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
# import faiss
from data import read_dataset, read_faiss

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VOTING_F = {'discrete':(lambda s, i: 1), 'rank_weighted':(lambda s, i: 1/(i+1)), 'similarity_weighted':(lambda s, i: s)}

class NearestNeighBERT:
    '''
    A K-nearest neighbor classifier
    Optional parameters can be passed.
    Args:
        k (int): nearest neighbors to cast a vote
        voting_f (str): name of voting function accountable for the weight of a vote
        index (faiss index): a faiss index allowing for skipping of training
        index_table (array): a table which maps faiss indices to properties such as label
        tokenizer (Embedder): a tokenizer to tokenize and embed text
    '''

    def __init__(self, k=10, voting_f='similarity_weighted', entity_index=None, 
                 entity_table=[], relation_index=None, relation_table=[], tokenizer=None):
        self.k = k
        self.voting_f = voting_f
        self.entity_index = entity_index
        self.entity_table = entity_table
        self.relation_index = relation_index
        self.relation_table = relation_table
        self.tokenizer = tokenizer


    def configure(self, path="configs/config_za.json"):
        with open(path, 'r', encoding='utf-8') as json_config:
            configuration = json.load(json_config)
            self.k = configuration.get("k", self.k)
            self.voting_f = configuration.get("voting_f", self.voting_f)
        
        return self


    def train(self, dataset_path, tokenizer_path='scibert-base-uncased', 
              save_path="data/", index_name="faiss_index", index_table_name="faiss_index_table.json",
              max_span_length=None):   
        self.tokenizer = BertEmbedder(tokenizer_path)
        # self.index = faiss.IndexFlatL2(self.tokenizer.embedding_size)
        for spans, relations in read_dataset(dataset_path, self.tokenizer, max_span_length):
            # self.index.add(torch.stack(embeddings).numpy())
            entity_entries = [span.to_entry_table for span in spans]
            relation_entries = [relation.to_entry_table for relation in relations]

            # check entities
            for entry in entity_entries:
                print(entry)
            print()

            # check relations
            for entry in relation_entries:
                print(entry)
            print()

            self.entity_table += entity_entries
            self.relation_table += relation_entries
        
        # Save train results
        utils.create_dir(save_path)
        # faiss.write_index(self.index, save_path+index_name)
        with open(save_path+index_table_name, 'w') as json_file:
            json.dump(self.index_table, json_file)
        print("Indexed {} tokens with their labels".format(len(self.index_table)))

    def ready_inference(self, index_path, tokenizer_path='scibert-base-uncased', device=DEVICE,
                        index_name="faiss_index", index_table_name="faiss_index_table.json"):
        self.tokenizer = BertEmbedder(tokenizer_path)
        self.index, self.index_table = read_faiss(index_path, device, index_name, index_table_name)

    def infer(self, document, span_based=False):
        # Classify each vector based on k-NN voting function
        inference_document = []
        for sentence in document["sentences"]:
            tokens, _, token2char, = self.tokenizer.split(sentence)
            spans = create_spans(tokens)
            for span in spans: span.calculate_embedding(tokens, tokenizer)
            D, I = self.index.search(torch.stack(selected_embeddings).numpy(), self.k)
            pred_labels = self.vote_labels(I, D)
            prediction = self.convert_prediction(tokens, pred_labels)
            inference_document.append(prediction)

        return inference_document

    def infer_(self, document):
        """
        This function alternative assumes document objects also contain embeddings,
        which would significantly speed up the inference process
        """
        # TODO: implement
        pass


    def vote_labels(self, nearest_neighberts, similarities):
        """
        Given the nearest neighberts of a query, along with their
        similarity, vote the predicted labelset
        """
        # print(similarities)
        a = similarities-np.min(similarities)
        b = np.max(similarities)-np.min(similarities)
        normalized_similarities = 1-np.divide(a, b, out=np.zeros_like(a), where=b!=0) # 1-similarity for distance
        pred_labels = []
        for i, row in enumerate(nearest_neighberts):
            weight_counter = Counter()
            votes = []
            neighbor_tokens = []
            for j, neighbor_index in enumerate(row):
                neighbor = self.index_table[neighbor_index]
                neighbor_tokens.append(neighbor["token"])
                vote = tuple(neighbor["label"])
                votes.append(vote)
                weight = VOTING_F[self.voting_f](normalized_similarities[i][j], j)
                weight_counter[vote] += weight

            pred_labelset = list(weight_counter.most_common(1)[0][0])
            pred_labels.append(pred_labelset)
        
        return pred_labels

    def which_spans_ending(self, prev, current, ):
        '''Returns which labels of the prev vector were the last in their span'''
        spans_ending = {label:True for label in prev}
        for prev_label in prev:
            if prev_label in current:
                spans_ending[prev_label] = False
                
        return spans_ending

    def convert_prediction(self, tokens, labels, no_entity='O'):
        entities = []
        label_start = {}
        prev = set()
        sequence_labels = set().union(*labels)
        for i, current in enumerate(labels):
            spans_ending = self.which_spans_ending(prev, current)
            for label in sequence_labels:
                # only convert entities
                if label == no_entity:
                    continue
                # span ends
                if spans_ending.get(label):
                    e = {"start": label_start.get(label, 0), "end": i, "type":label}
                    entities.append(e)
                # span starts
                elif label in current:
                    label_start[label] = i
                # last in sequence
                if i==(len(labels)-1):
                    e = {"start": label_start.get(label, 0), "end": i+1, "type":label}
                    entities.append(e)

        prediction = {
                        "tokens": tokens, 
                        "entities":entities, 
                        "relations": [], 
                        "orig_id": hash(' '.join(tokens))
                     }

        return prediction
    
    def __repr__(self):
        return "NearestNeighBERT()"


    def __str__(self):
        return "NearestNeighBERT"    

if __name__ == "__main__":
    TEST_DATA_PATH = "../spert/data/datasets/semeval2017_task10/semeval2017_task10_train.json"
    TOKENIZER_PATH = "../spert/scibert_scivocab_uncased/"
    nn = NearestNeighBERT()
    nn.train(TEST_DATA_PATH, TOKENIZER_PATH, max_span_length=7)