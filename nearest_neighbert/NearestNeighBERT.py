import sys
import torch
import torch.nn.functional as F
import json
import os
from collections import defaultdict, Counter
from tqdm import tqdm
import math
import numpy as np
import time
import faiss
import copy
from typing import List, Type
import random

# Local
import nearest_neighbert.data as nn_data
from nearest_neighbert.embedders import BertEmbedder
from nearest_neighbert.evaluate import compare_datasets
from nearest_neighbert.evaluate import evaluate as eval
import nearest_neighbert.utils as nn_utils




DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VOTING_F = {'discrete':(lambda s, i: 1), 'rank_weighted':(lambda s, i: 1/(i+1)), 'similarity_weighted':(lambda s, i: s)}

class NearestNeighBERT:
    '''
    A K-nearest neighbor classifier.
    Optional parameters can be passed.
    Args:
        k (int): nearest neighbors to cast a vote
        f_voting (str): name of voting function accountable for the weight of a vote
        f_similarity (str): name of function to compare vectors with e.g L2 or IP
        index (faiss index): a faiss index for entities allowing for skipping of training
        table (array): a table which maps faiss token indices to properties such as label
        tokenizer (Embedder): a tokenizer to tokenize and embed text
        f_reduce (str): function name to obtain combine subword tokens
        neg_label (str): label for a Datapoint that has no type
    '''

    def __init__(self, k=10, f_voting="similarity_weighted", f_similarity="L2", index=None, 
                 table=None, tokenizer=None, f_reduce="mean", neg_label="O", indicator="knn_module",
                 device=DEVICE, positive_multiplier=1):
        self.k = k
        self.f_voting = f_voting
        self.f_similarity = f_similarity
        self.index = index
        self.table = table = [] if not table else table
        self.tokenizer = tokenizer
        self.f_reduce = f_reduce
        self.neg_label = neg_label
        self.indicator = indicator
        self.config = None
        self.device = device
        self.index_count = 0
        self.positive_multiplier = positive_multiplier


    def configure(self, path="configs/config_za.json"):
        with open(path, 'r', encoding='utf-8') as json_config:
            config = json.load(json_config)
            self.k = config.get("k", self.k)
            self.f_voting = config.get("f_voting", self.f_voting)
            self.f_similarity = config.get("f_similarity", self.f_similarity)
            self.f_reduce = config.get("f_reduce", self.f_reduce)
            self.neg_label = config.get("neg_label", self.neg_label)
            self.positive_multiplier = config.get("positive_multiplier", self.positive_multiplier)
        self.config = config

        return self


    def ready_training(self, tokenizer_path, size=None):
        self.tokenizer = BertEmbedder(tokenizer_path)
        embedding_size = size if size else self.tokenizer.embedding_size
        self.index = nn_data.init_faiss(self.f_reduce, self.f_similarity, embedding_size)


    def train(self, dataset_path, tokenizer_path='scibert-base-cased', save_path="data/", save=True):   
        """
        Add whole dataset (not yet embedded) to memory 
        """
        self.ready_training(tokenizer_path)
        data_generator = nn_data.prepare_dataset(dataset_path, self.tokenizer, self.neg_label, self.f_reduce)
        print("Training...")
        for tokens in data_generator:
            if tokens:
                token_embeddings = [t.embedding for t in tokens]
                token_embeddings = torch.stack(token_embeddings).numpy()
                token_entries = [t.to_table_entry() for t in tokens]
                self.train_(token_embeddings, token_entries)
                self.index_count += len(token_embeddings)

        if save:
            nn_data.save_faiss(self.index, self.table, "tokens", save_path)
            train_config_path = os.path.join(save_path, "train_config.json")
            self.save_config(train_config_path)
    

    def train_(self, embeddings, entries):
        """
        Add embeddings and corresponding entries to memory.
        """  
        self.table += entries
        a = np.array(embeddings, dtype=np.float32)
        self.index.add(a)
        self.index_count += len(np.array(embeddings))


    def ready_inference(self, index_path, tokenizer_path='scibert-base-uncased'):
        self.tokenizer = BertEmbedder(tokenizer_path)
        self.index, self.table = nn_data.load_faiss(index_path, self.device, "tokens")


    def infer(self, document: List[List[str]], verbosity):
        """
        Do inference on whole document/dataset (not yet embedded)
        """
        print("Using {} as similarity index".format(self.f_similarity))
        inference_document = []
        for sentence in tqdm(document["sentences"]):
            bert_tokens, tok2orig, orig2tok = self.tokenizer.tokenize_with_mapping(sentence)
            embeddings = self.tokenizer.embed(bert_tokens)[1:-1] # skip special tokens
            tokens = nn_data.create_tokens(sentence)
            if tokens:
                token_embeddings = [t.calculate_embedding(embeddings, bert_tokens, orig2tok, self.f_reduce) for t in tokens]
                q = torch.stack(token_embeddings).numpy()
                labels, neighbors = self.infer_(q)
                self.assign_labels(tokens, labels, neighbors, verbosity)

            prediction = self.convert_prediction(sentence, tokens)
            inference_document.append(prediction)

        return inference_document


    def infer_(self, embeddings, label_type=str, label_key="label"):
        """
        Do one inference call on a query.
        @embeddings are type array_like.
        """
        q = np.array(embeddings, dtype=np.float32)
        D, I = self.index.search(q, self.k)
        a = D-np.min(D, axis=1)[:, np.newaxis]
        b = np.max(D, axis=1)[:, np.newaxis]-np.min(D, axis=1)[:, np.newaxis]
        D_norm = np.divide(a, b, out=np.zeros_like(a), where=b!=0)
        if self.f_similarity=='L2':
            D_norm = 1-D_norm # 1-distance for similarity
        labels, neighbors = self.vote(D_norm, I, label_type, label_key)

        return labels, neighbors


    def evaluate(self, evaluation_path, results_path="data/results/", verbosity=0):
        """
        Evaluate a dataset obtained by doing inference on the data without labels. 
        See data.py for example format of the dataset.
        """
        dataset = json.load(open(evaluation_path))
        sentences = [entry["tokens"] for entry in dataset]
        nn_utils.create_dir(results_path)

        print("Evaluating...")
        predictions = self.infer({"sentences": sentences[:666]}, verbosity=verbosity)
        predictions_path = os.path.join(results_path, "predictions.json")
        with open(predictions_path, 'w', encoding='utf-8') as f:
            json.dump(predictions, f)
        span_eval, token_eval = eval(evaluation_path, predictions_path, self.tokenizer, print_results=verbosity)
        if not verbosity:
            print("P/R/F1 (span):", span_eval)
            print("P/R/F1 (token):", token_eval)

        predicted_examples_path = os.path.join(results_path, "predicted_examples.txt")
        # compare_datasets(evaluation_path, predictions_path, predicted_examples_path)

        final_config_path = os.path.join(results_path, "final_config.json")
        self.save_config(final_config_path)

        return span_eval, token_eval


    def vote(self, similarities, indices, label_type=str, label_key="label"):
        """
        Given an array of similairties and neighbor indices,
        vote labels for each entry
        """
        pred_labels = []
        all_neighbors = []

        for i, row in enumerate(indices):
            weight_counter = Counter()
            nearest_neighbors = []
            for j, neighbor_index in enumerate(row):
                neighbor = self.table[neighbor_index]
                nearest_neighbors.append(neighbor)
                vote = neighbor[label_key]
                weight = VOTING_F[self.f_voting](similarities[i][j], j)
                if vote != self.neg_label:
                    weight = weight * self.positive_multiplier
                weight_counter[vote] += weight
            # print(">>> {} <<<".format(nearest_neighbors[0]["token"]))
            # for l, c in weight_counter.items():
            #     print(l, c)
            # print()
            pred_label = weight_counter.most_common(1)[0][0]
            pred_labels.append(label_type(pred_label))
            all_neighbors.append(nearest_neighbors)

        return pred_labels, all_neighbors


    def assign_labels(self, tokens, labels, neighbors, verbosity=0):
        for t, l, n in zip(tokens, labels, neighbors):
            t.label = l
            if random.uniform(0, 1) < verbosity:
                print("Predicted: ", t)
                for i in range(min(len(n), 5)):
                    neighbor = n[i]
                    print("\t (nn {}) {}".format(i, neighbor["string"]))
                print()


    def _expand_entities(self, string_tokens, tokens):
        '''
        Greedy span selection heuristic
        '''
        def _is_entity(label): 
            return label!=self.neg_label
        def _span_continues(prev, current):
            return  prev_label==current_label and _is_entity(prev_label)
        def _span_ends(prev, current):
            return prev_label!=current_label and _is_entity(prev_label)

        entities = []
        labels = [t.label for t in tokens] # assume same position as tokens
        prev_label = 'O'
        start = 0
        for i, current_label in enumerate(labels):
            if _span_continues(prev_label, current_label):
                prev_label = current_label
                continue

            if _span_ends(prev_label, current_label):
                entities.append({"start": start, "end": i, "type": prev_label})       
            
            start = i
            prev_label = current_label

        # last token of the sentence is entity
        if _is_entity(current_label):
            entities.append({"start": start, "end": i+1, "type": prev_label})

        return entities


    def convert_prediction(self, string_tokens, tokens):
        entities = self._expand_entities(string_tokens, tokens)
        prediction = {
                        "tokens": string_tokens, 
                        "entities": entities, 
                        "relations": [], 
                        "orig_id": hash(' '.join(string_tokens))
                    }
        
        return prediction  

    def save_config(self, save_path):
        if self.config:
            with open(save_path, 'w') as f:
                json.dump(self.config, f)
    
    def __repr__(self):
        return "NearestNeighBERT()"


    def __str__(self):
        return "NearestNeighBERT"    

if __name__ == "__main__":
    # TRAIN_PATH = "data/datasets/semeval2017/semeval2017_task10_train.json"
    # SAVE_PATH_TRAIN = "data/save/semeval2017/train/"
    # EVAL_PATH = "data/datasets/semeval2017/semeval2017_task10_test.json"
    # SAVE_PATH_EVAL = "data/save/semeval2017/eval/"
    # CONFIG_PATH = "configs/semeval.json"
    # TOKENIZER_PATH = "scibert_scivocab_uncased/"

    # TRAIN_PATH = "data/datasets/conll03/conll03_train.json"
    # SAVE_PATH_TRAIN = "data/save/conll03/train/"
    # EVAL_PATH = "data/datasets/conll03/conll03_test.json"
    # SAVE_PATH_EVAL = "data/save/conll03/eval/"
    # CONFIG_PATH = "configs/conll03.json"
    # TOKENIZER_PATH = "bert-base-uncased"

    for version in [1,2,3]:
        for fraction in [0.25, 0.5, 0.75]:
            TRAIN_PATH = "../speer/data/datasets/za{}/za_train_{}.json".format(version, fraction)
            SAVE_PATH_TRAIN = "data/save/za{}/train/{}/".format(version, fraction)
            EVAL_PATH = "../speer/data/datasets/za{}/za_test.json".format(version)
            SAVE_PATH_EVAL = "data/save/za{}/eval/{}/".format(version, fraction)
            CONFIG_PATH = "configs/za.json"
            TOKENIZER_PATH = "scibert_scivocab_uncased/"



    # TRAIN_PATH = "../speer/data/datasets/za{}_string/za_train_{}.json".format(za_version, frac)
    # SAVE_PATH_TRAIN = "data/save/za{}_string/train/{}/".format(za_version, frac)
    # EVAL_PATH = "../speer/data/datasets/za{}_string/GOLD.json".format(za_version)
    # SAVE_PATH_EVAL = "data/save/za{}_string/eval/{}/".format(za_version, frac)
    # CONFIG_PATH = "configs/za.json"
    # TOKENIZER_PATH = "scibert_scivocab_uncased/"

    


    