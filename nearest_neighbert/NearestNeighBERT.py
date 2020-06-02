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

# Local
import data
from embedders import BertEmbedder
from evaluate import compare_datasets
from evaluate import evaluate as eval
import utils




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
                 table=[], tokenizer=None, f_reduce="mean", neg_label="O"):
        self.k = k
        self.f_voting = f_voting
        self.f_similarity = f_similarity
        self.index = index
        self.table = table
        self.tokenizer = tokenizer
        self.f_reduce = f_reduce
        self.neg_label = neg_label
        self.config = None


    def configure(self, path="configs/config_za.json"):
        with open(path, 'r', encoding='utf-8') as json_config:
            config = json.load(json_config)
            self.k = config.get("k", self.k)
            self.f_voting = config.get("voting_f", self.f_voting)
            self.f_similarity = config.get("similarity_f", self.f_similarity)
            self.f_reduce = config.get("f_reduce", self.f_reduce)
            self.neg_label = config.get("neg_label", self.neg_label)
        self.config = config

        return self


    def ready_training(self, tokenizer_path):
        self.tokenizer = BertEmbedder(tokenizer_path)
        self.index = data.init_faiss(self.f_reduce, self.f_similarity, self.tokenizer)


    def train(self, dataset_path, tokenizer_path='scibert-base-cased', save_path="data/", save=True):   
        self.ready_training(tokenizer_path)
        data_generator = data.prepare_dataset(dataset_path, self.tokenizer, self.neg_label, self.f_reduce)
        print("Training...")
        for tokens in data_generator:
            if tokens:
                token_embeddings = [t.embedding for t in tokens]
                token_embeddings = torch.stack(token_embeddings).numpy()
                token_entries = [t.to_table_entry() for t in tokens]
                self.train_(token_embeddings, token_entries)

        if save:
            data.save_faiss(self.index, self.table, "tokens", save_path)
            train_config_path = os.path.join(save_path, "train_config.json")
            self.save_config(train_config_path)
    

    def train_(self, embeddings, entries):
        """
        This function can be used to train incremently, assumes already processed embeddings
        and corresponding entries with info about them such as label.
        """  
        self.table += entries
        self.index.add(np.array(embeddings))


    def ready_inference(self, index_path, tokenizer_path='scibert-base-uncased', device=DEVICE):
        self.tokenizer = BertEmbedder(tokenizer_path)
        self.index, self.table = data.load_faiss(index_path, device, "tokens")


    def infer(self, document, verbose=False):
        inference_document = []
        for sentence in tqdm(document["sentences"]):
            string_tokens, _, token2char, = self.tokenizer.split(sentence)
            bert_tokens, tok2orig, orig2tok = self.tokenizer.tokenize_with_mapping(string_tokens)
            embeddings = self.tokenizer.embed(bert_tokens)[1:-1] # skip special tokens

            tokens = data.create_tokens(string_tokens)
            if tokens:
                token_embeddings = [t.calculate_embedding(embeddings, bert_tokens, orig2tok, self.f_reduce) for t in tokens]
                q = torch.stack(token_embeddings).numpy()
                D, I = self.index.search(q, self.k)
                a = D-np.min(D)
                b = np.max(D)-np.min(D)
                D_norm = np.divide(a, b, out=np.zeros_like(a), where=b!=0)
                if self.f_similarity=='L2':
                    D_norm = 1-D_norm # 1-distance for similarity
                labels, neighbors = self.vote(D_norm, I)
                self.assign_labels(tokens, labels, neighbors, verbose)

            prediction = self.convert_prediction(string_tokens, tokens)
            inference_document.append(prediction)

        return inference_document


    def infer_(self, embeddings, label_type=str):
        """
        This function alternative assumes already calculated embeddings
        and returns voted labels based on training index labels
        """
        q = np.array(embeddings)
        D, I = self.index.search(q, self.k)
        labels, neighbors = self.vote(D, I, label_type)

        return labels, neighbors


    def evaluate(self, evaluation_path, results_path="data/results/", verbose=True):
        """
        Evaluate a dataset by doing inference on the data without labels. 
        See data.py for example format of the dataset.
        """
        dataset = json.load(open(evaluation_path))
        sentences = [' '.join(entry["tokens"]) for entry in dataset]
        utils.create_dir(results_path)

        print("Evaluating...")
        predictions = self.infer({"sentences": sentences}, verbose=verbose)
        predictions_path = os.path.join(results_path, "predictions.json")
        with open(predictions_path, 'w', encoding='utf-8') as f:
            json.dump(predictions, f)
        ner_eval, rel_eval = eval(evaluation_path, predictions_path, self.tokenizer)

        predicted_examples_path = os.path.join(results_path, "predicted_examples.txt")
        compare_datasets(evaluation_path, predictions_path, predicted_examples_path)

        final_config_path = os.path.join(results_path, "final_config.json")
        self.save_config(final_config_path)

        return ner_eval, rel_eval


    def vote(self, similarities, indices, label_type=str):
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
                vote = neighbor["label"]
                weight = VOTING_F[self.f_voting](similarities[i][j], j)
                weight_counter[vote] += weight
            pred_label = weight_counter.most_common(1)[0][0]
            pred_labels.append(label_type(pred_label))
            all_neighbors.append(nearest_neighbors)

        return pred_labels, all_neighbors


    def assign_labels(self, tokens, labels, neighbors, verbose=False):
        for t, l, n in zip(tokens, labels, neighbors):
            t.label = l
            if verbose:
                print("Predicted: ", t)
                for i, neighbor in enumerate(n):
                    print("\t (nn {}) {}".format(i, neighbor["string"]))
                print()


    def _expand_entities(self, string_tokens, tokens):
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
    # TRAIN_PATH = "../spert/data/datasets/semeval2017_task10/semeval2017_task10_train.json"
    # SAVE_PATH_TRAIN = "data/save/semeval2017/train/"
    # EVAL_PATH = "../spert/data/datasets/semeval2017_task10/semeval2017_task10_dev.json"
    # SAVE_PATH_EVAL = "data/save/semeval2017/eval/"
    # CONFIG_PATH = "configs/semeval.json"
    # TOKENIZER_PATH = "scibert_scivocab_uncased/"

    TRAIN_PATH = "data/datasets/conll03/conll03_train.json"
    SAVE_PATH_TRAIN = "data/save/conll03/train/"
    EVAL_PATH = "data/datasets/conll03/conll03_dev.json"
    SAVE_PATH_EVAL = "data/save/conll03/eval/"
    CONFIG_PATH = "configs/conll03.json"
    TOKENIZER_PATH = "bert-base-uncased"

    
    knn = NearestNeighBERT().configure(CONFIG_PATH)
    # knn.train(TRAIN_PATH, TOKENIZER_PATH, SAVE_PATH_TRAIN)
    knn.ready_inference(SAVE_PATH_TRAIN, TOKENIZER_PATH)
    knn.evaluate(EVAL_PATH, SAVE_PATH_EVAL, verbose=False)

    