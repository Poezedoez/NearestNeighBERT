from abc import ABC, abstractmethod
import utils
import os, string
import torch
from transformers import BertModel, BertTokenizer

class Embedder(ABC):
    def __init__(self, embedding_size, indicator):
        self.embedding_size = embedding_size
        self.indicator = indicator
        super().__init__()

    
    def split(self, text):
        """
        Split a text string into word tokens.
        !! No (BERT) subword tokenization here !!
        """
        doc_tokens = []
        char_to_word_offset = []
        word_to_char_offset = []
        new_token = True
        for i, c in enumerate(text):
            if utils.is_whitespace(c):
                new_token = True
            else:
                if c in string.punctuation:
                    doc_tokens.append(c)
                    word_to_char_offset.append(i)
                    new_token = True
                elif new_token:
                    doc_tokens.append(c)
                    word_to_char_offset.append(i)
                    new_token = False
                else:
                    doc_tokens[-1] += c
                    new_token = False
            char_to_word_offset.append(len(doc_tokens) - 1)
            
        
        return doc_tokens, char_to_word_offset, word_to_char_offset

    @abstractmethod
    def tokenize(self):
        pass

    @abstractmethod
    def embed(self):
        pass
    
    @abstractmethod
    def __repr__(self):
        pass
    
    @abstractmethod
    def __str__(self):
        pass

class BertEmbedder(Embedder):
    def __init__(self, pretrained_weights='bert-base-uncased', transformer_layer='last',
    embedding_size=768):
        
        self.pretrained_weights = pretrained_weights
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_weights)
        self.encoder = BertModel.from_pretrained(pretrained_weights, output_hidden_states=True)
        self.layer = {'last':-1, 'penult':-2}.get(transformer_layer)

        super().__init__(embedding_size, transformer_layer+'_'+pretrained_weights)

    def tokenize(self, sequence):
        if isinstance(sequence, str):
            tokens = self.tokenizer.tokenize(sequence)
        else: # is list
            tokens = []
            for word in sequence:
                tokens += self.tokenizer.tokenize(word)

        return tokens

    def embed(self, sequence):
        indices = torch.tensor([self.tokenizer.encode(sequence, add_special_tokens=True)])
        with torch.no_grad():
            hidden_states = self.encoder(indices)[-1]
            embeddings = hidden_states[self.layer]
        
        return torch.squeeze(embeddings)
        
    def tokenize_with_mapping(self, doc_tokens):
        ''' Returns mapping between BERT tokens
        and input tokens.
        '''
        tok_to_orig_index = []
        orig_to_tok_index = []
        all_doc_tokens = []
        for (i, token) in enumerate(doc_tokens):
            orig_to_tok_index.append(len(all_doc_tokens))
            sub_tokens = self.tokenizer.tokenize(token)
            for sub_token in sub_tokens:
                tok_to_orig_index.append(i)
                all_doc_tokens.append(sub_token)

        return all_doc_tokens, tok_to_orig_index, orig_to_tok_index

    def select_embeddings(self, original_tokens, selection_f="first"):
        """
        Here it can be decided which embeddings are selected
        to represent the original tokens. Several strategies can be applied.
        E.g. the first subtoken, an average of subtokens, or a pool.
        """

        def _first(embeddings, tok2orig, orig2tok, bert_tokens):
            selected_embeddings = []
            embedding_tokens = []
            for _, token in enumerate(orig2tok):
                selected_embeddings.append(embeddings[token])
                embedding_tokens.append(bert_tokens[token]) 

            return selected_embeddings, embedding_tokens
        
        def _pool(embeddings, tok2orig, orig2tok, bert_tokens):
            pass

        def _average(embeddings, tok2orig, orig2tok, bert_tokens):
            pass

        f_map = {"first": _first, "pool": _pool, "average": _average}
        bert_tokens, tok2orig, orig2tok = self.tokenize_with_mapping(original_tokens)
        embeddings = self.embed(bert_tokens)[1:-1] # skip special tokens  

        return f_map[selection_f](embeddings, tok2orig, orig2tok, bert_tokens)

    def __repr__(self):
        return "BertEmbedder()"

    def __str__(self):
        return "_BertEmbedder_{}Layer_{}Weights".format(self.transformer_layer, self.pretrained_weights)