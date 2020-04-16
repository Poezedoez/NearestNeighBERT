from typing import List, Dict

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
        bert_tokens, tok2orig, orig2tok = tokenizer.tokenize_with_mapping(original_tokens)
        embeddings = tokenizer.embed(bert_tokens)[1:-1] # skip special tokens  

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