from itertools import groupby
from collections import defaultdict
import string
import json
import os
from pathlib import Path
import math
from tqdm import tqdm
        
def get_subword_mapping(tokens):
    ''' Unknown words are mapped by BERT tokenizer to multiple subword tokens.
    This functions identifies which subwords are part of the original unknown words
    by mapping their string positions in the text to the sentence indices of the subwords'''
    mapping = defaultdict(list)
    sentence_index = 0
    for i, token in enumerate(tokens):
        mapping[sentence_index].append(i)
        if not token.startswith('##'):
            sentence_index += 1
    
    return mapping
    
def is_whitespace(c):
    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F or c=="\xa0":
        return True
    return False

def chunk(sequence, chunk_size=64):
    chunk = []
    for item in sequence:
        if len(chunk)==chunk_size:
            yield chunk
            chunk = []
        chunk.append(item)
    yield chunk

def is_file(path):
    if path == None:
        return False

    return os.path.isfile(path)

def create_dir_structure(path_dict):
    for _, path in path_dict.items():
        directory = os.path.dirname(path)
        Path(directory).mkdir(parents=True, exist_ok=True) 

def create_dir(path):
    directory = os.path.dirname(path)
    Path(directory).mkdir(parents=True, exist_ok=True)

