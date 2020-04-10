from itertools import groupby
from collections import defaultdict
import string
import json
import os
from pathlib import Path
import math
from tqdm import tqdm

def split_with_indices(s, c=' '):
    '''Split string and return start and end positions of words'''
    p = 0
    for k, g in groupby(s, lambda x:x==c):
        q = p + sum(1 for i in g)
        if not k:
            yield p, q
        p = q
        
def get_token_span(text, start, end):
    span = text[start:end]
    span_length = len(span.split(' '))
    word_positions = []
    positions = list(split_with_indices(text))
    for position in positions:
        if position[0] == start:
            word_positions.append(positions.index(position))
            break
    word_positions += [p+word_positions[0] for p in range(1, span_length)]

    return word_positions

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

def soft_list_equality(list1, list2):
    equal = True
    if not len(list1)==len(list2):
        equal = False
    for elem in list1:
        if elem not in list2:
            equal = False

    return equal
    
def is_whitespace(c):
    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F or c=="\xa0":
        return True
    return False

def get_vector_batches(data, embedder, target_set, fraction=1):
    print("Getting {} batches...".format(target_set))
    batches = []
    max_samples = get_max_samples(data, target_set, fraction)
    for i, sample in tqdm(enumerate(data.sample_generators[target_set]), total=max_samples):
        vectors = data.vectorize(sample, embedder)
        batches.append(vectors)
        if i == max_samples:
            break
    data.sample_generators[target_set] = data.read_again(target_set)
    
    return batches

def get_max_samples(data, target_set, fraction):
    dataset_length = sum([1 for sample in data.sample_generators[target_set]])
    data.read_again(target_set)
    max_samples = math.ceil(fraction*dataset_length)
    data.sample_generators[target_set] = data.read_again(target_set)

    return max_samples

def cache_vectors(vectors, path):
    print("Caching {} vectors...".format(len(vectors)))
    d = {}
    for i, v in enumerate(vectors):
        d[i] = v.to_dict()
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(d, f)

def get_cached_vectors(path):
    vectors = []
    if not path:
        return vectors
    if os.path.isfile(path):
        print("Using cached vectors from file...")
        with open(path, encoding='utf-8') as f:
            vectors_dict = json.load(f)
            vectors = [Vector().initialize_from_dict(vectors_dict[d]) for d in vectors_dict]   
    return vectors

def chunk(sequence, chunk_size=64):
    chunk = []
    for item in sequence:
        if len(chunk)==chunk_size:
            yield chunk
            chunk = []
        chunk.append(item)
    yield chunk

def check_path(path, verbose=True):
    if os.path.exists(path):
        if verbose:
            print("Path [{}] already exists, please rename or delete existing file.".format(path))
            print("Adding '_temp' to original path for now")
        extension = path.split('.')[-1]
        return path.replace("."+extension, "")+"_temp.{}".format(extension)
    
    return path

def is_file(path):
    if path == None:
        return False

    return os.path.isfile(path)
    

def tensor2D_to_file(tensor_2D, file_object):
    for row in tensor_2D:
        line =  " ".join(map(str, row.tolist()))+'\n'
        file_object.write(line)

def create_spans(sequence):
    #TODO
    pass

def create_dir_structure(path_dict):
    for _, path in path_dict.items():
        directory = os.path.dirname(path)
        Path(directory).mkdir(parents=True, exist_ok=True) 

def create_dir(path):
    directory = os.path.dirname(path)
    Path(directory).mkdir(parents=True, exist_ok=True)

