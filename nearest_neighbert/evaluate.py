from sklearn.metrics import precision_recall_fscore_support as prfs
import numpy as np
import json
import argparse
from typing import List, Tuple, Dict
from nearest_neighbert.data import adjust_annotation
import sys

# From spert.evaluator class
# https://github.com/markus-eberts/spert/blob/master/spert/evaluator.py

def _get_row(data, label):
    row = [label]
    for i in range(len(data) - 1):
        row.append("%.2f" % (data[i] * 100))
    row.append(data[3])
    return tuple(row)

def _print_results(per_type: List, micro: List, macro: List, types: List):
    columns = ('type', 'precision', 'recall', 'f1-score', 'support')

    row_fmt = "%20s" + (" %12s" * (len(columns) - 1))
    results = [row_fmt % columns, '\n']

    metrics_per_type = []
    for i, t in enumerate(types):
        metrics = []
        for j in range(len(per_type)):
            metrics.append(per_type[j][i])
        metrics_per_type.append(metrics)

    for m, t in zip(metrics_per_type, types):
        results.append(row_fmt % _get_row(m, t))
        results.append('\n')

    results.append('\n')

    # micro
    results.append(row_fmt % _get_row(micro, 'micro'))
    results.append('\n')

    # macro
    results.append(row_fmt % _get_row(macro, 'macro'))

    results_str = ''.join(results)
    print(results_str)

def _compute_metrics(gt_all, pred_all, types, print_results: bool = False):
    labels = [t for t in types]
    per_type = prfs(gt_all, pred_all, labels=labels, average=None)
    micro = prfs(gt_all, pred_all, labels=labels, average='micro')[:-1]
    macro = prfs(gt_all, pred_all, labels=labels, average='macro')[:-1]
    total_support = sum(per_type[-1])

    if print_results:
        _print_results(per_type, list(micro) + [total_support], list(macro) + [total_support], types)

    return [m * 100 for m in micro + macro]

## Tuple = (start, end, entity_type)

def _score(gt: List[List[Tuple]], pred: List[List[Tuple]], print_results: bool = False):
    assert len(gt) == len(pred)

    gt_flat = []
    pred_flat = []
    types = set()

    for (sample_gt, sample_pred) in zip(gt, pred):
        union = set()
        union.update(sample_gt)
        union.update(sample_pred)

        for s in union:
            if s in sample_gt:
                t = s[2]
                gt_flat.append(t)
                types.add(t)
            else:
                gt_flat.append("0")

            if s in sample_pred:
                t = s[2]
                pred_flat.append(t)
                types.add(t)
            else:
                pred_flat.append("0")
    metrics = _compute_metrics(gt_flat, pred_flat, types, print_results)

    return metrics

def _convert_span_tuples(sequence):
    span_tuples = []
    entities = sequence["entities"]
    for span in entities:
        tuple_ = (span["start"], span["end"], span["type"])
        span_tuples.append(tuple_)
        
    return span_tuples

def _convert_token_tuples(sequence):
    token_tuples = []
    entities = sequence["entities"]
    string_tokens = sequence["tokens"]
    for span in entities:
        span_range = range(span["start"], span["end"])
        for index in span_range:
            tuple_ = (index, index+1, span["type"])
            token_tuples.append(tuple_)
        
    return token_tuples

def evaluate(gt_path, pred_path, tokenizer):
    with open(gt_path, 'r', encoding='utf-8') as f:
        gt_dataset = json.load(f)

    with open(pred_path, 'r', encoding='utf-8') as f:
        pred_dataset = json.load(f)

    gt_spans = []
    pred_spans = []
    gt_tokens = []
    pred_tokens = []

    for gt_sequence, pred_sequence in zip(gt_dataset, pred_dataset):
        adjust_annotation(gt_sequence, tokenizer)
        adjust_annotation(pred_sequence, tokenizer)
        gt_spans.append(_convert_span_tuples(gt_sequence))
        pred_spans.append(_convert_span_tuples(pred_sequence))
        gt_tokens.append(_convert_token_tuples(gt_sequence))
        pred_tokens.append(_convert_token_tuples(pred_sequence))
        
    
    print("")
    print("--- Entities (named entity recognition (NER)) ---")
    print("An entity span is considered correct if the entity type and span start/end is predicted correctly")
    ner_span_eval = _score(gt_spans, pred_spans, print_results=True)[:3]
    print("")
    print("An entity token is considered correct if the entity type is predicted correctly")
    ner_token_eval = _score(gt_tokens, pred_tokens, print_results=True)[:3]
    print("")

    return ner_span_eval, ner_token_eval


def compare_datasets(gt_path, pred_path, output_path=None):
    with open(gt_path, 'r', encoding='utf-8') as f:
        gt_dataset = json.load(f)

    with open(pred_path, 'r', encoding='utf-8') as f:
        pred_dataset = json.load(f)

    assert len(gt_dataset)==len(pred_dataset)

    file_object = open(output_path, 'w', encoding='utf-8') if output_path else sys.stdout
    for gt_sentence, pred_sentence in zip(gt_dataset, pred_dataset):
        gt_tokens = gt_sentence["tokens"]
        print("|{}| {} \n".format(gt_sentence["orig_id"], " ".join(gt_tokens)), file=file_object)
        for entity in gt_sentence["entities"]:
            entity_tokens = gt_tokens[entity["start"]:entity["end"]]
            line = "[gold] \t {} \t {}".format(" ".join(entity_tokens), entity["type"])
            print(line, file=file_object)
        pred_tokens = pred_sentence["tokens"]
        for entity in pred_sentence["entities"]:
            entity_tokens = pred_tokens[entity["start"]:entity["end"]]
            line = "[pred] \t {} \t {}".format(" ".join(entity_tokens), entity["type"])
            print(line, file=file_object)
        print('-'*50, file=file_object)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate spert json formatted dataset')
    parser.add_argument('gt_path', type=str, help='path to the ground truth dataset.json file')
    parser.add_argument('pred_path', type=str, help='path to the predicted dataset.json file')
    args = parser.parse_args()
    evaluate(args.gt_path, args.pred_path)
    