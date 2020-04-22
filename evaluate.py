from sklearn.metrics import precision_recall_fscore_support as prfs
import numpy as np
import json
import argparse
from typing import List, Tuple, Dict

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
    print("labels:", labels[:30])
    print("gt:", gt_all[:30])
    print("pred:", pred_all[:30])
    per_type = prfs(gt_all, pred_all, labels=labels, average=None)
    print(per_type)
    micro = prfs(gt_all, pred_all, labels=labels, average='micro')[:-1]
    macro = prfs(gt_all, pred_all, labels=labels, average='macro')[:-1]
    total_support = sum(per_type[-1])

    if print_results:
        _print_results(per_type, list(micro) + [total_support], list(macro) + [total_support], types)

    return [m * 100 for m in micro + macro]

## Tuple = 
#   (start, end, entity_type)
#   or
#   ((head_start, head_end, pred_head_type), (tail_start, tail_end, pred_tail_type), pred_rel_type)
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

    # for gt, pred in zip(gt_flat, pred_flat):
        # print(gt)
        # print(pred)
        # print()

    metrics = _compute_metrics(gt_flat, pred_flat, types, print_results)
    return metrics

def _convert_entity_tuples(sequence):
    entity_tuples = []
    entities = sequence["entities"]
    for entity in entities:
        tuple_ = (entity["start"], entity["end"], entity["type"])
        entity_tuples.append(tuple_)
        
    return entity_tuples

def _convert_relation_tuples(sequence):
    relation_tuples = []
    entities = sequence["entities"]
    relations = sequence["relations"]
    for relation in relations:
        head_entity = entities[relation["head"]]
        head_tuple = (head_entity["start"], head_entity["end"], head_entity["type"])
        tail_entity = entities[relation["tail"]]
        tail_tuple = (tail_entity["start"], tail_entity["end"], tail_entity["type"])
        tuple_ = (head_tuple, tail_tuple, relation["type"])
        relation_tuples.append(tuple_)
        
    return relation_tuples

def evaluate(gt_path, pred_path):
    with open(gt_path, 'r', encoding='utf-8') as f:
        gt_dataset = json.load(f)

    with open(pred_path, 'r', encoding='utf-8') as f:
        pred_dataset = json.load(f)

    gt_entities = []
    pred_entities = []
    gt_relations = []
    pred_relations = []

    for gt_sequence, pred_sequence in zip(gt_dataset, pred_dataset):
        gt_entity_tuples = _convert_entity_tuples(gt_sequence)
        pred_entity_tuples = _convert_entity_tuples(pred_sequence)
        gt_relation_tuples = _convert_relation_tuples(gt_sequence)
        pred_relation_tuples = _convert_relation_tuples(pred_sequence)
        gt_entities.append(gt_entity_tuples)
        pred_entities.append(pred_entity_tuples)
        gt_relations.append(gt_relation_tuples)
        pred_relation_tuples.append(pred_relation_tuples)

    print("")
    print("--- Entities (named entity recognition (NER)) ---")
    print("An entity is considered correct if the entity type and span is predicted correctly")
    print("")
    ner_eval = _score(gt_entities, pred_entities, print_results=True)
    print("")
    print("--- Relations ---")
    print("")
    print("With named entity classification (NEC)")
    print("A relation is considered correct if the relation type and the two "
            "related entities are predicted correctly (in span and entity type)")
    print("")
    print(len(gt_relations), len(pred_relations))
    rel_eval = _score(gt_relations, pred_relations, print_results=True)

    return ner_eval, rel_eval

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate spert json formatted dataset')
    parser.add_argument('gt_path', type=str, help='path to the ground truth dataset.json file')
    parser.add_argument('pred_path', type=str, help='path to the predicted dataset.json file')
    args = parser.parse_args()
    evaluate(args.gt_path, args.pred_path)
    