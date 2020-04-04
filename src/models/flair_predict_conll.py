# Inspired on https://github.com/ELS-RD/anonymisation/blob/master/flair_display_errors.py
'''Predicts and evaluates a CoNLL format file given a model

Usage:
    flair_predict_conll.py <model_folder> <conll_file> <output_conll_file>

Arguments:
    <model_folder> Trained Flair NER model folder
    <conll_file> Folder path with the DOC decisions files to transform to TXT
    <output_conll_file> Annotated CoNLL file using the model
'''

import copy
import os
import random

import pandas as pd
from argopt import argopt
from sacremoses import MosesDetokenizer
import torch

from src.results.confusion_matrix_pretty_print import print_confusion_matrix
from src.results.evaluate_results import print_errors

torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import numpy as np

np.random.seed(0)
from flair.data import Corpus, Sentence
from flair.datasets import DataLoader, ColumnDataset
from flair.models import SequenceTagger
from pathlib import Path
from seqeval.metrics import classification_report
from seqeval.metrics import f1_score

# reproducibility
random.seed(5)
detok = MosesDetokenizer()


def print_results(y_true, y_pred):
    print(classification_report(y_true, y_pred))
    fscore = f1_score(y_true, y_pred)
    print(f"F=score (micro??): {fscore:2f}")


def main(conll_file_path: str, model_folder: str, output_file_path: str) -> None:
    conll_file_path = Path(conll_file_path)
    test_set = ColumnDataset(path_to_column_file=conll_file_path, column_name_map={0: 'text', 1: 'ner'})
    tagger: SequenceTagger = SequenceTagger.load(model=os.path.join(model_folder, 'final-model.pt'))
    # test_results, _ = tagger.evaluate(data_loader=DataLoader(test_set, batch_size=8))
    # print(test_results.detailed_results)
    sentences_original = test_set
    sentences_predict = copy.deepcopy(test_set)
    # clean tokens in case there is a bug
    for s in sentences_predict:
        for t in s:
            t.tags = {}

    _ = tagger.predict(sentences=sentences_predict,
                       mini_batch_size=8,
                       embedding_storage_mode="none",
                       verbose=True)
    tokens, y_pred, y_true = [], [], []
    with open(output_file_path, "w") as out:
        for sent_orig, sent_pred in zip(sentences_original, sentences_predict):
            for tok_orig, tok_pred in zip(sent_orig, sent_pred):
                tokens.append(tok_pred.text)
                y_pred.append(tok_pred.get_tag('ner').value)
                y_true.append(tok_orig.get_tag('ner').value)
                result_str = f"{detok.detokenize([tok_pred.text])}\t{tok_orig.get_tag('ner').value}\t{tok_pred.get_tag('ner').value}"
                # print(result_str)
                out.write(result_str + "\n")

            # print()
            out.write("\n")

    print_results(y_true=y_true, y_pred=y_pred)
    print()
    print_confusion_matrix(y_true=y_true, y_pred=y_pred, labels=["B-LOC", "I-LOC", "B-PER_NOM", "I-PER_NOM",
                                                                 "B-PER_PRENOM", "B-PER_PRENOM", "O"])

    results_df = pd.read_csv(output_file_path, sep="\t", names=["token", "true_tag", "pred_tag"], skip_blank_lines=False)
    print_errors(results_df)



if __name__ == '__main__':
    parser = argopt(__doc__).parse_args()
    model_folder = parser.model_folder
    conll_file_path = parser.conll_file
    output_file_path = parser.output_conll_file
    main(conll_file_path=conll_file_path,
         model_folder=model_folder,
         output_file_path=output_file_path)
