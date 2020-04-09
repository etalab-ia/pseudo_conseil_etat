'''Predicts a text file given a model. Outputs the annotation as a CoNLL file (with two columns:
token, predicted_tag) or as a text file

Usage:
    flair_predict_text.py <model_folder> <text_file>  <output_format> [options]

Arguments:
    <model_folder>              Trained Flair NER model folder
    <text_file>                Folder or file path with the DOC decisions files to transform to TXT
    <output_format>            If "text" only save the document without the entities found.
                               If "conll" save the document as a CoNLL file. "both" save it as both formats
'''

import copy
import glob
import os
import random
import re

import pandas as pd
from argopt import argopt
from sacremoses import MosesDetokenizer
import torch

from flair.data import Corpus, Sentence
from flair.datasets import ColumnDataset
from flair.models import SequenceTagger
from pathlib import Path

from tqdm import tqdm

from src.utils.tokenizer import moses_tokenize, moses_detokenize
from pathlib import Path

SUFFIX_CONLL = "_NLP_CoNLL"
SUFFIX_TEXT = "_annon"

def main(text_file_path: str, model_folder: str, output_format: str) -> None:
    text_file_path = Path(text_file_path)

    # Get
    if os.path.isfile(text_file_path):
        list_files = [text_file_path]
    else:
        list_files = [Path(path) for path in glob.glob(text_file_path.as_posix() + "/**/*.txt", recursive=True)
                      if SUFFIX_TEXT not in path and SUFFIX_CONLL not in path]


    # load model
    tagger = SequenceTagger.load(model=os.path.join(model_folder, 'best-model.pt'))

    file_sentences = []
    for file_ in tqdm(list_files):
        with open(file_, encoding="utf-8-sig") as text_file:
            sentences_predict = [Sentence(line.strip(), use_tokenizer=lambda x: moses_tokenize(x))
                                 for line in text_file.readlines() if line.strip()]

        _ = tagger.predict(sentences=sentences_predict,
                           mini_batch_size=8,
                           embedding_storage_mode="none",
                           verbose=True)

        if output_format == "conll" or output_format == "both":
            with open(os.path.join(file_.parent, file_.stem + f"{SUFFIX_CONLL}.txt"), "w") as out:
                out.write(f"-DOCSTART-\tO\n\n")
                for sent_pred in sentences_predict:
                    for tok_pred in sent_pred:
                        result_str = f"{moses_detokenize([tok_pred.text])}\t{tok_pred.get_tag('ner').value}"
                        # print(result_str)
                        out.write(result_str + "\n")
                    out.write("\n")
        if output_format == "txt" or output_format == "both":
            with open(os.path.join(file_.parent, file_.stem + f"{SUFFIX_TEXT}.txt"), "w") as out:
                for sent_pred in sentences_predict:
                    out.write(re.sub(r"<.*>", "...", sent_pred.to_tagged_string()))
                    out.write("\n")


if __name__ == '__main__':
    parser = argopt(__doc__).parse_args()
    model_folder = parser.model_folder
    text_file_path = parser.text_file
    output_format = parser.output_format
    main(text_file_path=text_file_path,
         model_folder=model_folder,
         output_format=output_format)
