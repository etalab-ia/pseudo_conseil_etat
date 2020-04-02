#  Licensed to the Apache Software Foundation (ASF) under one
#  or more contributor license agreements.  See the NOTICE file
#  distributed with this work for additional information
#  regarding copyright ownership.  The ASF licenses this file
#  to you under the Apache License, Version 2.0 (the
#  "License"); you may not use this file except in compliance
#  with the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing,
#  software distributed under the License is distributed on an
#  "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
#  KIND, either express or implied.  See the License for the
#  specific language governing permissions and limitations
#  under the License.
# Based on https://github.com/ELS-RD/anonymisation/blob/master/flair_display_errors.py
'''Predicts and evaluates a CoNLL format file given a model

Usage:
    flair_predict_conll.py <model_folder> <conll_file>

Arguments:
    <model_folder> Trained Flair NER model folder
    <conll_file> Folder path with the DOC decisions files to transform to TXT
'''

import copy
import os
import random
from argopt import argopt

import torch
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import numpy as np
np.random.seed(0)
from flair.data import Corpus, Sentence
from flair.datasets import DataLoader, ColumnDataset
from flair.models import SequenceTagger
from pathlib import Path

# reproducibility
random.seed(5)


def main(conll_file_path: str, model_folder: str) -> None:
    conll_file_path = Path(conll_file_path)
    test_set = ColumnDataset(path_to_column_file=conll_file_path, column_name_map={0: 'text', 1: 'ner'})
    tagger: SequenceTagger = SequenceTagger.load(model=os.path.join(model_folder, 'final-model.pt'))
    test_results, _ = tagger.evaluate(data_loader=DataLoader(test_set, batch_size=8))
    print(test_results.detailed_results)
    sentences_original = test_set
    sentences_predict = copy.deepcopy(test_set)
    # clean tokens in case there is a bug
    for s in sentences_predict:
        for t in s:
            t.tags = {}

    _ = tagger.predict(sentences=sentences_predict,
                       mini_batch_size=32,
                       embedding_storage_mode="none",
                       verbose=True)

    for sent in sentences_predict:
        for tok in sent:
            print(f"{tok.text}\t{tok.get_tag('ner').value}")
        print()

if __name__ == '__main__':
    parser = argopt(__doc__).parse_args()
    model_folder = parser.model_folder
    conll_file_path = parser.conll_file
    main(conll_file_path=conll_file_path,
         model_folder=model_folder)
