'''Transforms a DOC decision file into a TXT pseudonymized decision file. It loads a model and uses it to predict.

Usage:
    run_pseudo.py <file_path>  [options]

Arguments:
    <file_path>                        File path of the DOC decision  to transform to a pseudonymized TXT
    --cores=<n> CORES                  Number of cores to use [default: 1:int]
'''

import subprocess
import re
import sys

sys.path.insert(0, '/home/pavel/temp/ukp_forks/emnlp2017-bilstm-cnn-crf/')
from argopt import argopt
from flair.data import Sentence
from flair.models import SequenceTagger
from tqdm import tqdm

if __name__ == '__main__':
    parser = argopt(__doc__).parse_args()
    decision_file_path = parser.file_path
    n_jobs = parser.cores

    # 1 Convert to txt
    # subprocess.check_call(["python", "/home/pavel/etalab/code/conseil_etat/src/data/doc2txt.py", decision_file_path])
    decision_txt_path = decision_file_path[:-3] + "txt"
    decision_txt_path = "/data/conseil_etat/decisions/test_28nov/4018841pr1.txt"

    # 2 Predict text
    model = SequenceTagger.load('/home/pavel/etalab/code/conseil_etat/models/baseline_ner/final-model.pt')
    with open(decision_txt_path) as filo:
        flair_lines = [Sentence(l.strip(), use_tokenizer=lambda x: x.split()) for l in filo.readlines() if
                       l and len(l.strip()) > 1]

    # predict tags and print
    anon_lines = []
    for flair_line in tqdm(flair_lines):
        model.predict(flair_line)
        print(flair_line.to_tagged_string())
        anon_lines.append(re.sub(r"<.*>", "...", flair_line.to_tagged_string()))
    # print(sentence.to_tagged_string())

    for line in anon_lines:
        print(line)

    # 3 Load and Predict
