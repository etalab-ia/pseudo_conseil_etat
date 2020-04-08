'''
Transforms text files into CoNLL like text files, that is, one word per line.

Usage:
    hand_made2conll.py <docs_folder>  [options]

Arguments:
    <docs_folder>           Folder path with the txt annotated files to transform to CoNLL or the folder with the files
    --cores=<n> CORES       Number of cores to use [default: 1:int]
'''
import glob
import logging
import os
import re

from argopt import argopt
from joblib import Parallel, delayed
from sacremoses import MosesTokenizer, MosesPunctNormalizer
from tqdm import tqdm

# logging.basicConfig(filename='./logs/normal_doc2conll.log', filemode='w',
#                     format="%(asctime)s:%(levelname)s:\t%(message)s", level=logging.INFO)

mt = MosesTokenizer(lang="fr")
mpn = MosesPunctNormalizer(lang="fr")


def tokenize(phrase):
    phrase = mpn.normalize(phrase)
    tokens = mt.penn_tokenize(phrase)
    return tokens


def get_word_per_line(text_lines):
    all_tokens = []
    for line in text_lines:
        if not line:
            continue
        line_tokens = tokenize(line)
        all_tokens.append(line_tokens)

    return all_tokens


def run(annotated_txt_path):
    logging.info(f"Treating file {annotated_txt_path}")
    tqdm.write(f"Treating file {annotated_txt_path}")
    decision_folder = os.path.dirname(annotated_txt_path)

    # TODO Here, change to a sentence segmentation text_lines
    with open(annotated_txt_path) as filo:
        text_lines = [l.strip() for l in filo.readlines()]

    all_tokens = get_word_per_line(text_lines)

    with open(os.path.join(decision_folder, os.path.basename(annotated_txt_path)[:-4] + "_Tokens_CoNLL.txt"),
              "w") as conll:
        for tokens in all_tokens:
            for tok in tokens:
                conll.write(f"{tok}\n")
            conll.write("\n")

    return 1, annotated_txt_path


if __name__ == '__main__':
    parser = argopt(__doc__).parse_args()
    tagged_file_path = parser.docs_folder
    n_jobs = parser.cores
    annotated_txt_paths = []

    if not os.path.isdir(tagged_file_path) and os.path.isfile(tagged_file_path):
        annotated_txt_paths = [tagged_file_path]
    else:
        annotated_txt_paths = glob.glob(tagged_file_path + "**/*ann.txt", recursive=True)
    if not annotated_txt_paths:
        raise Exception("Path not found")

    # annotated_txt_paths = ["/data/conseil_etat/hand_annotated/testing/C4121_ann.txt"]
    if n_jobs < 2:
        job_output = []
        for annotated_txt_path in tqdm(annotated_txt_paths):
            job_output.append(run(annotated_txt_path))
    else:
        job_output = Parallel(n_jobs=n_jobs)(
            delayed(run)(annotated_txt_path) for annotated_txt_path in tqdm(annotated_txt_paths))

    # Get correctly processed paths
    processed_fine = [f"{c[1]}\n" for c in job_output if c[0] == 1]
    # with open("./logs/success_correct_normal2txt.txt", "w") as filo:
    #     filo.writelines(processed_fine)

    tqdm.write(
        f"{len(processed_fine)} normal TXT files were treated and saved as CoNLL. "
        f"{len(job_output) - len(processed_fine)} files had some error.")
    logging.info(
        f"{len(processed_fine)} normal TXT files were treated and saved as CoNLL. "
        f"{len(job_output) - len(processed_fine)} files "
        f"had some error.")
