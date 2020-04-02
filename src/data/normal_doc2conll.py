'''
Transforms input hand-annotated text files (with tags of type <LOCATION></LOCATION>) into CoNLL like text files
that is, one word per line and the tag to the right.

Usage:
    hand_made2conll.py <docs_folder>  [options]

Arguments:
    <docs_folder>                       Folder path with the txt annotated files to transform to CoNLL
    --cores=<n> CORES                  Number of cores to use [default: 1:int]
'''
import glob
import logging
import os
import re

from argopt import argopt
from joblib import Parallel, delayed
from sacremoses import MosesTokenizer
from tqdm import tqdm

# logging.basicConfig(filename='./logs/normal_doc2conll.log', filemode='w',
#                     format="%(asctime)s:%(levelname)s:\t%(message)s", level=logging.INFO)

TAGS_DICT = {"Nom": "PER_NOM", "Prenom": "PER_PRENOM", "Adresse": "LOC", "O": "O"}

mt = MosesTokenizer(lang="fr")


def moses_tokenize(phrase):
    tokens = mt.penn_tokenize(phrase)
    return tokens


def tokenize(phrase):
    # TODO: Tokenize with proper tokenizer
    tokens = re.split("[\s,.]+", phrase)
    tokens = [t for t in tokens if t]
    return tokens


def tags_to_bio(all_tags):
    new_tags = []
    for seq in all_tags:
        new_tags.append([TAGS_DICT[t] for t in seq])
    for seq in new_tags:
        for i, tag in enumerate(seq):
            if tag == "O":
                continue
            if i > 1:
                if tag in seq[i - 1]:
                    seq[i] = f"I-{tag}"
                    continue
            seq[i] = f"B-{tag}"
    return new_tags


def replace_tags(line):
    line = re.sub(r"<(PRENOM|NOM|ADRESSE)>", r" BEGINTAG\g<1> ", line)
    line = re.sub(r"</(PRENOM|NOM|ADRESSE)>", r" ENDTAG\g<1> ", line)
    return line


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

    with open(os.path.join(decision_folder, os.path.basename(annotated_txt_path)[:-4] + "_TestCoNLL.txt"),
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
