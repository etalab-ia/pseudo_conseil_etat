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
from tqdm import tqdm

from src.utils.tokenizer import moses_tokenize, moses_detokenize

logging.basicConfig(filename='./logs/hand_annotated2conll.log', filemode='w',
                    format="%(asctime)s:%(levelname)s:\t%(message)s", level=logging.INFO)

TAGS_DICT = {"Nom": "PER_NOM", "Prenom": "PER_PRENOM", "Adresse": "LOC", "O": "O"}


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
    all_tags = []
    all_tokens = []
    for line in text_lines:
        seq_list = []
        if not line:
            continue
        line = replace_tags(line)
        line_tokens = moses_tokenize(line)
        tagged_sequence = False
        tag = "O"
        for tok in line_tokens:
            matched_begin = re.match(r'BEGINTAG(PRENOM|NOM|ADRESSE)', tok)
            matched_end = re.match(r'ENDTAG(PRENOM|NOM|ADRESSE)', tok)
            if matched_end:
                tag = "O"
                tagged_sequence = False
                continue
            if tagged_sequence:
                seq_list.append((tok, tag))
                continue
            if matched_begin:
                tag = matched_begin.groups()[0].capitalize()
                tagged_sequence = True
                continue

            if tok == "&apos;" and seq_list[-1][0] == "d":
                seq_list[-1][0] = "d"+tok
                continue

            seq_list.append([tok, tag])
        tokens, tags = list(zip(*seq_list))
        all_tokens.append(tokens)
        all_tags.append(tags)

    return all_tokens, all_tags


def run(annotated_txt_path):
    logging.info(f"Treating file {annotated_txt_path}")
    tqdm.write(f"Treating file {annotated_txt_path}")
    decision_folder = os.path.dirname(annotated_txt_path)

    # TODO Here, change to a sentence segmentation text_lines
    with open(annotated_txt_path, encoding='utf-8-sig') as filo:
        text_lines = [l.strip() for l in filo.readlines()]

    all_tokens, all_tags = get_word_per_line(text_lines)
    all_tags = tags_to_bio(all_tags)

    with open(os.path.join(decision_folder, os.path.basename(annotated_txt_path)[:-4] + "_hand_CoNLL.txt"),
              "w") as conll:
        conll.write(f"-DOCSTART-\tO\n\n")
        for tokens, tags in zip(all_tokens, all_tags):
            for tok, tag in zip(tokens, tags):
                conll.write(f"{moses_detokenize([tok])}\t{str(tag)}\n")
                logging.debug(f"{tok}\t{str(tag)}")
            conll.write("\n")
            logging.debug("\n")

    return 1, annotated_txt_path


if __name__ == '__main__':
    parser = argopt(__doc__).parse_args()
    tagged_file_path = parser.docs_folder
    n_jobs = parser.cores

    if os.path.isfile(tagged_file_path):
        annotated_txt_paths = [tagged_file_path]
    else:
        annotated_txt_paths = glob.glob(tagged_file_path + "/**/*ann.txt", recursive=True)
    if n_jobs < 2:
        job_output = []
        for annotated_txt_path in tqdm(annotated_txt_paths):
            job_output.append(run(annotated_txt_path))
    else:
        job_output = Parallel(n_jobs=n_jobs)(
            delayed(run)(annotated_txt_path) for annotated_txt_path in tqdm(annotated_txt_paths))

    # Get correctly processed paths
    processed_fine = [f"{c[1]}\n" for c in job_output if c[0] == 1]
    with open("./logs/correct_human_annotated_txts.txt", "w") as filo:
        filo.writelines(processed_fine)

    tqdm.write(
        f"{len(processed_fine)} hand annotated TXT files were treated and saved as CoNLL. "
        f"{len(job_output) - len(processed_fine)} files had some error.")
    logging.info(
        f"{len(processed_fine)} hand annotated TXT files were treated and saved as CoNLL. "
        f"{len(job_output) - len(processed_fine)} files "
        f"had some error.")
