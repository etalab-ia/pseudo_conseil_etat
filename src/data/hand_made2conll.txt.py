'''Transforms input hand-annotated text files (with tags of type <LOCATION></LOCATION>) into CoNLL like text files
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
import xml.etree.ElementTree as ET
from collections import defaultdict

from joblib import Parallel, delayed

import numpy as np
from argopt import argopt
from tqdm import tqdm
import re

logging.basicConfig(filename='../logs/hand_annotated2conll.log', filemode='w',
                    format="%(asctime)s:%(levelname)s:\t%(message)s", level=logging.INFO)

TAGS_DICT = {"Nom": "PER_NOM", "Prenom": "PER_PRENOM", "Adresse": "LOC", "O": "O"}


def is_txt_xml_position_aligned(per_line_tagged_tokens, text_lines, replacement_line=0):
    """
    Check if the lines of the txt and the line of the xml are aligned
    :param per_line_tagged_tokens:
    :param text_lines:
    :param replacement_line:
    :return:
    """

    line_tagged_tokens = [info["token"] for info in per_line_tagged_tokens[replacement_line]]
    replacement_line = replacement_line - 1  # zero indexed text lines
    replacement_line = min(len(text_lines) - 1, replacement_line - 1)
    txt_line = text_lines[replacement_line]
    tagged_tokens_matched = []
    for idx, tagged_token in enumerate(line_tagged_tokens):
        matched = list(re.finditer(r"{}".format(tagged_token), txt_line))
        if matched:
            tagged_tokens_matched.extend([(t.group(),) + t.span() for t in matched])
        else:
            return False

    # check that the matched tagged tokens are in the same order as the original tagged tokens
    sorted_matched = sorted(tagged_tokens_matched, key=lambda x: x[1])
    if not all(line_tagged_tokens[i] == sorted_matched[i][0] for i in range(len(line_tagged_tokens))):
        # We found the same number of tagged tokens but they are not the same :(. Next line..."
        return False

    return True


def align_txt_xml(per_line_tagged_tokens, text_lines, xml_path):
    """
    Check that the text and xml informations are aligned. Align them if not aligned
    :param per_line_tagged_tokens:
    :param text_lines:
    :return:
    """
    per_line_tagged_tokens = text_xml_alignment(per_line_tagged_tokens=per_line_tagged_tokens,
                                                text_lines=text_lines,
                                                xml_path=xml_path)

    return per_line_tagged_tokens


def text_xml_alignment(per_line_tagged_tokens, text_lines, xml_path):
    """
    Align text with xml
    :param per_line_tagged_tokens:
    :param text_lines:
    :return:
    """
    found_index = -1
    per_line_tagged_tokens_copy = {}
    for line_nb_xml, replacements in per_line_tagged_tokens.items():
        line_tagged_tokens = [info["token"] for info in replacements]

        # We iterate over all the lines of the text file to find the line with the tagged tokens of this replacement_position
        for line_index, line in enumerate(text_lines[found_index + 1:], start=found_index + 1):
            if not line:
                continue
            tagged_tokens_matched = []

            for idx, tagged_token in enumerate(line_tagged_tokens):
                matched = list(re.finditer(r"{}".format(re.escape(tagged_token)), line))
                if matched:
                    tagged_tokens_matched.extend([(t.group(),) + t.span() for t in matched])
                else:
                    break

            if not tagged_tokens_matched:
                continue
            tagged_tokens_matched = list(set(tagged_tokens_matched))
            # check that the matched tagged tokens are in the same order as the original tagged tokens
            if len(line_tagged_tokens) == len(tagged_tokens_matched):
                # sort tagged_tokens_matched
                sorted_matched = sorted(tagged_tokens_matched, key=lambda x: x[1])
                if not all(line_tagged_tokens[i] == sorted_matched[i][0] for i in range(len(line_tagged_tokens))):
                    # We found the same number of tagged tokens but they are not the same :(. Next line..."
                    continue
                else:
                    # We have the same number of tagged tokens and the same tagged tokens :)"
                    found_index = line_index
                    per_line_tagged_tokens_copy[found_index] = replacements
                    break
    if not len(per_line_tagged_tokens_copy) == len(per_line_tagged_tokens):
        logging.error(f"Could not perfectly align this file {xml_path}. "
                      f"Missing {abs(per_line_tagged_tokens - per_line_tagged_tokens_copy)} annotations")
    return per_line_tagged_tokens_copy


def get_target_file(root):
    try:
        for ch1 in root[0].findall('FichierResultat'):
            for ch2 in ch1:
                if ch2.tag == "Chemin" and "anon_compl" in ch2.text:
                    return ch1
    except:
        tqdm.write(
            "Could not parse XML file. Check it follows the same scheme.")


def get_tagged_tokens(file_info):
    tagged_tokens_dict = []
    tagged_tokens = file_info.findall("MotsAnonymises")[0].findall("MotAnonymise")
    for tok in tagged_tokens:
        token = tok.find("Mots").text
        replacement = tok.find("RemplacesAvec").text
        tipo = tok.find("Type").text
        position = tok.find("Position").text
        line = tok.find("Ligne").text
        tagged_tokens_dict.append({"token": token, "replacement": replacement, "tipo": tipo, "position": position,
                                   "line": line})
    return tagged_tokens_dict


def get_per_line_replacements(tagged_tokens, sort_lines=False, zero_index=False, fix_first_index=False):
    per_line_dict = defaultdict(list)
    for tag_tok in tagged_tokens:
        if zero_index:
            key = int(tag_tok["line"]) - 1
        else:
            key = int(tag_tok["line"])

        per_line_dict[key].append(tag_tok)
    # sort line replacements by position
    for line_nb, replacements in per_line_dict.items():
        sorted_replacements = sorted(replacements, key=lambda x: int(x["position"]))
        per_line_dict[line_nb] = sorted_replacements

    if fix_first_index:
        first_index = list(sorted(per_line_dict.keys()))[0]
        per_line_dict[first_index + 1] = per_line_dict[first_index]
        per_line_dict.pop(first_index)
    if sort_lines:
        per_line_dict = dict(sorted(per_line_dict.items(), key=lambda x: x[0]))

    return per_line_dict


def tokenize(phrase):
    # TODO: Tokenize with proper tokenizer
    tokens = re.split("[\s,.]+", phrase)
    tokens = [t for t in tokens if t]
    return tokens


def find_index(list_tokens, start_position, replacement_token):
    if not list_tokens:
        return -1
    for i, tok in enumerate(list_tokens[start_position:], start_position):
        if replacement_token in tok:
            return i
    tqdm.write(f"Could not find {replacement_token} in phrase tokens {list_tokens}")
    return -1


def get_line_tags(line_replacement, line_nb, lines):
    tokens = tokenize(lines[line_nb])
    tags = np.array(["O"] * len(tokens), dtype=np.object)

    if not line_replacement:
        return tokens, tags.tolist()

    start_position_id = -1
    for repl in line_replacement:
        replacement_tokenenized = tokenize(repl["token"])

        start_position = find_index(tokens, start_position_id + 1, replacement_tokenenized[0])
        end_position = find_index(tokens, start_position, replacement_tokenenized[-1])
        # line_nb_offset = line_nb
        if start_position < 0 and end_position < 0:  # We didn't find one or both of the tokes to replace !
            logging.error("Did not find a token to replace. This is not good. It means that the xml is not aligned to "
                          "the text file!!")
            return None, None
        tipo = repl["tipo"]
        tags[start_position: end_position + 1] = tipo
        start_position_id = end_position
    return tokens, tags.tolist()


def load_annotation(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()
    file_info = get_target_file(root)
    tagged_tokens = get_tagged_tokens(file_info)
    per_line_tagged_token = get_per_line_replacements(tagged_tokens, sort_lines=True, zero_index=False,
                                                      fix_first_index=False)
    return per_line_tagged_token


def load_decision(decision_path, encoding='utf-8-sig'):
    with open(decision_path, encoding=encoding) as filo:
        lines = [l.strip() for l in filo.readlines()]
    # TODO: Segment in sentences with proper segmentator

    return lines


def get_decision_tokens_tags(lines, annotations_per_line, file_treated):
    all_tags = []
    all_tokens = []
    for line_nb, line in enumerate(lines):
        if not line:
            continue
        line_tagged_tokens = annotations_per_line.get(line_nb, [])

        line_tokens, line_tagged_tags = get_line_tags(line_tagged_tokens, line_nb, lines)
        if not line_tokens or not line_tagged_tags:
            return None, None
        all_tags.append(line_tagged_tags)
        all_tokens.append(line_tokens)
    if len(all_tags) != len(all_tokens):
        logging.error(f"Different number of tags and tokens in file {file_treated}")

    return all_tags, all_tokens


def tags_to_bio(all_tags):
    new_tags = []
    for seq in all_tags:
        new_tags.append([TAGS_DICT[t] for t in seq])
    # new_tags.insert(0, ["NOM", "O"])
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


def run(annotation_xml_path):
    logging.info(f"Treating file {annotation_xml_path}")
    tqdm.write(f"Treating file {annotation_xml_path}")


    return 1, annotation_xml_path


if __name__ == '__main__':
    parser = argopt(__doc__).parse_args()
    tagged_file_path = parser.docs_folder
    n_jobs = parser.cores

    annotated_txt_paths = glob.glob(tagged_file_path + "**/*.txt", recursive=True)
    if n_jobs < 2:
        job_output = []
        for annotation_xml_path in tqdm(annotated_txt_paths):
            job_output.append(run(annotation_xml_path))
    else:
        job_output = Parallel(n_jobs=n_jobs)(
            delayed(run)(annotation_xml_path) for annotation_xml_path in tqdm(annotated_txt_paths))

    # Get correctly processed paths
    processed_fine = [f"{c[1]}\n" for c in job_output if c[0] == 1]
    with open("../logs/correct_xmls.txt", "w") as filo:
        filo.writelines(processed_fine)

    tqdm.write(
        f"{len(processed_fine)} XML/DOC files were treated and saved as CoNLL. {len(job_output) - len(processed_fine)} files "
        f"had some error.")
    logging.info(
        f"{len(processed_fine)} XML/DOC files were treated and saved as CoNLL. {len(job_output) - len(processed_fine)} files "
        f"had some error.")
