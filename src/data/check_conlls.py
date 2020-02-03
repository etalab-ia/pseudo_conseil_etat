'''
Check that the CoNLL files have no errors regarding their original XMLs versions
Usage:
    check_conlls.py <files_folder>  [options]

Arguments:
    <files_folder>                     Folder path with the CoNLL annotation files to check
    --cores=<n> CORES                  Number of cores to use [default: 1:int]
'''
import glob
import logging
import os
import pandas as pd

from argopt import argopt
from joblib import Parallel, delayed
from tqdm import tqdm

from src.data.xml2conll import load_annotation

logger = logging.getLogger('check_conlls')
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler(f'./logs/check_conlls.log', mode='w')
fh.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s:%(levelname)s:%(funcName)s:\t%(message)s")
fh.setFormatter(formatter)
logger.addHandler(fh)


def add_brackets(stringo):
    dict_brackets = {'-LRB-':'(', '-RRB-':'(', '-LSB-':'(', '-RSB-':'(', '-LCB-':'(', '-RCB-':'('}
    if stringo in dict_brackets:
        return dict_brackets[stringo]
    else:
        return stringo


def run(annotation_conll_path):
    logger.info(f"Checking file {annotation_conll_path}")
    annotation_xml_path = annotation_conll_path[:-10] + ".xml"
    if not os.path.exists(annotation_conll_path):
        logger.debug(f"{annotation_conll_path} was not found!!")
        return 0, annotation_conll_path
    if not os.path.exists(annotation_xml_path):
        logger.debug(f"{annotation_xml_path} was not found!!")
        return 0, annotation_conll_path

    try:
        # 1. Load annotations from xml, create line replacements dict
        per_line_tagged_entity, outcome = load_annotation(annotation_xml_path)

        if outcome:
            logger.error(f"Could not load annotations for file {annotation_xml_path}")
            return 0, annotation_conll_path
        xml_annotations = [entity for entities in per_line_tagged_entity.values() for entity in entities]

        # 2. Load CoNLL file annotations
        conll_df = pd.read_csv(annotation_conll_path, names=['token', 'tag'], sep="\t", header=None)
        conll_df = conll_df[conll_df['tag'] != 'O']
        list_annotations = list(map(list, conll_df.values))
        conll_annotations = recreate_sequences(list_annotations)

        # 3. Check they are the same!
        # First check we have the same number of annotations
        if len(xml_annotations) != len(conll_annotations):
            logger.error(f"Different number of annotations in {annotation_conll_path} and in {annotation_xml_path}")
            return 0, annotation_conll_path
        # Check they are the same entities (or similar)
        token_in_xml_annotation = []
        for seq_id in range(len(conll_annotations)):
            for token_id, (token, tag) in enumerate(conll_annotations[seq_id]):
                if add_brackets(token) not in xml_annotations[seq_id]['token']:
                    token_in_xml_annotation.append(token)

        if token_in_xml_annotation:
            logger.error(f"There are some tokens that are not in the XML file {annotation_xml_path}")
            return 0, annotation_conll_path

        return 1, annotation_conll_path

    except Exception as e:
        logger.error(str(e))
        return 0, annotation_conll_path


def recreate_sequences(list_tokens_tags: list):
    """
    From a list of tuples (token, tag) where tag != "O", recreate a list of lists where each inner list is a
    tagged sequence
    :param list_tokens_tags: List of tuples [(token1, tag1), (token2, tag2), ...]
    :return:
    """
    sequence = []
    final_sequences = []
    for token, tag in list_tokens_tags:
        if "I" in tag:
            sequence.append((token, tag))
        if "B" in tag:
            if sequence:
                final_sequences.append(sequence)

            sequence = [(token, tag)]
            # inside_sequence = True
        # else:
        # inside_sequence = True
        # sequence.append((token, tag))
    final_sequences.append(sequence)
    return final_sequences

    pass


if __name__ == '__main__':
    parser = argopt(__doc__).parse_args()
    conll_folder_path = parser.files_folder
    n_jobs = parser.cores

    # annotation_conll_paths = glob.glob(conll_folder_path + "**/*CoNLL.txt", recursive=True)
    annotation_conll_paths = ["/data/conseil_etat/text_alignment_check/15DA00426_CoNLL.txt"]
    if n_jobs < 2:
        job_output = []
        for conll_file_path in tqdm(annotation_conll_paths):
            job_output.append(run(conll_file_path))
    else:
        job_output = Parallel(n_jobs=n_jobs)(
            delayed(run)(conll_file_path) for conll_file_path in tqdm(annotation_conll_paths))

    processed_fine = [f"{c[1]}\n" for c in job_output if c[0] == 1]


    logger.info(
        f"{len(processed_fine)} CoNLL files were compared and are good. "
        f"{len(job_output) - len(processed_fine)} files do not agree with its corresponding XML.")

