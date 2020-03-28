'''Transforms the XMLs of the conseil d'etat annotated decisions into the CoNLL format. Each XML should have
an accompanying txt decision file. Very important module !

Usage:
    xml2conll.py <docs_folder>  [options]

Arguments:
    <docs_folder>                       Folder path with the XML annotation file to transform to CoNLL
    --cores=<n> CORES                  Number of cores to use [default: 1:int]
'''
import glob
import logging
import os
import re
import xml.etree.ElementTree as ET
from collections import defaultdict, Counter

import numpy as np
from argopt import argopt
from joblib import Parallel, delayed
from sacremoses import MosesTokenizer
from tqdm import tqdm

try:
    logger = logging.getLogger('xml2conll')
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(f'./logs/xml2conll.log', mode='w')
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s:%(levelname)s:%(funcName)s:\t%(message)s")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
except:
    print("No logger available.")

TAGS_DICT = {"Nom": "PER_NOM", "Prenom": "PER_PRENOM", "Adresse": "LOC", "O": "O"}


def remove_nested_entities(matched_entities):
    """
    Find and remove matched entities that overlap with a larger one. I.e.: detect and remove entities contained
    in others. Ex. (Idris, 12, 17) is overlapped in (Faris Idris, 6, 17).
    :param matched_entities:
    :return: clean_matched_entities: without nested entities
    """
    clean_matched_entities = []
    len_matched_entities = len(matched_entities)
    for i in range(len_matched_entities):
        is_nested = False
        _, i_start_pos, i_end_pos = matched_entities[i]
        for j in range(len_matched_entities):
            _, j_start_pos, j_end_pos = matched_entities[j]
            if i_start_pos == j_start_pos and i_end_pos == j_end_pos:
                continue
            if i_start_pos >= j_start_pos and i_end_pos <= j_end_pos:
                # print(f"{matched_entities[i]} is nested :(")
                is_nested = True
                break
        if not is_nested:
            clean_matched_entities.append(matched_entities[i])

    return clean_matched_entities


def text_xml_alignment(per_line_tagged_tokens, text_lines, xml_path):
    """
    Align text with xml annotations. Returns a dict {line_1: [{token:'', replacement:'', tipo:'', line:''}, {} ],
                                                    line_2: [{token:'', replacement:'', tipo:'', line:''}, {} ],
                                                    ...}
    with the correct line numbers as found in the corresponding txt file.
    :param per_line_tagged_tokens:
    :param text_lines:
    :return:
    """
    found_index = -1
    per_line_tagged_tokens_copy = {}
    for line_nb_xml, replacements in per_line_tagged_tokens.items():
        line_tagged_tokens = [info["token"] for info in replacements]

        # We iterate over all the lines of the text file to find the line with the tagged tokens of this
        # replacement_position
        for line_index, line in enumerate(text_lines[found_index + 1:], start=found_index + 1):
            if not line:
                continue
            line_tagged_tokens_matched = []

            for idx, tagged_token in enumerate(line_tagged_tokens):
                matched = list(re.finditer(r"\b{}(?!\w)".format(re.escape(tagged_token)), line))
                if matched:
                    line_tagged_tokens_matched.extend([(t.group(),) + t.span() for t in matched])
                else:
                    break

            if not line_tagged_tokens_matched:
                continue
            line_tagged_tokens_matched_sorted = list(sorted(set(line_tagged_tokens_matched), key=lambda x: x[1]))
            line_tagged_tokens_matched_sorted = remove_nested_entities(line_tagged_tokens_matched_sorted)
            # check that the matched tagged tokens are in the same order as the original tagged tokens
            if len(line_tagged_tokens) == len(line_tagged_tokens_matched_sorted):
                if not all(line_tagged_tokens[i] == line_tagged_tokens_matched_sorted[i][0]
                           for i in range(len(line_tagged_tokens))):
                    # We found the same number of tagged tokens but they are not the same :( Next line..."
                    continue
                else:
                    # We have the same number of tagged tokens and the same tagged tokens :)"
                    found_index = line_index
                    per_line_tagged_tokens_copy[found_index] = replacements
                    break
    if len(per_line_tagged_tokens_copy) != len(per_line_tagged_tokens):
        return None
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


def get_tagged_sequences(file_info):
    tagged_sequences_dict = []
    tagged_sequences = file_info.findall("MotsAnonymises")[0].findall("MotAnonymise")
    for tagged_sequence in tagged_sequences:
        text = tagged_sequence.find("Mots").text
        replacement = tagged_sequence.find("RemplacesAvec").text
        tipo = tagged_sequence.find("Type").text
        position = tagged_sequence.find("Position").text
        line = tagged_sequence.find("Ligne").text
        tagged_sequences_dict.append({"token": text, "replacement": replacement, "tipo": tipo, "position": position,
                                      "line": line})
    return tagged_sequences_dict


def get_per_line_replacements(tagged_sequences, sort_lines=False, zero_index=False):
    per_line_dict = defaultdict(list)
    for tag_tok in tagged_sequences:
        line = int(tag_tok["line"])
        per_line_dict[line].append(tag_tok)
    # sort line replacements by position
    for line_nb, replacements in per_line_dict.items():
        sorted_replacements = sorted(replacements, key=lambda x: int(x["position"]))
        per_line_dict[line_nb] = sorted_replacements

    if sort_lines:
        per_line_dict = dict(sorted(per_line_dict.items(), key=lambda x: x[0]))

    return per_line_dict


mt = MosesTokenizer(lang="fr")


def moses_tokenize(phrase):
    tokens = mt.tokenize(phrase)
    return tokens


def tokenize(phrase):
    # TODO: Tokenize with proper tokenizer
    tokens = re.split("[\s,.]+", phrase)
    tokens = [t for t in tokens if t]
    return tokens


def find_sub_list(sl, l, start_position):
    """
    Super copy&paste from SO: https://stackoverflow.com/questions/17870544/find-starting-and-ending-indices-of-sublist-in-list
    :param sl:
    :param l:
    :return:
    """

    def check_if_sublist_in_list(sl, l):
        if len(sl) != len(l):
            return False
        for index in range(len(sl)):
            if not sl[index] in l[index]:
                return False
        return True

    sll = len(sl)
    for ind in (i for i, e in enumerate(l[start_position:], start_position) if sl[0] in e):
        # if l[ind:ind+sll]==sl:
        if check_if_sublist_in_list(sl, l[ind:ind + sll]):
            return ind, ind + sll - 1
    return None


def find_index(list_tokens, start_position, replacement_token):
    if not list_tokens:
        return -1, []
    for i, tok in enumerate(list_tokens[start_position:], start_position):
        if replacement_token in tok:
            return i, []
    return -1, replacement_token


def get_line_tags(line_replacements, line_nb, lines, file_treated):
    # TODO: This whole method has to be rethought. It is waay too hacky now :(
    tokens = moses_tokenize(lines[line_nb])
    tags = np.array(["O"] * len(tokens), dtype=np.object)

    if not line_replacements:
        return tokens, tags.tolist()

    start_position_id = 0
    sequences_tagged = 0
    for line_id, replacement in enumerate(line_replacements):
        replacement_tokenenized = moses_tokenize(replacement["token"])

        # start_position, debug_list = find_index(tokens, start_position_id, replacement_tokenenized[0])
        # end_position = start_position + len(replacement_tokenenized)

        start_position, end_position = find_sub_list(replacement_tokenenized, tokens, start_position_id)
        end_position += 1

        if start_position < 0:
            logger.error(f"Could not find {debug_list} in phrase tokens {tokens[start_position_id:]}")
            return None, None

        else:
            tipos = TAGS_DICT[replacement['tipo']]
            # if len(replacement_tokenenized) > 1:
            tipos = [f"B-{tipos}" if f == 0 else f"I-{tipos}" for f in range(len(replacement_tokenenized))]
            if len(tags[start_position: end_position]) != len(tipos):
                pass
                print()
            tags[start_position: end_position] = tipos
            start_position_id = end_position
            # oh god horrible hack to not increment the index if we have multiple tokens nested in a xml annotation :(
            if start_position == end_position - 1 and len(replacement_tokenenized[0]) < len(tokens[start_position]):
                start_position_id = start_position_id - 1

            sequences_tagged += 1
    if sum(1 for t in tags if "B-" in t) != len(line_replacements):
        logger.warning(f"We had a different number of sequences in the xml than those created for this line. " +
                       f"XML:{[(t['token'], t['tipo']) for t in line_replacements if t['tipo'] != 'O']}. " +
                       f"Created:{[(tok, tag) for tok, tag in zip(tokens, tags) if tag != 'O']}")

    return tokens, tags.tolist()


def load_annotation(file_path, count_differing_annotations=False):
    error = ""
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
        file_info = get_target_file(root)
        tagged_sequences = get_tagged_sequences(file_info)
        per_line_tagged_sequence = get_per_line_replacements(tagged_sequences, sort_lines=True, zero_index=False)

        if not per_line_tagged_sequence:
            error = f"There were no annotations found in {file_path}."
            return None, error

        if count_differing_annotations:
            # There are xml files that have different annotations for the same entity. We count them here.
            # We could fix it by using the majority tag (if any) but there may be cases where the same word may
            # represent a first AND a last name. So we will assume the annotator is right :/

            token_tags_dict = defaultdict(set)
            for line_nb, entities in per_line_tagged_sequence.items():
                for entity in entities:
                    token_tags_dict[entity['token']].update([entity['tipo']])

            different_tags = [(token, tag) for token, tag in token_tags_dict.items() if len(tag) > 1]

            if different_tags:
                logger.warning(f"File {file_path} has different tags for the same entity. {different_tags}")

        return per_line_tagged_sequence, error
    except Exception as e:
        error = f"Invalid XML file {file_path}: " + str(e)
        return None, error


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

        line_tokens, line_tagged_tags = get_line_tags(line_tagged_tokens, line_nb, lines, file_treated)
        if not line_tokens or not line_tagged_tags:
            return None, None
        all_tags.append(line_tagged_tags)
        all_tokens.append(line_tokens)
    if len(all_tags) != len(all_tokens):
        logger.error(f"\tDifferent number of tags and tokens in file {file_treated}")

    return all_tags, all_tokens


def tags_to_bio(all_tags):
    new_tags = []
    # for seq in all_tags:
    # new_tags.append([TAGS_DICT[t] for t in seq])
    for seq in all_tags:
        for i, tag in enumerate(seq):

            if tag == "O":
                continue
            new_tag = TAGS_DICT[tag.replace("_SINGLE_", "")]
            if i > 1:
                if new_tag in seq[i - 1] and "_SINGLE_" not in tag:
                    seq[i] = f"I-{new_tag}"
                    continue
            seq[i] = f"B-{new_tag}"
    return new_tags


#
# def postprocess_sequence(all_tags, all_tokens):
#     """
#     Applies postropcessing techniques to the sequences extracted from a text file (either created from xml+txt, from
#     hand annodated txt, or from a plain txt file)
#     :param all_tags: List of lists (a list per sequence) with the tags of each token of the sequence
#     :param all_tokens: List of lists (a list per sequence) with the tokens of each sequence
#     :return: all_tags
#     :return: all_tokens
#     """
#     def partition(alist, indices):
#         return [alist[i:j] for i, j in zip([0]+indices, indices+[None])]
#
#     def break_long_sequences(sequence_tokens, sequence_tags, max_sequence_len=200, split_char=";"):
#         """
#         Break a sequence if it is too long according to the threshold
#         :param sequence_tokens: List of sequence tokens
#         :param sequence_tags: List of sequence tags
#         :param max_sequence_len: Threshold of max number of characters
#         :return:
#         """
#         new_sequences_tokens = []
#         new_sequences_tags = []
#         if len(sequence_tokens) > max_sequence_len:
#             get_split_char_indices = [i for i, val in enumerate(sequence_tokens) if val == split_char]
#             temps_sequences = partition(sequence_tokens, get_split_char_indices)
#             # if len(temps_sequences)  1:
#
#
#             for token, tag in temps_sequences:
#         else:
#             return sequence_tokens, sequence_tags
#
#
#         pass
#
#     pass

def find_reason_alignment_fail(per_line_tagged_entity: dict, text_lines: list):
    """
    Determine the cause of the alignment fail. Until now there are 3 possible causes found:
    1. There is a mysterious line 0 in the xml which seems to be an error. This is the most pervasive error.
    2. The xml contains misspelled entities that are not found on the txt.
    3. The xml contains entities that do not exist in the txt. (the xml comprehends a larger file)

    Error 2 and 3 are detected similarly, if we do not find an entity in the text but it is in the xml, there is one of
    these two errors. To specifically detect error 2, we would need to find some edit distance between all the entities
    and all the words of the text which seems unnecessary and quite expensive...
    Error 1 is simply detecting whether there is a key=0 in the per_per_line_tagged_entity dict.
    :param per_line_tagged_entity:
    :return:reason
    """

    def remove_nested_xml_entities(per_line_entities):
        """
        Find xml entities that overlap and remove them
        :param per_line_entities: {line_nb:[{token, replacement, tipo, position, line}, {}]}
        :return: has_nested_entities: bool telling whether there are nested entities or not
        :return: clean_matched_entities: without nested entities
        """

        clean_per_line_entities = {}
        seen_nested = []
        nested_entities = []
        for line_nb, list_annotations in per_line_entities.items():
            clean_matched_entities = []
            len_list_annotations = len(list_annotations)

            for i in range(len_list_annotations):
                token_i = list_annotations[i]['token']
                position_i = int(list_annotations[i]['position'])
                is_nested = False
                for j in range(len_list_annotations):
                    token_j = list_annotations[j]['token']
                    position_j = int(list_annotations[j]['position'])
                    if token_j == token_i and position_i == position_j:
                        continue
                    if token_i in token_j:
                        if position_i >= position_j and position_i <= position_j + len(token_j):
                            # print(f"{matched_entities[i]} is nested :(")
                            is_nested = True
                            nested_entities.append(list_annotations[i])
                            break
                seen_nested.append(is_nested)
                if not is_nested:
                    clean_matched_entities.append(list_annotations[i])
            clean_per_line_entities[line_nb] = clean_matched_entities
        return clean_per_line_entities, any(seen_nested), nested_entities

    text = " ".join(text_lines)
    reason = ("-1", "Error not contemplated. You should investigate!")
    all_entities = [token['token'] for list_tokens in per_line_tagged_entity.values() for token in list_tokens]
    if 0 in per_line_tagged_entity:
        reason = ("1", "There is a zero line in the XML file.")
        return reason

    all_entities_found = True
    for entity in list(set(all_entities)):
        if entity not in text:
            all_entities_found = False
            break
    if not all_entities_found:
        reason = ("2/3", f"Entity not found in TXT file. {entity} was not found.")
        return reason
    # check if we have entities not annotated in the XML. Count the number of occurrences of each entity and
    # compare it to the number of entities in the TXT file.
    count_entities = Counter(all_entities)
    for entity, count in count_entities.items():
        found_in_text = len(re.findall(r"\b{}(?!\w)".format(re.escape(entity)), text))
        if found_in_text != count:
            reason = ("4", f"Missing an instance of entity '{entity}' in the XML file. "
                           f"A name/address was not properly pseudonymized.")
            return reason

    clean_per_line_entities, seen_nested, nested_entities = remove_nested_xml_entities(
        per_line_entities=per_line_tagged_entity)
    if seen_nested:
        reason = ("6", f"Nested entities found (we have two annotations for the same entity): {str(nested_entities)}")
        return reason
    return reason


def run(annotation_xml_path):
    logger.info(f"Treating file {annotation_xml_path}")
    tqdm.write(f"Treating file {annotation_xml_path}")
    decision_folder = os.path.dirname(annotation_xml_path)
    decision_txt_path = annotation_xml_path[:-4] + ".txt"
    if not os.path.exists(decision_txt_path):
        logger.debug(f"{decision_txt_path} was not found!!")
        return 0, annotation_xml_path
    if not os.path.exists(annotation_xml_path):
        logger.debug(f"{annotation_xml_path} was not found!!")
        return 0, annotation_xml_path

    try:
        # Load annotation from xml, create line replacements dict
        per_line_tagged_entity, outcome = load_annotation(annotation_xml_path, count_differing_annotations=True)

        if not per_line_tagged_entity:
            logger.error(outcome)
            return 0, annotation_xml_path

        # open decision .txt file
        text_lines = load_decision(decision_txt_path)
        if not text_lines:
            logger.error(f"There were no lines found in {decision_txt_path}")
            return 0, annotation_xml_path

        # align xml and txt file
        per_line_tagged_entity_aligned = text_xml_alignment(per_line_tagged_entity, text_lines=text_lines,
                                                            xml_path=annotation_xml_path)
        if not per_line_tagged_entity_aligned:
            reason = find_reason_alignment_fail(per_line_tagged_entity, text_lines)
            logger.error(f"Could not perfectly align this file {annotation_xml_path}. Reason: {reason[1]}")
            return 0, annotation_xml_path

        # tokenize and get associated tags
        all_tags, all_tokens = get_decision_tokens_tags(lines=text_lines,
                                                        annotations_per_line=per_line_tagged_entity_aligned,
                                                        file_treated=annotation_xml_path)
        if not text_lines or not all_tokens:
            logger.error(f"Could not match the tokens and the tags in {decision_txt_path}")
            return 0, annotation_xml_path

        # all_tags = tags_to_bio(all_tags)

    except Exception as e:
        # logger.error(str(e), annotation_xml_path)
        logger.error(f"General exception : {str(e)}. File {annotation_xml_path}")
        # raise e
        return 0, annotation_xml_path
    # all_tags, all_tokens = postprocess_sequence(all_tags, all_tokens)

    # save tokens and tags in a file
    with open(os.path.join(decision_folder, os.path.basename(annotation_xml_path)[:-4] + "_CoNLL.txt"),
              "w") as conll:
        conll.write(f"-DOCSTART-\tO\n\n")
        for tokens, tags in zip(all_tokens, all_tags):
            for tok, tag in zip(tokens, tags):
                conll.write(f"{tok}\t{str(tag)}\n")
                # logger.debug(f"{tok}\t{str(tag)}")
            conll.write("\n")
            # logger.debug("\n")
    return 1, annotation_xml_path


if __name__ == '__main__':
    parser = argopt(__doc__).parse_args()
    tagged_folder_path = parser.docs_folder
    n_jobs = parser.cores

    # annotation_xml_paths = ["../notebooks/decisions/343837.xml"]
    # annotation_xml_paths = ["/data/conseil_etat/decisions/IN/DCA/CAA54/2013/20131128/13NC00060.xml"]
    annotation_xml_paths = glob.glob(tagged_folder_path + "/**/*.xml", recursive=True)
    if n_jobs < 2:
        job_output = []
        for annotation_xml_path in tqdm(annotation_xml_paths):
            job_output.append(run(annotation_xml_path))
    else:
        job_output = Parallel(n_jobs=n_jobs)(
            delayed(run)(annotation_xml_path) for annotation_xml_path in tqdm(annotation_xml_paths))

    # Get correctly processed paths
    processed_fine = [f"{c[1]}\n" for c in job_output if c[0] == 1]
    with open("./logs/correct_xmls.txt", "w") as filo:
        filo.writelines(processed_fine)

    logger.info(
        f"{len(processed_fine)} XML/DOC files were treated and saved as CoNLL. "
        f"{len(job_output) - len(processed_fine)} files had some error.")
