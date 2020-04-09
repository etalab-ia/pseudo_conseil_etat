'''Load the documents table (documents_table.csv) and extracts the XMLs of each conseil d'etat decisions in the file.
Then saves it to disk, right next to the decision .doc file.

We are mainly interested on the "corrigé" decisions, i.e., those that were corrected because due to a mistake of the
annotation system. So the decision we status code 5 are the ones we look for.

The status codes are:
0 non traite
1 ok
2 en doute
3 en erreur
4 en modification
5 corrige

Usage:
    table2xml.py <documents_table>  <documents_folder> [options]

Arguments:
    <documents_table>                  CSV containing the table from the conseil d'etat decisions
    <documents_folder>                 Root folder that contains the decisions. Inside this folder should be a IN/OUT folder
    --only_corriges                    Whether to extract only the corrected (corrige) decisions or not
    --cores=<n> CORES                  Number of cores to use [default: 1:int]
'''

import logging
import os
from joblib import Parallel, delayed
import pandas as pd

from argopt import argopt
# from tqdm import tqdm
from tqdm import tqdm

logger = logging.getLogger('table2xml')
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler(f'./logs/table2xml.log', mode='w')
fh.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s:%(levelname)s:\t%(message)s")
fh.setFormatter(formatter)
logger.addHandler(fh)

decision_table_cols = ['id', 'utilisateur_id_blocage', 'lot_id', 'statut', 'statut_precedent',
                       'chemin_source', 'dossier_destination', 'detail_statut',
                       'code_juridiction', 'detail_anonymisation', 'timestamp_blocage',
                       'timestamp_modification']

col2id = dict(zip(decision_table_cols, range(len(decision_table_cols))))


def save_decision_xml(df_row, documents_folder):
    row = df_row[1:]
    source_path = (row[col2id["chemin_source"]]).replace("\\", "/")  # Windows path -> Linux path cool hack
    # Deal with manuel paths
    if "manuel" in source_path:
        source_path = "/".join(source_path.split("/")[1:]).lower()
        decision_file_id = os.path.splitext(os.path.basename(source_path))[0]
        chosen_folder = documents_folder
    else:
        source_path = "/".join(source_path.split("/")[3:])  # Remove server name from path
        decision_file_id = os.path.splitext(os.path.basename(source_path))[0]
        chosen_folder = documents_folder

    annotations_xml = row[col2id["detail_anonymisation"]]
    doc_path = os.path.join(chosen_folder, source_path)
    # We check that the folder exists. No sense in having an annotation XML file without the decision .doc
    if not os.path.exists(doc_path) and not os.path.exists(doc_path[:-4] + ".txt"):
        logger.debug(f"File {doc_path} not found !!")
        return 0
    xml_path = os.path.join(chosen_folder, os.path.dirname(source_path), decision_file_id + ".xml")
    with open(xml_path, "w", encoding="utf-16") as xmlo:
        logger.info(f"Saving XML file as {xml_path}")
        xmlo.write(annotations_xml)
    return 1


def get_correct_line(df_decisions):
    """
    The passed df has repeated lines for the same file (same chemin_source).
    We take the most recent one.
    :param df_decisions: Dataframe of decisions
    :return: Dataframe without repeated lines (according to the chemin_source column)
    """

    return df_decisions.sort_values('timestamp_modification').drop_duplicates('chemin_source', keep='last')


if __name__ == '__main__':
    parser = argopt(__doc__).parse_args()
    decisions_csv = parser.documents_table
    only_corriges = parser.only_corriges
    documents_folder = parser.documents_folder
    n_jobs = parser.cores

    df_decisions = pd.read_csv(decisions_csv)
    if only_corriges:
        df_decisions = df_decisions[df_decisions.statut == 5]
    df_decisions.columns = df_decisions.columns.str.lower()
    df_decisions = get_correct_line(df_decisions)

    if n_jobs < 2:
        job_output = []
        for row in tqdm(list(df_decisions.itertuples())[:]):
            job_output.append(save_decision_xml(row, documents_folder))
    else:
        job_output = Parallel(n_jobs=10)(
            delayed(save_decision_xml)(annotation_xml_path, documents_folder=documents_folder)
            for annotation_xml_path in tqdm(list(df_decisions.itertuples())))
    logger.info(f"{sum(job_output)} XML files were extracted and saved. {len(job_output) - sum(job_output)} files "
                f"were not found or some error happened.")
