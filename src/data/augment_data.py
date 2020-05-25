'''
Script to augment the training dataset

Usage:
    augment_data.py <training_data_conll> [options]

Arguments:
    <training_data_conll>                     A required path parameter
    --cores=<n> CORES       Number of cores to use [default: 1:int]
'''
import logging
from glob import glob
from pathlib import Path

from argopt import argopt
from joblib import Parallel, delayed
from tqdm import tqdm
import pandas as pd
import numpy as np
import random

random.seed(42)
CASE_FUNCS = [str.lower, str.upper, str.title, str.capitalize,
              str.swapcase, str.casefold]


def run(doc_path):
    dataset_df = pd.read_csv(doc_path, delim_whitespace=True, skip_blank_lines=False,
                             names=["token", "tag"])

    df_sequences = split_sequences(dataset_df=dataset_df)
    # modify_title_case(df_sequences[0])
    # random_case_modification(df_sequences[0])
    remove_context(df_sequences[0])
    return 1


def remove_context(sequence_df: pd.DataFrame, tag=None, prob_remove=0.5):
    """
    Remove the previous word occurring before an entity
    Parameters
    ----------
    sequence_df : A dataframe containing a sequence
    tag: Only try to replace entities with this tag. If None, we treat them all
    prob_remove: Probability of actually removing the context word before each entity

    Returns
    -------

    """

    if tag:
        annotated_rows = sequence_df[sequence_df["tag"] == tag].index
    else:  # we treat all
        annotated_rows = sequence_df[sequence_df["tag"] != "O"].index
    non_annotated_rows = sequence_df[~sequence_df.index.isin(annotated_rows)].index
    to_modify_ids = [i - 1 for i in annotated_rows if random.random() < prob_remove]
    to_modify_ids = np.intersect1d(to_modify_ids, non_annotated_rows)  # only remove those non-entity words

    for i in to_modify_ids:
        sequence_df.drop(sequence_df.index[[i]])


def modify_title_case(sequence_df: pd.DataFrame):
    annotated_rows = sequence_df[sequence_df["tag"] != "O"].index
    sequence_df.loc[annotated_rows, "token"] = sequence_df.loc[annotated_rows, "token"].str.title()


def random_case_modification(sequence_df: pd.DataFrame, level="entity", prob_modif=0.5):
    if level == "entity":  # modify only the tokens of an entity
        annotated_rows = sequence_df[sequence_df["tag"] != "O"].index
    else:
        annotated_rows = sequence_df.index
    for i in annotated_rows:
        if random.random() <= prob_modif:
            chosen_case_modif = random.choice(CASE_FUNCS)
            sequence_df.loc[i, "token"] = chosen_case_modif(sequence_df.loc[i, "token"])


def split_sequences(dataset_df: pd.DataFrame):
    df_list = np.split(dataset_df, dataset_df[dataset_df.isnull().all(1)].index)
    df_list = [df.drop(df.index[[0]]) for df in df_list if not all(df["tag"].fillna("O").values == "O")]
    return df_list


def main(doc_files_path: Path, n_jobs: int):
    if not doc_files_path.exists():
        print(f"{doc_files_path.as_posix()} does not exist. Give an existing one!")
        exit(1)

    if not doc_files_path.is_dir() and doc_files_path.is_file():
        doc_paths = [doc_files_path]
    else:
        doc_paths = [Path(p) for p in glob(doc_files_path.as_posix() + "/**/*.txt", recursive=True)]
    if not doc_paths:
        raise Exception(f"Path {doc_paths} not found")

    if n_jobs < 2:
        job_output = []
        for doc_path in tqdm(doc_paths):
            tqdm.write(f"Treating file {doc_path}")
            job_output.append(run(doc_path))
    else:
        job_output = Parallel(n_jobs=n_jobs)(delayed(run)(doc_path) for doc_path in tqdm(doc_paths))

    logging.info(
        f"{sum(job_output)} DOC files were converted to TXT. {len(job_output) - sum(job_output)} files "
        f"had some error.")

    return doc_paths


if __name__ == '__main__':
    parser = argopt(__doc__).parse_args()
    doc_files_path = Path(parser.training_data_conll)
    n_jobs = parser.cores
    main(doc_files_path=doc_files_path, n_jobs=n_jobs)
