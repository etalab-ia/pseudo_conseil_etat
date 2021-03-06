'''
Script to augment the training dataset
Inspired on https://github.com/ELS-RD/anonymisation/ data augmentation techniques

Usage:
    augment_data.py <training_data_conll> [options]

Arguments:
    <training_data_conll>                     A required path parameter
    --cores=<n> CORES       Number of cores to use [default: 1:int]
'''
import logging
from glob import glob
from pathlib import Path
from typing import List

from argopt import argopt
from joblib import Parallel, delayed
from tqdm import tqdm
import pandas as pd
import numpy as np
import random

random.seed(42)
CASE_FUNCS = [str.lower, str.upper, str.title, str.capitalize, str.casefold]


def run(doc_path: Path):
    dataset_df = pd.read_csv(doc_path, delim_whitespace=True, skip_blank_lines=False,
                             names=["token", "tag"])

    df_sequences: List = split_sequences(dataset_df=dataset_df)
    augmented_df_sequences = []
    for i, (is_not_annotated, sequence_df) in tqdm(enumerate(df_sequences)):
        temp_df = sequence_df
        if not is_not_annotated:
            temp_df = modify_title_case(temp_df, tag="PER_PRENOM")
            temp_df = random_case_modification(temp_df, level="entity", prob_modif=0.3)
            temp_df = remove_context(temp_df, tag="PER_PRENOM", prob_remove=0.3)

        temp_df = temp_df.append(pd.DataFrame([["", ""]], columns=["token", "tag"]))
        augmented_df_sequences.append(temp_df)

    output_path = doc_path.as_posix()[:-4] + "_augmented.txt"
    final_pdf = pd.concat(augmented_df_sequences)
    save_conll_df(final_pdf, output_path)
    return 1


def save_conll_df(df: pd.DataFrame, path: Path):
    df.to_csv(path, header=None, index=None, sep="\t")


def remove_context(sequence_df: pd.DataFrame, tag="all", prob_remove=0.7):
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

    if tag != "all":
        annotated_rows = sequence_df[sequence_df["tag"].str.contains(tag)].index
    else:  # we treat all tags
        annotated_rows = sequence_df[sequence_df["tag"] != "O"].index
    non_annotated_rows = sequence_df[~sequence_df.index.isin(annotated_rows)].index
    to_modify_ids = [i - 1 for i in annotated_rows if random.random() < prob_remove]
    to_modify_ids = np.intersect1d(to_modify_ids, non_annotated_rows)  # only remove those non-entity words

    for i in to_modify_ids:
        sequence_df = sequence_df.drop(i)
    return sequence_df


def modify_title_case(sequence_df: pd.DataFrame, tag: str="all"):
    if tag != "all":
        annotated_rows = sequence_df[sequence_df["tag"].str.contains(tag)].index
    else:  # we treat all tags
        annotated_rows = sequence_df[sequence_df["tag"] != "O"].index

    sequence_df.loc[annotated_rows, "token"] = sequence_df.loc[annotated_rows, "token"].str.title()
    return sequence_df


def random_case_modification(sequence_df: pd.DataFrame, level="entity", prob_modif=0.7):
    if level == "entity":  # modify only the tokens of an entity
        annotated_rows = sequence_df[sequence_df["tag"] != "O"].index
    else:
        annotated_rows = sequence_df.index
    for i in annotated_rows:
        if random.random() <= prob_modif:
            chosen_case_modif = random.choice(CASE_FUNCS)
            sequence_df.loc[i, "token"] = chosen_case_modif(sequence_df.loc[i, "token"])
    return sequence_df


def split_sequences(dataset_df: pd.DataFrame):
    df_list = np.split(dataset_df, dataset_df[dataset_df.isnull().all(1)].index)
    sequence_list = []

    for df in df_list:
        sequence_not_annotated = all(df["tag"].fillna("O").values == "O")
        if len(df) > 1:  # only if there is more than one row
            df = df.drop(df.index[[0]])  # drop the empty space
        sequence_list.append((sequence_not_annotated, df))

    return sequence_list


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
