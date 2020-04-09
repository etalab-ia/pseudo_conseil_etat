'''Transforms the docs of the conseil d'etat decisions into txt format.

Usage:
    doc2txt.py <docs_folder> [options]

Arguments:
    <docs_folder>                      Folder path with the DOC decisions files to transform to TXT
    --only_convert_missing                  Only convert those DOCs that are missing [default: False: bool]
    --cores=<n> CORES                  Number of cores to use [default: 1:int]
'''
import logging
import os
import subprocess
from glob import glob

from argopt import argopt
from joblib import Parallel, delayed
from tqdm import tqdm


def doc2txt(doc_path):
    txt_path = doc_path[:-4] + ".txt"

    doc_folder = os.path.dirname(doc_path)
    if os.path.exists(txt_path) and only_convert_missing:
        tqdm.write(f"File {doc_path} already converted.")
        return 0
    subprocess.check_call(["soffice", "--headless", "--convert-to", "txt:Text", doc_path, "--outdir", doc_folder])
    return 1


if __name__ == '__main__':
    parser = argopt(__doc__).parse_args()
    doc_files_path = parser.docs_folder
    only_convert_missing = bool(parser.only_convert_missing)
    n_jobs = parser.cores
    doc_paths = []
    if not os.path.isdir(doc_files_path) and os.path.isfile(doc_files_path):
        doc_paths = [doc_files_path]
    else:
        doc_paths = glob(doc_files_path + "/**/*.doc", recursive=True)
    if not doc_paths:
        raise Exception(f"Path {doc_paths} not found")

    if only_convert_missing:
        transformed_docs_ids = [p[:-4] for p in glob(doc_files_path + "**/*.txt", recursive=True)]
        doc_paths_ids = [p[:-4] for p in doc_paths]
        doc_paths = list(set.difference(set(doc_paths_ids), transformed_docs_ids))
        doc_paths = [p + ".doc" for p in doc_paths]

    if n_jobs < 2:
        job_output = []
        for doc_path in tqdm(doc_paths):
            tqdm.write(f"Converting file {doc_path}")
            job_output.append(doc2txt(doc_path))
    else:
        job_output = Parallel(n_jobs=n_jobs)(delayed(doc2txt)(doc_path) for doc_path in tqdm(doc_paths))

    logging.info(
        f"{sum(job_output)} DOC files were converted to TXT. {len(job_output) - sum(job_output)} files "
        f"had some error.")
