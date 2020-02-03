'''
Once the CoNLL files have been generated, split these files in train, validation and test. Trying to keep the
distribution of tags stratified (same proportion in each split).

In all the 24k CoNLL files, there is the following distribution of sequences (counting only the B-X tags, as I am
interested in the sequence instead of the individual tokens count):
B-PER_NOM : 286681
B-PER_PRENOM: 59315
B-LOC: 4616


Usage:
    split_dataset.py <conll_folder> <train_dev_test_folder> [options]

Arguments:
    <conll_folder>                     Folder path with the CoNLL annotation files
    <train_dev_test_folder>            Folder path with the CoNLL annotation files splitted
    --cores=<n> CORES                  Number of cores to use [default: 1:int]
'''
import glob
import logging
import os
import subprocess
from collections import defaultdict

from joblib import Parallel, delayed

import numpy as np
from argopt import argopt
from random import sample, seed
import re

TAGS_DICT = [r"\sB-PER_NOM", r"\sB-PER_PRENOM", r"\sB-LOC"]

def count_tags(conll_paths):
    # conll_paths = ["/data/conseil_etat/corpus_CoNLL/410885_CoNLL.txt"]
    file_tag_counts = defaultdict(list)
    tag_file_counts = defaultdict(list)
    for path in conll_paths:
        with open(path) as filo:
            content = filo.read()
            for tag in TAGS_DICT:
                findo = re.findall(tag, content)
                if findo:
                    file_tag_counts[path].append((tag[2:], len(findo)))
                    tag_file_counts[tag[2:]].append((path, len(findo)))

    totals_dict = defaultdict(int)
    for file_path, tags_count in file_tag_counts.items():
        for tag in tags_count:
            totals_dict[tag[0]] += tag[1]
    return file_tag_counts, tag_file_counts, totals_dict


def get_min_class_samples(tag_file_counts, min_class_count=("B-LOC", 500)):
    min_sample = []
    sum_min_tags = 0
    min_class_files = tag_file_counts[min_class_count[0]]
    min_class_files = min_class_files
    i = 0

    while sum_min_tags < min_class_count[1]:

        min_sample.append(min_class_files[i][0])
        sum_min_tags += min_class_files[i][1]
        i += 1
    _, _, counts = count_tags(min_sample)
    print(counts)
    return min_sample

def split_sets(sample_paths, proportion):
    print(f"Number of files to use {len(sample_paths)}")
    train_p, dev_p, test_p = proportion
    train = np.random.choice(sample_paths, size=int(len(sample_paths) * train_p), replace=False)
    print(f"Number of files in train {len(train)}")
    dev_test = list(set(sample_paths) - set(train))
    dev = np.random.choice(dev_test, size=int(len(sample_paths) * dev_p), replace=False)
    test = list(set(dev_test) - set(dev))
    print(f"Number of files in dev {len(dev)}")
    print(f"Number of files in test {len(test)}")
    return train, dev, test


def save_datasets(train, dev, test, train_dev_test_folder):
    for name, dataset in {"train": train, "dev": dev, "test": test}.items():
        for path in dataset:
            subprocess.run(["cp", path, os.path.join(train_dev_test_folder, name)])


if __name__ == '__main__':
    parser = argopt(__doc__).parse_args()
    tagged_file_path = parser.conll_folder
    n_jobs = parser.cores
    number_decisions = 3000
    train_dev_test_folder = parser.train_dev_test_folder


    annotation_conll_paths = glob.glob(tagged_file_path + "**/*_CoNLL.txt", recursive=True)
    if number_decisions:
        annotation_conll_paths = sample(annotation_conll_paths, number_decisions)

    file_tag_counts, tag_file_counts, counts = count_tags(annotation_conll_paths)
    print(counts)
    sample_paths = get_min_class_samples(tag_file_counts, ("B-LOC", 625))
    train, dev, test = split_sets(sample_paths, (.80, .10, .10))
    save_datasets(train, dev, test, train_dev_test_folder)
