'''
Once the CoNLL files have been generated, split these files in train, validation and test. Trying to keep the
distribution of tags stratified (same proportion in each split).



Usage:
    create_dataset.py <conll_folder> <train_dev_test_folder> [options]

Arguments:
    <conll_folder>                     Folder path with the CoNLL annotation files
    <train_dev_test_folder>            Folder path to store the CoNLL created dataset
    --nb_min_class=<t> MiNCLASS        Min number of examples of a given class : ex. "B-LOC,2000"
                                       If zero, generate random sample (default: None)
    --number_decisions=<n> DEC         Number of decisions to use [default: None:int]
    --split_ratio=<s> RATIO            Split ratio of the generated train,dev,test dataset [default: "80,10,10":str]
    --skip_files_list=<z> SKIP         Skip the files inside the passed text file (one doc path per line) when creating the train set (default: None)

'''
import glob
import os
import shutil
import subprocess
from collections import defaultdict
from datetime import datetime

import numpy as np
from argopt import argopt
from random import sample, seed, shuffle
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


def concatenate_files2(list_files, dir_path, new_file_path):
    path_new_file = dir_path + f"/{new_file_path}.txt"
    with open(path_new_file, 'w') as outfile:
        for fname in list_files:
            with open(fname) as infile:
                for line in infile:
                    outfile.write(line)
    with open(dir_path + f"/{new_file_path}_used_docs.txt", "w") as outfile:
        outfile.writelines([f"{l}\n" for l in list_files])


def concatenate_files(path_to_files, dataset_name):
    time_stamp = str(datetime.now().strftime("%H_%M_%S"))
    file_dirname = os.path.dirname(path_to_files.rstrip("/")) + f"/{time_stamp}"
    os.mkdir(file_dirname)
    list_files = glob.glob(path_to_files + "/*_CoNLL.txt")
    dataset_path = os.path.join(file_dirname, f"{dataset_name.rstrip('/')}.txt")
    print(f"Creating {dataset_name} set in {dataset_path}")
    with open(dataset_path, 'w') as outfile:
        for fname in list_files:
            with open(fname) as infile:
                for line in infile:
                    outfile.write(line)


def get_min_class_samples(tag_file_counts, min_class_count=("B-LOC", 500)):
    """
    Function that chooses files continuously until the minimum for the selected class is reached (min_min_class_count)

    :param tag_file_counts:
    :param min_class_count:
    :return:
    """

    min_sample = []
    sum_min_tags = 0
    min_class_files = tag_file_counts[min_class_count[0]]
    i = 0

    while sum_min_tags < int(min_class_count[1]):
        if i >= len(min_class_files):
            print(f"All the files containing the minority tags are already added and we have {sum_min_tags}. "
                  f"There are no more files with this tag.")
            break
        min_sample.append(min_class_files[i][0])
        sum_min_tags += min_class_files[i][1]
        i += 1
    _, _, counts = count_tags(min_sample)
    print(f"Desired ags distribution of min_class {min_class_count}. Real sampled corpus documents : {counts}")
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


def save_chosen_conlls(train, dev, test, train_dev_test_folder):
    for dataset_name, dataset in {"train/": train, "dev/": dev, "test/": test}.items():
        set_folder_path = os.path.join(train_dev_test_folder, dataset_name)
        shutil.rmtree(set_folder_path, ignore_errors=True)
        os.mkdir(set_folder_path)
        for path in dataset:
            subprocess.run(["cp", path, os.path.join(train_dev_test_folder, dataset_name)])
        concatenate_files(os.path.join(train_dev_test_folder, dataset_name), dataset_name)


if __name__ == '__main__':
    parser = argopt(__doc__).parse_args()
    tagged_file_path = parser.conll_folder
    train_dev_test_folder = parser.train_dev_test_folder
    split_ratio = [int(r) / 100 for r in parser.split_ratio.split(",")]

    sampled_number_decisions = parser.number_decisions
    if sampled_number_decisions:
        nb_min_class = int(sampled_number_decisions)

    nb_min_class = parser.nb_min_class
    if parser.nb_min_class:
        nb_min_class = parser.nb_min_class.split(",")

    skip_file = parser.skip_files_list
    seed(0)

    annotation_conll_paths = glob.glob(tagged_file_path + "/**/*_CoNLL.txt", recursive=True)
    print(f"Number of CoNLL read files: {len(annotation_conll_paths)}")
    if skip_file and os.path.exists(skip_file):
        with open(skip_file, "r") as readfile:
            files_to_skip = [l.strip() for l in readfile.readlines()]
        annotation_conll_paths = list(set(annotation_conll_paths).difference(files_to_skip))
        print(f"After skipping the files in {skip_file} (with {len(files_to_skip)} files inside). "
              f"We are left with {len(annotation_conll_paths)} documents.")

    sample_paths = shuffle(annotation_conll_paths)

    file_tag_counts, tag_file_counts, counts = count_tags(annotation_conll_paths)
    print(f"Tags distribution of sampled corpus ({len(annotation_conll_paths)} documents) : {counts}")


    if sampled_number_decisions:
        if sampled_number_decisions < len(annotation_conll_paths):
            sample_paths = sample(annotation_conll_paths, sampled_number_decisions)

        else:
            print("You chose to sample more documents that there are available. I am going to give you the whole    "
                  "population of documents.")

    elif nb_min_class:
        sample_paths = get_min_class_samples(tag_file_counts, nb_min_class)


    train, dev, test = split_sets(sample_paths, split_ratio)

    new_dataset_path = train_dev_test_folder + f"/{len(train)}_{len(dev)}_{len(test)}"
    for dataset_name, dataset in {"train/": train, "dev/": dev, "test/": test}.items():
        if not os.path.exists(new_dataset_path):
            os.mkdir(new_dataset_path)
        concatenate_files2(dataset, new_dataset_path, dataset_name.rstrip('/'))
        pass
