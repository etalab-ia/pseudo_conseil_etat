'''
This script calculates a metric "metier", that is, among (N) documents how many have a pseudonymisation error (p_e).
E = p_e / N
A pseudonymisation error can be of three types: under, over, and miss anonymization.
* Under anonymization is when an entity is marked as an entity by the hand annotated data (gold standard)
and predicted to be not an entity. For example, B-NOM predicted as O.
* Over anonymization is when an entity is marked as not an entity by the hand annotated data (gold standard)
 and predicted to be an entity. For example, O predicted as B-PER.
* Miss anonymization is when an entity is marked as an entity by the hand annotated data (gold standard)
and also predicted to be an entity, but of different type (O). For example, B-NOM predicted as B-PRENOM.

* A fourth and final error is the number of any kind of error (amongst the last three) at least once in a document.
 This is a more strict metric as it considers all possible errors.
The metric is thus the number of documents that contain at least one of said errors.


The script takes one document with three columns, token, real tag, predicted tag
Usage:
    evaluation_metier.py <evaluation_conll_path> [options]

Arguments:
    <evaluation_conll_path>                    Evaluation CoNLL file (3 columns: token, real_tag, pred_tag)
    --consider_over_in_total                   If passed, take into consideration the E_{over} error in the total error
                                                computation
'''
from collections import defaultdict
from itertools import groupby
from pathlib import Path

from argopt import argopt
import pandas as pd


def main(evaluation_conll_path, consider_over_in_total=False):
    df_evaluation = pd.read_csv(evaluation_conll_path, names=["token", "true_tag", "pred_tag"],
                                delim_whitespace=True, engine="python",
                                skip_blank_lines=False).fillna("")
    documents_tokens = [list(grp) for is_docstart, grp in groupby(df_evaluation["token"].iteritems(),
                                                                  lambda x: x[1] == "-DOCSTART-") if not is_docstart]

    list_over_anonym = []
    list_miss_anonym = []
    list_under_anonym = []
    list_all_anonym = []
    N = len(documents_tokens)
    results_dict = defaultdict(list)
    for doc_nb, document in enumerate(documents_tokens):
        indices = [i[0] for i in document]
        temp_df = df_evaluation.loc[indices]

        under_anonym_df = pd.DataFrame(temp_df[(temp_df["pred_tag"] == "O") & (temp_df["true_tag"] != "O")])
        over_anonym_df = pd.DataFrame(temp_df[(temp_df["true_tag"] == "O") & (temp_df["pred_tag"] != "O")])
        miss_anonym_df = pd.DataFrame(temp_df[(temp_df["true_tag"] != "O") & (temp_df["pred_tag"] != "O") & (temp_df["true_tag"] != temp_df["pred_tag"])])
        if not consider_over_in_total:
            all_anonym_ids = set(under_anonym_df.index.to_list() + over_anonym_df.index.to_list() +
                                miss_anonym_df.index.to_list())
        else:
            all_anonym_ids = set(under_anonym_df.index.to_list() + miss_anonym_df.index.to_list())

        under_anonym_df.loc[:, "errors"] = under_anonym_df["token"] + " : " + \
                                           under_anonym_df["true_tag"] + "," + under_anonym_df["pred_tag"]

        over_anonym_df.loc[:, "errors"] = over_anonym_df["token"] + " : " + \
                                          over_anonym_df["true_tag"] + "," + over_anonym_df["pred_tag"]

        miss_anonym_df.loc[:, "errors"] = miss_anonym_df["token"] + " : " + \
                                          miss_anonym_df["true_tag"] + "," + miss_anonym_df["pred_tag"]

        under_anonym_freqs = under_anonym_df.errors.value_counts(sort=True).to_dict()
        over_anonym_freqs = over_anonym_df.errors.value_counts(sort=True).to_dict()
        miss_anonym_freqs = miss_anonym_df.errors.value_counts(sort=True).to_dict()

        nb_under_anonym_errors = len(under_anonym_df)
        nb_over_anonym_errors = len(over_anonym_df)
        nb_miss_anonym_errors = len(miss_anonym_df)
        nb_all_errors = len(all_anonym_ids)

        if not sum([nb_under_anonym_errors, nb_over_anonym_errors, nb_miss_anonym_errors]):
            continue

        results_dict["doc_nb"].append(doc_nb)
        for error_name, error_count, errors_instances in [("nb_under_anonym_errors", nb_under_anonym_errors,
                                                           under_anonym_freqs),
                                        ("nb_over_anonym_errors", nb_over_anonym_errors, over_anonym_freqs),
                                        ("nb_miss_anonym_errors", nb_miss_anonym_errors, miss_anonym_freqs)]:
            results_dict[error_name].append(error_count)
            results_dict[f"{error_name[3:]}_frequencies"].append(errors_instances)

        list_under_anonym.append(nb_under_anonym_errors)
        list_over_anonym.append(nb_over_anonym_errors)
        list_miss_anonym.append(nb_miss_anonym_errors)
        list_all_anonym.append(nb_all_errors)

    under_anonym_error = sum(1 for n in list_under_anonym if n) / N
    over_anonym_error = sum(1 for n in list_over_anonym if n) / N
    miss_anonym_error = sum(1 for n in list_miss_anonym if n) / N
    all_anonym_error = sum(1 for n in list_all_anonym if n) / N

    return under_anonym_error, over_anonym_error, miss_anonym_error, all_anonym_error, \
           list_under_anonym, list_over_anonym, list_miss_anonym,\
           results_dict


if __name__ == '__main__':
    parser = argopt(__doc__).parse_args()
    evaluation_conll_path = Path(parser.evaluation_conll_path)
    consider_over_in_total = parser.consider_over_in_total
    all_results = main(evaluation_conll_path=evaluation_conll_path, consider_over_in_total=consider_over_in_total)

    under_anonym_error = all_results[0]
    over_anonym_error = all_results[1]
    miss_anonnym_error = all_results[2]
    all_anonym_error = all_results[3]
    results_dict = all_results[7]

    results_df = pd.DataFrame.from_dict(results_dict)

    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 22000)
    print(f"Analysis of file {evaluation_conll_path}")
    print()
    for doc_nb, row in results_df.iterrows():
        print(f"Doc nb: {row['doc_nb']}")
        print(f"\tNb Under Anonyimized: {row['nb_under_anonym_errors']}")
        print(f"\tNb Miss Anonymized: {row['nb_miss_anonym_errors']}")
        print(f"\tNb Over Anonymized: {row['nb_over_anonym_errors']}")

        print("\tUnder Anonymized Errors:")
        for error, freq in row["under_anonym_errors_frequencies"].items():
            print(f"\t\t{error}: {freq}")

        print("\tMiss Anonymized Errors:")
        for error, freq in row["miss_anonym_errors_frequencies"].items():
            print(f"\t\t{error}: {freq}")

        print("\tOver Anonymized Errors:")
        for error, freq in row["over_anonym_errors_frequencies"].items():
            print(f"\t\t{error}: {freq}")



    print()
    print(f"The under anonymization error (not anonymizing an annotated entity (PRENOM as O)) is: "
          f" {under_anonym_error:.2f}")
    print(f"The miss anonymization error (not anonymizing correctly an entity (NOM as PRENOM)) is: "
          f" {miss_anonnym_error:.2f}")
    print(f"The over anonymization error (assigning a tag to non entities (O as NOM)) is:   "
          f"{over_anonym_error:.2f}")
    print(f"The total anonymization error (any kind of error) is: {all_anonym_error:.2f}")