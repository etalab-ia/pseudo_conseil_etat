'''
This script calculates a metric "metier", that is, among (N) documents how many have a pseudonymisation error (d_e).
M = d_e / N
A pseudonymisation error can be strict or flexible. A strict error is when the golden differ with the predicted tag,
no matter the predicted tag. A flexible error is when the golden tag is predicted as any entity, even if not the same
(eg., B-PER_NOM predicted as I-PER_PRENOM)

The script takes one document with three columns, token, real tag, predicted tag
Usage:
    evaluation_metier.py <evaluation_conll_path> [options]

Arguments:
    <evaluation_conll_path>                    Evaluation CoNLL file (3 columns: token, real_tag, pred_tag)
'''
from collections import defaultdict
from itertools import groupby
from pathlib import Path

from argopt import argopt
import pandas as pd


def main(evaluation_conll_path):
    df_evaluation = pd.read_csv(evaluation_conll_path, names=["token", "true_tag", "pred_tag"],
                                               delim_whitespace=True, engine="python",
                                               skip_blank_lines=False).fillna("")
    documents_tokens = [list(grp) for is_docstart, grp in groupby(df_evaluation["token"].iteritems(),
                                                                  lambda x: x[1] == "-DOCSTART-") if not is_docstart]

    indices = []
    list_not_identified = []
    list_miss_identified = []
    N = len(documents_tokens)
    results_dict = defaultdict(list)
    for doc_nb, document in enumerate(documents_tokens):
        indices = [i[0] for i in document]
        temp_df = df_evaluation.loc[indices]
        not_identified_df = pd.DataFrame(temp_df[(temp_df["true_tag"] != temp_df["pred_tag"]) &
                                        (temp_df["true_tag"] != "O")])
        miss_identified_df = pd.DataFrame(temp_df[(temp_df["true_tag"] != temp_df["pred_tag"]) &
                                         (temp_df["true_tag"] != "O") &
                                         (temp_df["pred_tag"] != "O")])

        not_identified_df.loc[:, "errors"] = not_identified_df["token"] + " : " + \
                                             not_identified_df["true_tag"] + "," + not_identified_df["pred_tag"]
        miss_identified_df.loc[:, "errors"] = miss_identified_df["token"] + " : " + \
                                              miss_identified_df["true_tag"] + "," + miss_identified_df["pred_tag"]
        errors_not_identified = not_identified_df.errors.value_counts(sort=True).to_dict()
        errors_miss_identified = miss_identified_df.errors.value_counts(sort=True).to_dict()
        nb_not_identified = len(not_identified_df)
        nb_miss_identified = len(miss_identified_df)
        if nb_not_identified:
            results_dict["doc_nb"].append(doc_nb)
            results_dict["nb_not_identified"].append(nb_not_identified)
            results_dict["nb_miss_identified"].append(nb_miss_identified)
            results_dict["not_identified_frequency"].append(errors_not_identified)
            results_dict["miss_identified_frequency"].append(errors_miss_identified)

        list_not_identified.append(nb_not_identified)
        list_miss_identified.append(nb_miss_identified)


    strict_error = sum(1 for n in list_not_identified if n)
    flexible_error = strict_error - sum(1 for n in list_miss_identified if n)

    strict_error /= N
    flexible_error /= N
    return strict_error, flexible_error, list_not_identified, list_miss_identified, results_dict


if __name__ == '__main__':
    parser = argopt(__doc__).parse_args()
    evaluation_conll_path = Path(parser.evaluation_conll_path)

    strict_error, flexible_error, _, _, results_dict = main(evaluation_conll_path=evaluation_conll_path)
    results_df = pd.DataFrame.from_dict(results_dict)

    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 22000)
    print(f"Analysis of file {evaluation_conll_path}")
    print()
    for doc_nb, row in results_df.iterrows():
        print(f"Doc nb: {row['doc_nb']}")
        print(f"\tNb Not Identified: {row['nb_not_identified']}")
        print(f"\tNb Miss Identified    : {row['nb_miss_identified']}")
        print("\tNot Identified Errors:")
        for error, freq in row["not_identified_frequency"].items():
            print(f"\t\t{error}: {freq}")
        print("\tMiss Identified Errors:")
        for error, freq in row["miss_identified_frequency"].items():
            print(f"\t\t{error}: {freq}")
    # print(results_df)
    print()
    print(f"The total strict error (predicted label differs from golden label) is : {strict_error}")
    print(f"The total flexible error (predicted label is not correct but still identified) is: {flexible_error}")