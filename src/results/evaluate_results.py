from itertools import groupby

import pandas as pd

pd.set_option('display.max_rows', 1000)
from seqeval.metrics import classification_report, f1_score

from src.results.confusion_matrix_pretty_print import print_confusion_matrix


def print_results(y_true, y_pred):
    print(classification_report(y_true, y_pred))
    fscore = f1_score(y_true, y_pred)
    print(f"F=score (micro??): {fscore:.2f}")

    print_confusion_matrix(y_true=y_true, y_pred=y_pred, labels=["B-LOC", "I-LOC", "B-PER_NOM", "I-PER_NOM",
                                                                 "B-PER_PRENOM", "B-PER_PRENOM", "O"])

def print_errors(results_df: pd.DataFrame, type_error={"true": "B-PER_NOM", "pred": "O"}, window=5):
    results_df = results_df.fillna("")

    # Find groups of sequences (separated by a empty space)
    gps_tokens = [list(grp) for nonempty, grp in groupby(results_df["token"], bool) if nonempty]
    gps_pred_tags = [list(grp) for nonempty, grp in groupby(results_df["pred_tag"], bool) if nonempty]
    gps_true_tags = [list(grp) for nonempty, grp in groupby(results_df["true_tag"], bool) if nonempty]

    for gp_tokens, gp_true, gp_pred in zip(gps_tokens, gps_true_tags, gps_pred_tags):
        differences_true_pred = [i for (i, t, p) in enumerate(zip(gp_true, gp_pred)) if t != p]
        if not differences_true_pred:
            continue
        if type_error:
            differences_true_pred = [line_nb for (line_nb, t, p) in enumerate(zip(gp_true, gp_pred))
                                 if t == type_error["true"] and p == type_error["pred"]]
            if not differences_true_pred:
                continue

        if isinstance(window, int):

            pass
        error_df = pd.DataFrame({"token": gp_tokens, "true_tag": gp_true, "pred_tag": gp_pred})

        print(error_df)
        print()

    pd.set_option('display.max_colwidth', 100)
    pass
