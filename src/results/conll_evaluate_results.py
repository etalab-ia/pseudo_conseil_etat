'''
Evaluates the performance of a CoNLL format annotated file. Also shows the errors that were found in the file.
The file should have three columns (token, true tag, predicted tag).

Usage:
    conll_evaluate_results.py <conll_file_path> <output_results_path> [options]

Arguments:
    <conll_file_path>                Annotated CoNLL file using the model
    <output_results_path>            Output text fiile where to save the analysis

    --window=<p>                     If equal to "single" print the single tokens that were misclassified. [default: single]
                                     If it is an int, show the previous and following n tokens around the error.
    --type_error=<n>                What type of errors to show. For ex., "B-PER,O" will show the errors when
                                    the true label was B-PER but the predicted label is O (default: None)
'''

import pandas as pd
from argopt import argopt

from seqeval.metrics import classification_report, f1_score

from src.results.confusion_matrix_pretty_print import print_confusion_matrix


def print_results(y_true, y_pred):
    classif_report = classification_report(y_true, y_pred)
    print(classif_report)

    fscore = f1_score(y_true, y_pred)
    print(f"F-score (micro): {fscore:.2f}")
    fscore_str = f"F-score (micro): {fscore:.2f}"

    labels = list(set(y_true))
    labels.pop(labels.index("O"))
    labels = sorted(labels, key=lambda x: x[2]) + ["O"]

    cm = print_confusion_matrix(y_true=y_true, y_pred=y_pred,
                                labels=labels,
                                return_string=True)
    print(cm)

    return classif_report, fscore_str, cm


def print_errors(results_df: pd.DataFrame, type_error=None, window="single", return_string=False):
    """
    Show the errors found in the read CoNLL file
    :param results_df: Input CoNLL file to test
    :param type_error: Dict containing the types of errors to show: ex.: {"true": "B-PER_NOM", "pred": "O"}.
                        Show all the errors by default
    :param window: If "single", show the single misclassified token, if an int, show the previous and next n tokens
    :return_string: If True, print AND return a string with the results
    :return:
    """
    from io import StringIO
    import sys

    errors_string = StringIO()
    old_stdout = sys.stdout
    if return_string:
        errors_string = StringIO()
        sys.stdout = errors_string

    results_df = results_df.fillna("")
    results_df.index = range(1, len(results_df) + 1)
    if type_error:
        errors_idx = results_df[(results_df["true_tag"] == type_error["true"]) &
                                (results_df["pred_tag"] == type_error["pred"])].index

    else:
        errors_idx = results_df[results_df["pred_tag"] != results_df["true_tag"]].index

    if window == "single":
        final_df = results_df.loc[errors_idx]
        print(final_df.to_string())
    elif isinstance(window, int):
        lower_bound, upper_bound = (-1, -1)
        for idx in errors_idx:
            if lower_bound < idx < upper_bound:
                continue
            lower_bound = max(0, idx - window)
            upper_bound = min(errors_idx.max(), idx + window)
            window_df = results_df.loc[lower_bound:upper_bound, :]
            print(f"Line {idx} of the CoNLL file:", end="\n\t")
            print(window_df, end="\n\n")

    if return_string:
        sys.stdout = old_stdout
    return errors_string.getvalue()


def main(conll_file_path, output_results_path, type_error, window):
    # Load conll file
    results_df = pd.read_csv(conll_file_path, delim_whitespace=True, names=["token", "true_tag", "pred_tag"],
                             skip_blank_lines=False)
    y_true = results_df["true_tag"].dropna().values.tolist()
    y_pred = results_df["pred_tag"].dropna().values.tolist()
    results = print_results(y_true=y_true, y_pred=y_pred)
    print()
    errors = print_errors(results_df=results_df, type_error=type_error, window=window, return_string=True)
    print(errors)
    results_errors = list(results) + [errors]

    with open(output_results_path, "w") as outo:
        for info in results_errors:
            outo.write(str(info))
            outo.write("\n\n")


if __name__ == '__main__':
    parser = argopt(__doc__).parse_args()
    conll_file_path = parser.conll_file_path
    output_results_path = parser.output_results_path
    window = parser.window
    if window.isdigit():
        window = int(window)

    if parser.type_error:
        type_error = parser.type_error.split(",")
        type_error = {"true": type_error[0], "pred": type_error[1]}
    else:
        type_error = parser.type_error

    main(conll_file_path=conll_file_path,
         output_results_path=output_results_path, type_error=type_error,
         window=window)
