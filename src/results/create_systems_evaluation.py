'''
This script generates a single CoNLL file with the golden tags in the second col and predicted tags in the
third column. These predicted tags may come from a RB system (found in its corresponding CoNLL files) or from a NLP
system (idem). The produced CoNLL file can be evaluated and thus both systems can be compared. The CoNLL files should
describe the same documents (same tokens) for both solutions.

The script takes two folders, one with gold labels CoNLL files, the second with labels predicted from a given system.
It extracts the CoNLL files inside folder one, determines which are in common with the second and
outputs one file with three columns: tokens, golden tag, and predicted tag
Usage:
    create_systems_evaluation.py <golden_folder_path> <predicted_folder_path>  <output_path> [options]

Arguments:
    <golden_folder_path>                    Folder with CoNLL files with golden tags
    <predicted_folder_path>                 Folder with CoNLL files with predicted tags
    <output_path>                           Output CoNLL file with the three coluns
    --comparison_folder_path=<c>   COMP     Folder with the comparison CoNLL files. Pass this to use the exact same files in solution A and B and Gold (default: None)
    --only_gold_annotated                   Keep only the predictions corresponding to hand annotated tags in the output CoNLL files
'''
import glob
from pathlib import Path

from argopt import argopt
import pandas as pd

def flexible_intersect(list_a, list_b):
    intersection = []
    for elem_a in list_a:
        for elem_b in list_b:
            if elem_a in elem_b:
                intersection.append(elem_a)
                break
            elif elem_b in elem_a:
                intersection.append(elem_b)
                break
    return intersection


def main(golden_conll_path: Path, predicted_conll_path: Path, output_path: Path,
         comparison_folder_path: Path = None, only_gold_annotated: bool = False):

    golden_files = glob.glob(golden_conll_path.as_posix() + "/**/*CoNLL*.txt", recursive=True)
    prediction_files = glob.glob(predicted_conll_path.as_posix() + "/**/*CoNLL*.txt", recursive=True)

    golden_files_ids = [Path(p).stem.split("_")[0].split()[0].replace("pr1", "") for p in golden_files]
    prediction_files_ids = [Path(p).stem.split("_")[0].split()[0].replace("pr1", "") for p in prediction_files]
    intersect_ids = sorted(flexible_intersect(golden_files_ids, prediction_files_ids))
    not_same_df = []
    if comparison_folder_path:
        comparison_files = glob.glob(comparison_folder_path.as_posix() + "/**/*CoNLL*.txt", recursive=True)
        comparison_files_ids = [Path(p).stem.split("_")[0].split()[0].replace("pr1", "") for p in comparison_files]
        intersect_ids = sorted(flexible_intersect(intersect_ids, comparison_files_ids))

    print(f"Using the following golden files IDs {intersect_ids}")
    list_dfs = []
    for id_doc in intersect_ids:
        golden_file_path = [f for f in golden_files if id_doc in f]
        prediction_file_path = [f for f in prediction_files if id_doc in f]

        assert golden_file_path, f"The file with id {id_doc} was not found in the golden folder"
        assert prediction_file_path, f"The file with id {id_doc} was not found in the predicted folder"
        print(f"\tReading gold file: {golden_file_path[0]}")
        gold_df: pd.DataFrame = pd.read_csv(golden_file_path[0], names=["token", "tag"], delim_whitespace=True, engine="python",
                              skip_blank_lines=False).fillna("")

        print(f"\tReading pred file: {prediction_file_path[0]}")
        pred_df: pd.DataFrame = pd.read_csv(prediction_file_path[0], names=["token", "tag"], delim_whitespace=True, engine="python",
                              skip_blank_lines=False).fillna("")

        if not len(gold_df) == len(pred_df):
            print(f"The files {golden_file_path} and {prediction_file_path} do not have the same dimensions")
            not_same_df.append((golden_file_path, prediction_file_path))
            continue

        if only_gold_annotated:
            gold_df = gold_df[(gold_df["tag"] != "O") & (gold_df["tag"] != "")]
            pred_df = pred_df.loc[gold_df.index]
        gold_df["pred_tag"] = pred_df["tag"]
        list_dfs.append(gold_df.copy(deep=True))
        print(f"Created true/pred dataframe with files {golden_file_path[0]} and {prediction_file_path[0]}")

    print(f"Total number of CoNLL files within the output file: {len(list_dfs)}")
    print(f"These files had some error")
    final_df: pd.DataFrame = pd.concat(list_dfs)
    final_df.to_csv(output_path, header=None, index=None, sep="\t")
    return final_df


if __name__ == '__main__':
    parser = argopt(__doc__).parse_args()
    golden_conll_path = Path(parser.golden_folder_path)
    predicted_conll_path = Path(parser.predicted_folder_path)
    output_path = Path(parser.output_path)
    comparison_folder_path = parser.comparison_folder_path
    only_gold_annotated = parser.only_gold_annotated

    if comparison_folder_path:
        comparison_folder_path = Path(comparison_folder_path)

    main(golden_conll_path=golden_conll_path,
         predicted_conll_path=predicted_conll_path,
         output_path=output_path,
         comparison_folder_path=comparison_folder_path,
         only_gold_annotated=only_gold_annotated)
