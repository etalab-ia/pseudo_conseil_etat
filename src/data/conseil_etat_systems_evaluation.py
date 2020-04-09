'''Takes two folders, one with gold labels CoNLL files, the second with labels predicted from a system
 It extracts the CoNLL files inside folder one, determines which are in common with the second and
outputs one file with three columns: tokens, golden tag, and predicted tag
Usage:
    add_golden_tags.py <golden_folder_path> <predicted_folder_path>  <output_path> [options]

Arguments:
    <golden_folder_path>       Folder with CoNLL files with golden tags
    <predicted_folder_path>    Folder with CoNLL files with predicted tags
    <output_path>              Output CoNLL file with the three coluns
'''
import glob
from pathlib import Path

from argopt import argopt
import pandas as pd


def main(golden_conll_path: Path, predicted_conll_path: Path, output_path: Path):
    golden_files = glob.glob(golden_conll_path.as_posix() + "/**/*CoNLL*.txt", recursive=True)
    prediction_files = glob.glob(predicted_conll_path.as_posix() + "/**/*CoNLL*.txt", recursive=True)

    golden_files_ids = [Path(p).stem.split("_")[0] for p in golden_files]
    prediction_files_ids = [Path(p).stem.split("_")[0] for p in prediction_files]

    intersect_ids = set(golden_files_ids).intersection(prediction_files_ids)

    list_dfs = []
    for id_doc in intersect_ids:
        golden_file_path = [f for f in golden_files if id_doc in f]
        prediction_file_path = [f for f in prediction_files if id_doc in f]

        assert golden_file_path, f"The file with id {id_doc} was not found in the golden folder"
        assert prediction_file_path, f"The file with id {id_doc} was not found in the predicted folder"
        print(f"\tReading gold file: {golden_file_path[0]}")
        gold_df = pd.read_csv(golden_file_path[0], names=["token", "tag"], delim_whitespace=True, engine="python")
        print(f"\tReading pred file: {prediction_file_path[0]}")
        pred_df = pd.read_csv(prediction_file_path[0], names=["token", "tag"], delim_whitespace=True, engine="python")

        assert len(gold_df) == len(pred_df), \
            f"The files {golden_file_path} and {prediction_file_path} do not have the same dimensions"

        gold_df["pred_tag"] = pred_df["tag"]
        list_dfs.append(gold_df.copy(deep=True))
        print(f"Created true/pred dataframe with files {golden_file_path[0]} and {prediction_file_path[0]}")

    final_df: pd.DataFrame = pd.concat(list_dfs)
    final_df.to_csv(output_path, header=None, index=None, sep="\t")
    return final_df

if __name__ == '__main__':
    parser = argopt(__doc__).parse_args()
    golden_conll_path = Path(parser.golden_folder_path)
    predicted_conll_path = Path(parser.predicted_folder_path)
    output_path = Path(parser.output_path)

    main(golden_conll_path=golden_conll_path,
         predicted_conll_path=predicted_conll_path,
         output_path=output_path)
