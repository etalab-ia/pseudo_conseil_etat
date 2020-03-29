'''Computes some statistics of a given CoNLL file

Usage:
    conll_stats.py <conll_file>

Arguments:
    <conll_file>                      Folder path with the DOC decisions files to transform to TXT
'''

from argopt import argopt
import pandas as pd
from itertools import groupby

if __name__ == '__main__':
    parser = argopt(__doc__).parse_args()
    conll_file_path = parser.conll_file

    df_conll = pd.read_csv(conll_file_path, sep="\t", skip_blank_lines=False, header=None, names=["token", "tag"])
    nb_decisions = df_conll[df_conll.token == "-DOCSTART-"].shape[0]

    df_conll["tags_nobio"] = df_conll.tag.str[2:]
    tags = df_conll.tags_nobio.value_counts()

    approx_nb_phrases = df_conll.token[df_conll.token.isnull()].shape[0]

    # Number of phrases w/o annotations
    tags_phrases = df_conll.tag.fillna("").values
    gps = (grp for nonempty, grp in groupby(tags_phrases, bool) if nonempty)
    all_lines = [list(g) for g in gps]
    nb_phrases_wo_annotation = sum(1 for p in all_lines if all([True if i == "O" else False for i in p]))
    tags_ratio = tags[1:] * 100 / tags[1:].sum()
    pd.set_option('display.max_colwidth', 100)
    results = pd.Series({"nb_decisions": nb_decisions, "tags_distribution": str(tags.to_dict()),
                         "BOI_tags_ratio": str(tags_ratio.to_dict()),
                         "nb_phrases": approx_nb_phrases, "nb_phrases_without_entities": nb_phrases_wo_annotation})
    print(results)
