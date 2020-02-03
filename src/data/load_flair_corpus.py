from flair.data import Corpus
from flair.datasets import ColumnCorpus


def create_flair_corpus(data_folder):
    # define columns
    columns = {0: 'text', 1: 'ner'}

    # init a corpus using column format, data folder and the names of the train, dev and test files
    corpus: Corpus = ColumnCorpus(data_folder, columns,
                                  train_file='train.txt',
                                  test_file='test.txt',
                                  dev_file='dev.txt')

    return corpus


if __name__ == '__main__':
    data_folder = './data/raw/train_dev_test/'

    create_flair_corpus(data_folder)
