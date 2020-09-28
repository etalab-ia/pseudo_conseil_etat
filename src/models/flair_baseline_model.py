from flair.data import Corpus
from flair.embeddings import TokenEmbeddings, WordEmbeddings, StackedEmbeddings, CharacterEmbeddings, FlairEmbeddings, \
    CamembertEmbeddings, BertEmbeddings, PooledFlairEmbeddings
from typing import List
import torch

from src.utils.monitoring import Monitor

torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import numpy as np
np.random.seed(0)
from flair.data import Corpus
from flair.datasets import ColumnCorpus
import flair
import sys

if len(sys.argv) < 3:
    print("Usage: Please give the name of the folder containing the train, dev, test sets and the output folder")
    exit(0)

data_folder = sys.argv[1]
output_folder = sys.argv[2]

flair.cache_root = "cache/"

def create_flair_corpus(data_folder):
    # define columns
    columns = {0: 'text', 1: 'ner'}

    # init a corpus using column format, data folder and the names of the train, dev and test files
    corpus: Corpus = ColumnCorpus(data_folder, columns,
                                  train_file='train.txt',
                                  test_file='test.txt',
                                  dev_file='dev.txt')
    return corpus

# 1. get the corpus
corpus: Corpus = create_flair_corpus(data_folder)
print(corpus)

# 2. what tag do we want to predict?
tag_type = 'ner'

# 3. make the tag dictionary from the corpus
tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)
print(tag_dictionary.idx2item)


# 4. initialize embeddings
embedding_types: List[TokenEmbeddings] = [

    # WordEmbeddings('fr'),
    WordEmbeddings("/data/embeddings/doctrine/doctine_gensim_embeddings.gensim"),
    # comment in this line to use character embeddings
    CharacterEmbeddings(),

    # comment in these lines to use flair embeddings
    # FlairEmbeddings('fr-forward'),
    # FlairEmbeddings('fr-backward'),

    # bert embeddings
    # BertEmbeddings('bert-base-multilingual-cased')
    # CamembertEmbeddings()
    # CCASS Flair Embeddings FWD
    # FlairEmbeddings('/data/embeddings_CCASS/flair_language_model/jurinet/best-lm.pt'),

    # CCASS Flair Embeddings BWD
    # FlairEmbeddings('/data/embeddings_CCASS/flair_language_model/jurinet/best-lm-backward.pt')
]
monitor = Monitor(50)
embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)

# 5. initialize sequence tagger
from flair.models import SequenceTagger

tagger: SequenceTagger = SequenceTagger(hidden_size=100,
                                        embeddings=embeddings,
                                        tag_dictionary=tag_dictionary,
                                        tag_type=tag_type,
                                        rnn_layers=2,
                                        dropout=0.5,
                                        use_crf=True,
                                        )

# 6. initialize trainer
from flair.trainers import ModelTrainer
from torch.optim.adam import Adam
trainer: ModelTrainer = ModelTrainer(tagger, corpus, optimizer=Adam)
trainer.num_workers = 8
# 7. start training

trainer.train(output_folder,
              learning_rate=0.1,
              mini_batch_size=8,
              max_epochs=5,
              use_amp=False,
              embeddings_storage_mode="gpu")

monitor.stop()
# 8. plot weight traces (optional)
# from flair.visual.training_curves import Plotter
# plotter = Plotter()
# plotter.plot_weights('models/baseline_ner/weights.txt')
