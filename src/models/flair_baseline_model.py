from flair.data import Corpus
from flair.embeddings import TokenEmbeddings, WordEmbeddings, StackedEmbeddings, CharacterEmbeddings, FlairEmbeddings, \
    BertEmbeddings
from typing import List


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

# 1. get the corpus
data_folder = '/data/conseil_etat/train_dev_test/'

corpus: Corpus = create_flair_corpus(data_folder)
print(corpus)

# 2. what tag do we want to predict?
tag_type = 'ner'

# 3. make the tag dictionary from the corpus
tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)
print(tag_dictionary.idx2item)


# 4. initialize embeddings
embedding_types: List[TokenEmbeddings] = [

    # WordEmbeddings('fr-crawl'),

    # comment in this line to use character embeddings
    # CharacterEmbeddings(),

    # comment in these lines to use flair embeddings
    FlairEmbeddings('fr-forward'),
    # FlairEmbeddings('fr-backward'),

    # bert embeddings
    # BertEmbeddings('bert-base-french')

    # CCASS Flair Embeddings FWD
    # FlairEmbeddings('/data/embeddings_CCASS/flair_language_model/jurinet/best-lm.pt'),

    # CCASS Flair Embeddings BWD
    # FlairEmbeddings('/data/embeddings_CCASS/flair_language_model/jurinet/best-lm-backward.pt')
]

embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)

# 5. initialize sequence tagger
from flair.models import SequenceTagger

tagger: SequenceTagger = SequenceTagger(hidden_size=256,
                                        embeddings=embeddings,
                                        tag_dictionary=tag_dictionary,
                                        tag_type=tag_type,
                                        use_crf=True)

# 6. initialize trainer
from flair.trainers import ModelTrainer

trainer: ModelTrainer = ModelTrainer(tagger, corpus)
trainer.num_workers = 20
# 7. start training
trainer.train('models/baseline_ner',
              learning_rate=0.1,
              mini_batch_size=2 ,
              max_epochs=150,
              embeddings_storage_mode="cpu")

# 8. plot weight traces (optional)
from flair.visual.training_curves import Plotter
plotter = Plotter()
plotter.plot_weights('models/baseline_ner/weights.txt')