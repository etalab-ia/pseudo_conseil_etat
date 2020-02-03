from flair.data import Corpus
from flair.embeddings import TokenEmbeddings, WordEmbeddings, StackedEmbeddings, CharacterEmbeddings, FlairEmbeddings, \
    BertEmbeddings
from typing import List


from src.data.load_flair_corpus import create_flair_corpus


# 1. get the corpus
data_folder = './data/interim/sampled_train_dev_test/'

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
    FlairEmbeddings('/data/embeddings_CCASS/pavel/flair_language_model/jurinet/best-lm.pt'),

    # CCASS Flair Embeddings BWD
    FlairEmbeddings('/data/embeddings_CCASS/pavel/flair_language_model/jurinet/best-lm-backward.pt')
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
trainer.num_workers = 10
# 7. start training
trainer.train('models/baseline_ner',
              learning_rate=0.1,
              mini_batch_size=32,
              max_epochs=150)

# 8. plot weight traces (optional)
from flair.visual.training_curves import Plotter
plotter = Plotter()
plotter.plot_weights('models/baseline_ner/weights.txt')