import re

from sacremoses import MosesPunctNormalizer, MosesTokenizer, MosesDetokenizer

mpn = MosesPunctNormalizer(lang="fr")
mt = MosesTokenizer(lang="fr")
detokenizer = MosesDetokenizer()


def moses_detokenize(list_strings):
    return detokenizer.detokenize(list_strings)


def moses_tokenize(phrase):
    phrase = mpn.normalize(phrase)
    tokens = mt.tokenize(phrase)
    return tokens


def tokenize(phrase):
    # TODO: Tokenize with proper tokenizer
    tokens = re.split("[\s,.]+", phrase)
    tokens = [t for t in tokens if t]
    return tokens
