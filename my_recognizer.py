import warnings
from asl_data import SinglesData
from collections import OrderedDict

def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []

    all_Xlengths = OrderedDict(sorted(test_set.get_all_Xlengths().items(), key=lambda t: t[0]))

    for X, length in all_Xlengths.values():
        word_probabilities = {}
        for model_word, model in models.items():
            try:
                word_probabilities[model_word] = model.score(X, length)
            except Exception as e:
                word_probabilities[model_word] = float("-inf")

        best_guess, score = sorted(word_probabilities.items(), key=lambda t: -t[1])[0]
        probabilities.append(word_probabilities)
        guesses.append(best_guess)

    return probabilities, guesses