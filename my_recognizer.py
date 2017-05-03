import warnings
from asl_data import SinglesData


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
    all_sequences = test_set.get_all_sequences()
    all_Xlengths = test_set.get_all_Xlengths()

    for word_id, _ in all_sequences.items():
      curr_X, curr_lengths = test_set.get_item_Xlengths(word_id)
      best_word = None
      best_score = -float('inf')
      p_dict = {}

      for word, model in models.items():
        try:
          score = model.score(curr_X, curr_lengths)
          p_dict[word] = score
          if score > best_score:
            best_score = score
            best_word = word
        except:
          p_dict[word] = 0
      probabilities.append(p_dict)
      guesses.append(best_word)

    return probabilities, guesses
    
