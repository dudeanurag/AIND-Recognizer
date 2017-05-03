import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences
from copy import deepcopy

class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Baysian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        best_score = float('inf')
        best_model = None
        min_n = self.min_n_components
        max_n = self.max_n_components
        n_frs = self.X.shape[1]
        logN = np.log(np.sum(self.lengths))

        for n in range(min_n, max_n+1):
            try:
                hmm_model = GaussianHMM(n_components=n, n_iter=1000, \
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
                logL = hmm_model.score(self.X, self.lengths)
            except:
                continue
            p = (n * n) + (2 * n_frs * n) - 1
            bic = -2 * logL + logN * p
            if bic < best_score:
                best_score = bic
                n_components = n
                best_model = hmm_model

        return best_model

# For SelectorDIC model selector, dictionary db_logL stores log-likelihood values for every word.
# key for this dictionary is the word and value is another dictionary with key:value in which
# key is the number of hidden states and value is the log-likelihood value.
# so it has the form {'word1': {n1: value1, n2:value2, ...}, 'word2': {n1: value1, n2:value2, ...}, ...}
db_logL = dict()
stored_sequences = dict()       # It stores the corresponding sequences of feature lists for all words in db_logL


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    # compare_set_dict() compares the stored_sequences with the current sequences of features for each word.
    # If both sequences are equal, the log-likelihood values in db_logL are used, otherwise it is created new. 
    def compare_set_dict(self):
        global stored_sequences
        for word, sequence in self.words.items():
            try:
                stored_word_seq = np.array(stored_sequences[word])
                curr_word_seq = np.array(sequence)
                # if both sequences are not equal, raise exception to populate db_logL again
                if not np.array_equal(curr_word_seq, stored_word_seq):  
                    raise Exception
            except:
                self.create_db()
                break

    # create_db() function populates the db_logL with new log-likehood values 
    # from new feature list sequences passed to the object
    def create_db(self):
        global stored_sequences
        global db_logL
        db_logL.clear()
        stored_sequences = deepcopy(self.words)

        for word, Xlengths in self.hwords.items():
            db_logL[word] = dict()
            word_X, word_lengths = Xlengths
            for nb_hidden in range(self.min_n_components, self.max_n_components + 1):
                try:
                    model = GaussianHMM(n_components=nb_hidden, n_iter=1000, random_state=self.random_state, \
                                        verbose=False).fit(word_X, word_lengths)
                    db_logL[word][nb_hidden] = model.score(word_X, word_lengths)
                except:
                    continue

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        self.compare_set_dict()

        all_words = [k  for  k in  self.words.keys()]
        best_score = -float('inf')
        best_model = None
        n_components = None

        for nb_hidden in range(self.min_n_components, self.max_n_components + 1):
            try:
                logL_this_word = db_logL[self.this_word][nb_hidden]
            except:
                continue

            logL_rest_words = 0.0
            count = 0
            for word in all_words:
                if word == self.this_word:
                    continue
                try:    
                    logL_rest_words += db_logL[word][nb_hidden]
                    count += 1
                except:
                    continue
            dic_score = logL_this_word - logL_rest_words/count

            if dic_score > best_score:
                best_score = dic_score
                n_components = nb_hidden
        if n_components != None:
            return self.base_model(n_components)
        else:
            return None


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''
    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # Split into 3 folds when number of sequences for a word is atleast three
        # Make separate list of train and test folds in train_folds and test_folds
        n_splits = min(3, len(self.sequences))
        if n_splits > 1:
            split_method = KFold(n_splits)
            train_folds = list()
            test_folds = list()
            for cv_train, cv_test in split_method.split(self.sequences):
                train_folds.append(cv_train)
                test_folds.append(cv_test)

        best_score = -float('inf')
        best_model = None
        n_components = None

        for n in range(self.min_n_components, self.max_n_components + 1):
            model = GaussianHMM(n_components=n, n_iter=1000, random_state=self.random_state, verbose=False)
            mean_logL = 0.0
            # If number of sequences is one, we train and score on the same single fold
            if n_splits == 1:
                try:
                    fit_model = model.fit(self.X, self.lengths)
                    mean_logL = fit_model.score(self.X, self.lengths)
                except:
                    continue
            else:
                count = 0       #count keeps record of the total number of folds which were successfully tested and scored.
                for ii in range(len(train_folds)):      #  Iterate and train over each fold 
                    train_X, train_lengths = combine_sequences(train_folds[ii], self.sequences)
                    test_X, test_lengths = combine_sequences(test_folds[ii], self.sequences)
                    try:
                        fit_model = model.fit(train_X, train_lengths)
                        mean_logL += fit_model.score(test_X, test_lengths)
                        count += 1
                    except:
                        continue
                if count > 0:
                    mean_logL = mean_logL / count       # Take mean of all the scores 

            if mean_logL > best_score:
                best_score = mean_logL
                n_components = n
                best_model = fit_model
            
        return best_model
