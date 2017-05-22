import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


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


    @property
    def n_components(self):
        return range(self.min_n_components, self.max_n_components + 1)



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
        bic_score = lambda logL, p, N: (-2.0 * logL) + (p * np.log(N))
        best_score = float("inf")
        best_model = None

        for num_component in self.n_components:
            try:
                model = self.base_model(num_component).fit(self.X, self.lengths)
                logL = model.score(self.X, self.lengths)
                score = bic_score(logL, model.n_features, len(self.sequences))
                if score < best_score:
                    best_score = score
                    best_model = model
            except Exception as e:
                pass

        return best_model

class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        score_models_list = []
        for num_components in self.n_components:
            try:
                model = self.base_model(num_components)
                model.fit(self.X, self.lengths)
                score = model.score(self.X)
                score_models_list.append((score, model))
            except Exception as e:
                pass
        scores = [score for score, model in score_models_list]
        sum_log_L = sum(scores)
        best_dic_score = float("inf")
        num_models = len(score_models_list)
        best_model = None
        for log_L, model in score_models_list:
            dic_score = SelectorDIC.dic_score(log_L, sum_log_L, num_models)
            if dic_score < best_dic_score:
                best_model = model
                best_dic_score = dic_score

        return best_model

    @staticmethod
    def dic_score(log_L, sum_log_L, num_models):
        return log_L - (sum_log_L - log_L) / (num_models - 1.0)


class SelectorCV(ModelSelector):
    ''' 
    select best model based on average log Likelihood of cross-validation folds
    '''

    def select(self, n_splits=3, shuffle=True):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        n_splits = min(n_splits, len(self.sequences))
        kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=self.random_state)

        # for train_index, test_index in kf.split(self.X):
        split_data = [(train_idx, test_idx) for train_idx, test_idx in kf.split(self.sequences)]
        components_models_dict = {}
        for num_components in self.n_components:
            for train_idx, test_idx in split_data:
                model = self.base_model(num_components)
                if model is None:
                    break
                test_idx = list(test_idx)
                train_idx = list(train_idx)
                test_data, test_data_lengths = combine_sequences(test_idx, self.sequences)
                train_data, train_data_lengths = combine_sequences(train_idx, self.sequences)
                try:
                    model.fit(train_data, train_data_lengths)

                    score = model.score(test_data, test_data_lengths)

                    score_model_list = components_models_dict.get(num_components, [])

                    score_model_list.append((score, model))

                    components_models_dict[num_components] = score_model_list
                except Exception as e:
                    pass

        average_score_num_components_list = []
        for num_components in self.n_components:
            score_model_list = components_models_dict.get(num_components, None)
            if score_model_list is None:
                continue

            # scores = list(map(lambda score, model: model, score_model_list))
            scores = [score for score, model in score_model_list]
            average_score = np.average(scores)
            average_score_num_components_list.append((average_score, num_components))

        if not average_score_num_components_list:
            return None

        best_avg_score, num_components = sorted(average_score_num_components_list, key=lambda t: t[0], reverse=True)[0]

        best_model = self.base_model(num_components)

        best_model.fit(self.X, self.lengths)

        return best_model
