import sys
from scipy.sparse import csr_matrix
import numpy as np
from Eval import Eval
from math import log, exp
import time
from imdb import IMDBdata
import matplotlib.pyplot as plt

class NaiveBayes:
    def __init__(self, data, ALPHA=1.0):
        """Initialize parameters"""
        self.ALPHA = ALPHA  # smoothing parameter
        self.data = data  # data object
        # self.data.X = self.data.X[:50, :] # slicing first 50 reviews
        # self.data.Y = self.data.Y[:50]    # slicing first 50 labels
        self.vocab_len = self.data.vocab.GetVocabSize()

        positive_indices = np.argwhere(self.data.Y == 1.0).flatten()   # indices of reviews with positive label
        negative_indices = np.argwhere(self.data.Y == -1.0).flatten()   # indices of reviews with negative label
        # X is a 2-dimensional array with bag of words as columns and
        # documents as row vectors
        self.positive_reviews = self.data.X[positive_indices, :]  # we are using positive indices as rows and
        # all the columns of X for collecting all positive reviews
        self.negative_reviews = self.data.X[negative_indices, :]  # we are using negative indices as rows and all the
        # columns of X for collecting all negative reviews

        # debug using self.positive_reviews.toarray() which converts sparse matrix into dense matrix where both
        # are 2-dimensional.

        # number of rows indicate the positive reviews and columns are bag of words in those reviews
        self.num_positive_reviews = self.positive_reviews.shape[0]
        self.num_negative_reviews = self.negative_reviews.shape[0]

        # create array of 0s with 1 row and bag of words count for count_positive and count_negative parameters
        self.count_positive = np.zeros(self.data.X.shape[1])
        self.count_negative = np.zeros(self.data.X.shape[1])
        self.total_positive_words = np.sum(self.positive_reviews)
        self.total_negative_words = np.sum(self.negative_reviews)

        # instantiate naive bayes parameters
        self.word_phi_positive = np.zeros(self.data.X.shape[1])
        self.word_phi_negative = np.zeros(self.data.X.shape[1])

        self.deno_pos = self.total_positive_words + (self.vocab_len * self.ALPHA)
        self.deno_neg = self.total_negative_words + (self.vocab_len * self.ALPHA)

        self.P_positive = 0
        self.P_negative = 0

        self.Train(data.X, data.Y)

    # Train model - X are instances, Y are labels (+1 or -1)
    # X and Y are sparse matrices
    def Train(self, X, Y):
        """Estimate Naive Bayes model parameters"""

        for i in range(self.data.X.shape[1]):
            self.count_positive[i] = np.sum(self.positive_reviews[:, i])  # filling each element word id with its count
            # in positive reviews by summing the column elements of 2d array of positive reviews
            self.count_negative[i] = np.sum(self.negative_reviews[:, i])  # filling each element word id with its count
            # in negative reviews by summing the column elements of 2d array of negative reviews
            self.word_phi_positive[i] = (self.count_positive[i] + self.ALPHA) / self.deno_pos
            self.word_phi_negative[i] = (self.count_negative[i] + self.ALPHA) / self.deno_neg

        self.P_positive = self.num_positive_reviews/(self.num_positive_reviews + self.num_negative_reviews)
        self.P_negative = self.num_negative_reviews/(self.num_negative_reviews + self.num_positive_reviews)

        return


    def PredictLabel(self, X, probThresh=0.5):
        """Implement Naive Bayes Classification and Predict labels for instances X
        :return Sparse matrix Y with predicted labels (+1 or -1)
        """

        pred_labels = []

        prob_pos = self.PredictProb(X, list(range(X.shape[0])))
        for prob in prob_pos:
            if prob > probThresh:
                pred_labels.append(1.0)
            else:
                pred_labels.append(-1.0)

        '''
        for i in range(X.shape[0]):
            non_zero_indices = X[i].nonzero()[1]
            pred_pos, pred_neg = self.Pred_Log_Prob(X, i, non_zero_indices)

            if pred_pos > pred_neg:            # Predict positive
                pred_labels.append(1.0)
            else:               # Predict negative
                pred_labels.append(-1.0)
        '''
        return pred_labels

    def Pred_Log_Prob(self, X, i, non_zero_indices):
        pred_pos = log(self.P_positive)
        pred_neg = log(self.P_negative)
        for j in range(len(non_zero_indices)):
            # Look at each feature
            pred_pos += X[i, j] * log(self.word_phi_positive[j])
            pred_neg += X[i, j] * log(self.word_phi_negative[j])
        return pred_pos, pred_neg

    def LogSum(self, logx, logy):   
        # Return log(x+y), avoiding numerical underflow/overflow.
        m = max(logx, logy)        
        return m + log(exp(logx - m) + exp(logy - m))


    def PredictProb(self, X, indexes):
        """
         Predict the probability of each indexed review in sparse matrix text
         of being positive and print results

        :param test: test data
        :param indexes: the rows/docs in test data for which we want to predict probability
        :return:
        """
        prob_pos = []
        for i in indexes:
            # Predicting the probability of the i_th review in test being positive review
            # Using the LogSum function to avoid underflow/overflow

            non_zero_indices = X[i].nonzero()[1] # indices of words with non-zero word counts for ith review
            pred_log_pos, pred_log_neg = self.Pred_Log_Prob(X,i,non_zero_indices)  # tuple of logs of predicted values
            # of labels
            log_deno = self.LogSum(pred_log_pos, pred_log_neg)  # denominator for calculating the probability of
            # each label for the ith review
            log_prob_pos = pred_log_pos - log_deno  # calculating log of probability of positive label for ith review

            predicted_prob_positive = exp(log_prob_pos) # calculating the probability of positive label for ith review
            prob_pos.append(predicted_prob_positive)
            '''
            log_prob_neg = pred_log_neg - log_deno  # calculating log of probability of negative label for ith review
            predicted_prob_negative = exp(log_prob_neg) # calculating the probability of negative label for ith review 
            
            
            if predicted_prob_positive > predicted_prob_negative:
                predicted_label = 1.0
            else:
                predicted_label = -1.0
            
            #print test.Y[i], test.X_reviews[i]
            # TO DO: Comment the line above, and uncomment the line below
            print(test.Y[i], predicted_label, predicted_prob_positive, predicted_prob_negative, test.X_reviews[i])
            '''
        return prob_pos


    def EvalPrecision(self,test, threshold=0.5, label=1.0):
        '''Evaluate performance on test data by calculating Precision'''
        Y_pred = np.array(self.PredictLabel(test.X))
        true_pred_indices = np.argwhere(test.Y == Y_pred)
        count = 0
        for i in true_pred_indices:
            if Y_pred[i] == label:
                count+=1
        return count/len(np.argwhere(Y_pred == label))


    def EvalRecall(self,test, threshold=0.5, label=1.0):
        '''Evaluate performance on test data by calculating Recall'''
        Y_pred = np.array(self.PredictLabel(test.X))
        true_pred_indices = np.argwhere(test.Y == Y_pred)
        count = 0
        for i in true_pred_indices:
            if Y_pred[i] == label:
                count += 1
        return count / len(np.argwhere(test.Y == label))

    def EvalPrecisionAndRecall(self, test, threshold=0.5, label=1.0):
        """Evaluate performance on test data by calculating Recall and Precision"""
        Y_pred = np.array(self.PredictProb(nb, test.X, threshold))
        true_pred_indices = np.argwhere(test.Y == Y_pred)
        count = 0
        for i in true_pred_indices:
            if Y_pred[i] == label:
                count+=1
        return (
            count / len(np.argwhere(Y_pred == label)),
            count / len(np.argwhere(test.Y == label))
        )

def plot_pr(pr_arr, label=''):
    precision = []
    recall = []

    for item in pr_arr:
        precision.append(item[0])
        recall.append(item[1])

    plt.figure()
    axs = plt.gca()
    axs.plot(recall, precision, 'o--')
    axs.set_title(f"Precision vs Recall {label} label")
    axs.set_xlabel("Recall")
    axs.set_ylabel("Precision")

if __name__ == "__main__":
    
    print("Reading Training Data")
    traindata = IMDBdata("%s/train" % sys.argv[1])
    print("Reading Test Data")
    testdata  = IMDBdata("%s/test" % sys.argv[1], vocab=traindata.vocab)    
    print("Computing Parameters")
    nb = NaiveBayes(traindata, float(sys.argv[2]))
    print("Evaluating")
    print("Test Accuracy: ", nb.Eval(testdata))

    pos_inf_ids = np.argsort(nb.word_phi_positive)[::-1][:20]  # ids of positive influential words
    neg_inf_ids = np.argsort(nb.word_phi_negative)[::-1][:20]  # ids of negative influential words

    thresholds = np.arange(0.1, 1.0, 0.1)
    prec_recall_pos = []
    prec_recall_neg = []
    for threshold in thresholds:
        prec_recall_pos.append(nb.EvalPrecisionAndRecall(testdata, threshold, 1.0))
        prec_recall_neg.append(nb.EvalPrecisionAndRecall(testdata, threshold, -1.0))

    plot_pr(prec_recall_pos, 'positive')
    plot_pr(prec_recall_neg, 'negative')
    plt.show()

    pos_str = ""
    for id in pos_inf_ids:
        pos_str = pos_str + f"{nb.data.vocab.GetWord(id)}  {nb.word_phi_positive[id]}  "
    print(pos_str)

    neg_str = ""
    for id in neg_inf_ids:
        neg_str = neg_str + f"{nb.data.vocab.GetWord(id)}  {nb.word_phi_negative[id]}  "
    print(neg_str)
