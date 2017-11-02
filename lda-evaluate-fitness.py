from __future__ import division, print_function, unicode_literals
import numpy as np
import os, pickle, time
from scipy.special import gamma, digamma, gammaln, psi
from scipy.sparse import coo_matrix
from onlinelda import OnlineLDA
# import matplotlib.pyplot as plt


def predictive_distribution(docs, ratio, ldaModel):
    predictive_prob = 0
    not_count = 0
    for (d, doc) in enumerate(docs):
        total_words = np.sum(list(doc.values()))
        # split doc into observed + hold-out
        obs = {}
        ho = {}
        for w in doc.keys():
            if (np.sum(list(ho.values())) / total_words < ratio):
                ho[w] = doc[w]
            else:
                obs[w] = doc[w]
        ho_num_words = np.sum(list(ho.values()))
        if(np.sum(list(obs.values())) == 0):
            not_count += 1
            continue
        # inference
        var_phi, var_gamma = ldaModel.inference(obs, iters = 50)
        mean_theta = var_gamma / var_gamma.sum()
        LPP_d = 0
        for w_new in ho.keys():
            word_prob = 0
            for k in range(ldaModel.K):
                word_prob += mean_theta[k] * ldaModel.beta[k, w_new] * ho[w_new]
        
            LPP_d += np.log(word_prob)

        LPP_d = LPP_d/ho_num_words

        predictive_prob += LPP_d
    print("Not count doc: ", not_count)
    return predictive_prob / (len(docs) - not_count)

def main():
    time_stack = [time.time()]
    savedir = os.path.dirname(os.path.realpath(__file__)) + '/generated_files'
    print("Loading meta data...")
    with open('/media/hieu/DATA/ShareOS/PythonVirtualEnv/WorkSpace/test/topic model/LDA/lda-online/dataset/gro/meta.data', 'r') as f:
        for line in f:
            if line.split()[0] == "num_docs": 
                D = int(line.split()[1])
            if line.split()[0] == "num_terms": 
                V = int(line.split()[1])
    print("num docs: {}, num terms: {}".format(D, V))

    print("Loading sparse docs from file...")
    X = [{} for _ in range(D)]
    i = 0
    with open('/media/hieu/DATA/ShareOS/PythonVirtualEnv/WorkSpace/test/topic model/LDA/lda-online/dataset/gro/grolier_test_24k.txt', 'r') as f:
        for line in f:
            if line == '': continue
            line = line.split(' ')
            for term in line[1:]:
                word_index, freq = term.split(':')
                X[i][int(word_index)] = int(freq)
            i += 1
    
    print("Evaluating lda algorithm...")
    K = 100
    doc_iters = 50
    alpha = 0.1
    batchsize = 1000
    pre_prob_tests = []
    loopsArr = [23]
    for loops in loopsArr:
        print("Evaluate model with loops: ", loops)
        ldaModel = OnlineLDA(K = K, V = V, alpha = alpha, method="loaded", batch_size= batchsize ,loops=loops, doc_iters= doc_iters)
        pre_prob = predictive_distribution(X, 0.9, ldaModel)
        pre_prob_tests.append(pre_prob)
    print(pre_prob_tests)
    # plt.figure()
    # plt.plot(loopsArr, pre_prob_tests, '-b')
    # plt.xlabel("Number of trained minibatches")
    # plt.ylabel("Log Predictive Probability")
    # plt.show()

    
if __name__ == "__main__":
    main()