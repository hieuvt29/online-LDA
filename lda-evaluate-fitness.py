from __future__ import division, print_function, unicode_literals
import numpy as np
import os, pickle, time
from scipy.special import gamma, digamma, gammaln, psi
from scipy.sparse import coo_matrix
from onlinelda_eta_level import OnlineLDA
import matplotlib.pyplot as plt

''' def predictive_distribution_docs_raw(docs, ratio, ldaModel, var_iters):
    predictive_prob = 0
    not_count = 0
    for (d, doc) in enumerate(docs):
        total_words = np.sum(list(doc.values()))
        # split doc into observed + hold-out
        obs = {}
        ho = {}
        for w in doc.keys():
            if ((np.sum(list(obs.values())) + doc[w]) / total_words < ratio):
                obs[w] = doc[w]
            else:
                ho[w] = doc[w]
        ho_num_words = np.sum(list(ho.values()))
        if(np.sum(list(ho.values())) == 0 and np.sum(list(obs.values())) == 0):
            not_count += 1
            continue
        # inference
        var_phi, var_gamma = ldaModel.inference(obs, iters = var_iters)
        mean_theta = var_gamma / var_gamma.sum()
        LPP_d = 0
        for w_new in ho.keys():
            word_prob = 0
            for k in range(ldaModel._K):
                word_prob += mean_theta[k] * mean_beta[k, w_new]
        
            LPP_d += np.log(word_prob)  * ho[w_new]

        LPP_d = LPP_d/ho_num_words

        predictive_prob += LPP_d
    print("Not count doc: ", not_count)
    return predictive_prob / (len(docs) - not_count) '''

def predictive_distribution_doc(obs, ho, ldaModel, var_iters = 50):
    obs_num_words = np.sum(list(obs.values()))
    ho_num_words = np.sum(list(ho.values()))
    if (ho_num_words == 0 or obs_num_words == 0):
        return None
    # inference
    var_phi, var_gamma = ldaModel.inference(obs, iters = var_iters)
    mean_theta = var_gamma / var_gamma.sum()
    LPP_d = 0
    for w_new in ho.keys():
        word_prob = 0
        for k in range(ldaModel._K):
            mean_beta = ldaModel._lambda[k] / ldaModel._lambda[k].sum()
            word_prob += mean_theta[k] * mean_beta[w_new]
    
        LPP_d += np.log(word_prob) * ho[w_new]

    LPP_d = LPP_d/ho_num_words
    return LPP_d

def predictive_distribution_docs(docs_obs, docs_ho, ldaModel, var_iters = 50):
    num_docs = len(docs_ho)
    LPP_docs = 0
    not_count = 0
    for i in range(num_docs):
        LPP_doc = predictive_distribution_doc(docs_obs[i], docs_ho[i], ldaModel, var_iters = var_iters)
        if (LPP_doc):
            LPP_docs += LPP_doc
        else:
            not_count += 1

    LPP_docs = LPP_docs/(num_docs - not_count)
    print("not count: ", not_count)
    return LPP_docs

def evaluate(test_batch, ldaModel, var_iters = 50):
    """
    test_batch : number specify test set preprocessed
    """
    docs_obs = []
    docs_ho = []
    i = 0
    with open(os.path.dirname(os.path.realpath(__file__)) + '/dataset/gro/data_test_'+ str(test_batch) +'_part_1.txt', 'r') as f:
        for line in f:
            if line == '': continue
            line = line.split(' ')
            docs_obs.append({})
            for term in line[1:]:
                word_index, freq = term.split(':')
                docs_obs[i][int(word_index)] = int(freq)
            i += 1

    i = 0
    with open(os.path.dirname(os.path.realpath(__file__)) + '/dataset/gro/data_test_'+ str(test_batch) +'_part_2.txt', 'r') as f:
        for line in f:
            if line == '': continue
            line = line.split(' ')
            docs_ho.append({})
            for term in line[1:]:
                word_index, freq = term.split(':')
                docs_ho[i][int(word_index)] = int(freq)
            i += 1

    return predictive_distribution_docs(docs_obs, docs_ho, ldaModel, var_iters = var_iters)

def general_evaluate(ldaModel, var_iters = 50):
    LPP_docs = 0
    for test_batch in range(1, 11, 1):
        val = evaluate(test_batch, ldaModel, var_iters = var_iters)
        LPP_docs += val
        print("val: ", val)
    return LPP_docs / 10
    
def main():
    time_stack = [time.time()]
    savedir = os.path.dirname(os.path.realpath(__file__)) + '/generated_files'
    print("Loading meta data...")
    with open(os.path.dirname(os.path.realpath(__file__)) + '/dataset/gro/meta.data', 'r') as f:
        for line in f:
            if line.split()[0] == "num_docs": 
                D = int(line.split()[1])
            if line.split()[0] == "num_terms": 
                V = int(line.split()[1])
    print("num docs: {}, num terms: {}".format(D, V))

    print("Evaluating lda algorithm...")
    K = 100
    alpha = 0.01
    eta = 0.01
    batchsize = 500
    var_iters = 25
    To = 64
    kappa = 0.7
    pre_prob_tests = []
    loopsArr = [46]
    for loops in loopsArr:
        print("Evaluate model with loops: ", loops)
        ldaModel = OnlineLDA(K = K, V = V, alpha = alpha, eta = eta, method="loaded", batch_size= batchsize, To = To, kappa= kappa, loops=loops)
        pre_prob = general_evaluate(ldaModel, var_iters = var_iters)
        # pre_prob = predictive_distribution_docs_raw(X, 0.9, ldaModel)
        pre_prob_tests.append(pre_prob)
    
    print(pre_prob_tests)
    with open(ldaModel._savedir + '/evaluate-result.txt', 'w') as f:
        for i in range(len(loopsArr)):
            f.write("{} : {}\n".format(loopsArr[i], pre_prob_tests[i]))

    plt.figure()
    plt.plot(loopsArr, pre_prob_tests, '-b')
    plt.xlabel("Number of trained minibatches")
    plt.ylabel("Log Predictive Probability")
    plt.show()

    
if __name__ == "__main__":
    st = time.time()
    main()
    print("run in: ", time.time() -st )