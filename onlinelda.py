from __future__ import division, print_function, unicode_literals
import numpy as np 
import os, time
from scipy.special import gammaln, digamma, psi
from scipy.sparse import coo_matrix
import _pickle as pickle

def dirichlet_expectation(gamma):
    """
    For a vector theta ~ Dir(gamma), computes E[log(theta)] given gamma.
    """
    if (len(gamma.shape) == 1):
        return(psi(gamma) - psi(np.sum(gamma)))
    return(psi(gamma) - psi(n.sum(gamma, 1))[:, np.newaxis])

class OnlineLDA(object):
    def __init__(self, K = 1, V = 1, alpha = 0.1, beta = None, docs = None, method=None, loops=None, run_em=False, batch_size = 0, To = 64, kappa = 0.5, doc_iters = 0, doc_converge = 1e-6, isLogged = False):
        """
        K - number of topics
        V - number of distint terms in corpus
        method 
            random: random will initialize beta randomly based on data , data is required
            loaded: load from file, loops is required to specify which version we want to load
        loops - specify which model to load
        """
        self.alpha = alpha
        self.beta = beta
        self.docs = docs
        self.K = K
        self.V = V
        self.doc_converge = doc_converge
        self.em_iters_ran = 0
        if self.docs: self.gammas = np.zeros((len(self.docs), self.K))
        self.isLogged = isLogged
        self.batch_size = batch_size
        self.To = To
        self.kappa = kappa 
        self.savedir = os.path.dirname(os.path.realpath(__file__)) + '/generated_files' + '/model.K_' + str(self.K) + '.V_' + str(self.V) + '.alpha_' + str(self.alpha) + '.batchsize_' + str(self.batch_size) + '.To_' + str(self.To) + '.kappa_' + str(self.kappa)
        
        if method == "random":
            self.initBeta()
            if run_em:
                self.em_alg(batch_size = batch_size, doc_iters = doc_iters, verbose = True)
            
        if method == "loaded":
            self.load(loops)

    def initBeta(self):
        self.beta = np.zeros((self.K, self.V))
        num_doc_per_topic = 5

        for i in range(num_doc_per_topic):
            rand_index = np.random.permutation(len(self.docs)).tolist()
            for k in range(self.K):
                d = rand_index[k]
                doc = self.docs[d]
                for n in doc.keys():
                    self.beta[k][n] += doc[n]
        self.beta += 1
        self.beta = self.beta / np.sum(self.beta, axis=1).reshape(self.K, 1)

    def em_alg(self, batch_size = 1, doc_iters = 10, verbose = False):
        if not os.path.exists(savedir):
            os.makedirs(savedir)
            
        beta_t = np.zeros((self.K, self.V))
        D = len(self.docs)
        p_ts = []
        num_batches = int(D/batch_size)
        for t in range(num_batches):
            st = time.time()
            p_t = np.power((self.To + t), -self.kappa)
            if verbose: 
                print("minibatch: ", t)
                print("p_t: ", p_t)
            p_ts.append(p_t)
            old_lb = 0
            phi_sum = np.zeros((self.K, self.V))
            for offset in range(batch_size):
                index = t*batch_size + offset
                if (index > len(self.docs) - 1): break
                doc = self.docs[index]
                var_phi, var_gamma = self.inference(doc, iters = doc_iters, verbose=False)
                self.gammas[index] = var_gamma
                # make indicator matrix in which: I[w_dn = j] = doc[w_dn] (count)
                Nd = len(doc)
                row = range(Nd)
                col = list(doc.keys())
                data = list(doc.values())
                ind = coo_matrix((data, (row, col)), shape=(Nd, self.V)) # (Nd, V)
                # cumulate for beta
                phi_sum += var_phi.T.dot(ind.toarray()) # (Nd, K).T x (Nd, V) = (K, V)

            beta_t = phi_sum / np.sum(phi_sum, axis=1).reshape(self.K, 1) + 1e-100

            self.beta = p_t * beta_t + (1 - p_t) * self.beta
            print("beta: ", self.beta)
            print("------ Run in: {} s ------".format(time.time() - st))

            # if self.isLogged: 
            #     self.write_log(it, lb, changed, time.time() - st)
            if (t + 1 == num_batches):
                self.save(state = str(t + 1))
            elif (t + 1) % 5 == 0:
                self.save(state = str(t + 1))
        

    def inference(self, doc, iters = 10, verbose=False):
        """
        doc - specify which doc to infer
        iters - maximum number of iterations to run if not converge yet
        var_phi, var_gamma - act as references to have not to initialize a new local variables
        """
        Nd = len(doc)
        K = self.K
        # init phi and gamma
        var_phi = np.ones((Nd, K)) * 1/K
        var_gamma = np.ones(K) * (np.sum(list(doc.values()))/K + self.alpha)
        old_gamma = 0
        for i in range(iters):
            old_gamma = var_gamma
            E_log_theta = dirichlet_expectation(var_gamma)

            w_indices = list(doc.keys())
            var_phi = self.beta[:, w_indices].T * np.exp(E_log_theta) # (K, Nd).T * (K, ) = (Nd, K)
            var_phi = var_phi / np.sum(var_phi, axis=1).reshape(Nd, 1) + 1e-100
            var_gamma = self.alpha + var_phi.T.dot(np.array(list(doc.values()))) # (Nd, K).T * (Nd, ) = (K, )
            
            meanchanged = np.mean(np.fabs(var_gamma - old_gamma))
            if (meanchanged <= self.doc_converge):
                if verbose: print("{} iterations, var_gamma: {}".format(i, var_gamma))
                break
        return var_phi, var_gamma

    def write_log(self, it, lb, changed, runtime):
        savedir = self.savedir
        if not os.path.exists(savedir):
            os.makedirs(savedir)
        with open(savedir + '/training.logs', 'a') as f:
            f.write(str(it) + '\t' + str(lb) + '\t' + str(changed) + '\t' + str(runtime) + '\n')

    def save(self, state):
        savedir = self.savedir
        if not os.path.exists(savedir):
            os.makedirs(savedir)
        pickle.dump(self.gammas, open(savedir + '/gamma.' + state, 'wb'))
        pickle.dump(self.beta, open(savedir + '/beta.' + state, 'wb'))
        pass

    def load(self, loops):
        savedir = self.savedir
        self.gammas = pickle.load(open(savedir + '/gamma.'+ str(loops), 'rb'))
        self.beta = pickle.load(open(savedir + '/beta.'+ str(loops), 'rb'))
        pass

    def print_topics(self, vocab, nwords = 20, verbose=False):
        indices = range(len(vocab))
        topic_no = 0
        savedir = self.savedir
        if not os.path.exists(savedir):
            os.makedirs(savedir)     
        f = open(savedir + '/topic_words.txt', 'w')
        for topic in self.beta:
            # print('topic {}'.format(topic_no))
            f.write('topic {}  '.format(topic_no))
            indices = sorted(indices, key = lambda x: topic[x], reverse=True)
            if verbose:
                topic = sorted(topic, reverse=True)
                for i in range(nwords):
                    # print('  {} * {}'.format(vocab[indices[i]],topic[i]), end=",")
                    f.write('  {} * {},'.format(vocab[indices[i]],topic[i]))
            else:
                for i in range(nwords):
                    # print('  {}'.format(vocab[indices[i]]), end=",")
                    f.write('  {},'.format(vocab[indices[i]]))
            topic_no = topic_no + 1
            # print("\n")
            f.write("\n")
        f.close()

if __name__ == "__main__":
    time_stack = [time.time()]
    savedir = os.path.dirname(os.path.realpath(__file__)) + '/generated_files'
    print("Loading meta data...")
    # with open(savedir + '/meta.data', 'r') as f:
    #     for line in f:
    #         if line.split()[0] == "num_docs": 
    #             D = int(line.split()[1])
    #         if line.split()[0] == "num_terms": 
    #             V = int(line.split()[1])
    with open('/media/hieu/DATA/ShareOS/PythonVirtualEnv/WorkSpace/test/topic model/LDA/lda-online/dataset/gro/voca_counts.txt', 'r') as f:
        line = f.readline()
        V = int(line.split()[-1])
    with open('/media/hieu/DATA/ShareOS/PythonVirtualEnv/WorkSpace/test/topic model/LDA/lda-online/dataset/gro/document_counts.txt', 'r') as f:
        line = f.readline()
        D = int(line.split()[-1])
        
    print("num docs: {}, num terms: {}".format(D, V))
    print("Loading sparse docs from file...")
    X = [{} for _ in range(D)]
    i = 0
    with open('/media/hieu/DATA/ShareOS/PythonVirtualEnv/WorkSpace/test/topic model/LDA/lda-online/dataset/gro/grolier-train.txt', 'r') as f:
        for line in f:
            if line == '': continue
            line = line.split(' ')
            for term in line[1:]:
                word_index, freq = term.split(':')
                X[i][int(word_index)] = int(freq)
            i += 1
    
    print("Running lda algorithm...")
    K = 100
    doc_iters = 50
    alpha = 0.1
    batch_size = 1000
    To = 64
    kappa = 0.5
    ldaModel = OnlineLDA(K = K, V = V, alpha = alpha, docs = X, method="random", run_em=True, batch_size = batch_size, To = To, kappa = kappa, doc_iters= doc_iters, isLogged = True)

    vocab = []
    with open('/media/hieu/DATA/ShareOS/PythonVirtualEnv/WorkSpace/test/topic model/LDA/lda-online/dataset/gro/grolier-voca.txt', 'r') as f:
        for line in f: vocab.append(line[:-1])

    ldaModel.print_topics(vocab = vocab, nwords = 20, verbose = False)

    time_stack.append(time.time())
    print("Done in {} seconds \n".format(time_stack[-1] - time_stack[0]))
    with open(savedir + '/logs', 'a') as f:
        f.write("{} terms, {} topics, {} EM iterations, {} max doc iterations, {} alpha: {} seconds\n".format(ldaModel.V, ldaModel.K, ldaModel.em_iters_ran, doc_iters, ldaModel.alpha, time_stack[-1] - time_stack[0]))
