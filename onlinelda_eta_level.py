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
    return(psi(gamma) - psi(np.sum(gamma, 1))[:, np.newaxis])

class OnlineLDA(object):
    def __init__(self, K = 1, V = 1, alpha = 0.01, eta = 0.01, docs = None, method=None, loops=None, run_em=False, batch_size = 0, To = 64, kappa = 0.5, doc_iters = 0, doc_converge = 1e-6, isLogged = False):
        """
        K - number of topics
        V - number of distint terms in corpus
        method 
            random: random will initialize eta randomly based on data , data is required
            loaded: load from file, loops is required to specify which version we want to load
        loops - specify which model to load
        """
        self._alpha = alpha
        self._eta = eta
        self._docs = docs
        self._K = K
        self._V = V
        self._doc_converge = doc_converge
        self._em_iters_ran = 0
        self._isLogged = isLogged
        self._batch_size = batch_size
        self._To = To
        self._kappa = kappa 
        if self._docs: 
            self._gammas = np.zeros((len(self._docs), self._K))
            self._lambda = np.random.rand(self._K, self._V)

        self._savedir = os.path.dirname(os.path.realpath(__file__)) + '/generated_files' + '/model.K_' + str(self._K) + '.V_' + str(self._V) + '.alpha_' + str(self._alpha) + '.eta_' + str(self._eta) + '.batchsize_' + str(self._batch_size) + '.To_' + str(self._To) + '.kappa_' + str(self._kappa)
        
        if method == "random":
            if run_em:
                self.em_alg(batch_size = batch_size, doc_iters = doc_iters, verbose = True)
            
        if method == "loaded":
            self.load(loops)

    def em_alg(self, batch_size = 1, doc_iters = 10, verbose = False):
        if not os.path.exists(savedir):
            os.makedirs(savedir)
            
        D = len(self._docs)
        p_ts = []
        num_batches = int(D/batch_size)
        for t in range(num_batches):
            st = time.time()
            p_t = np.power((self._To + t), -self._kappa)
            if verbose: 
                print("minibatch: ", t)
                print("p_t: ", p_t)
            p_ts.append(p_t)
            old_lb = 0
            phi_sum = np.zeros((self._K, self._V))
            for offset in range(batch_size):
                index = t*batch_size + offset
                print("__doc ", index)
                if (index > len(self._docs) - 1): break
                doc = self._docs[index]
                var_phi, var_gamma = self.inference(doc, iters = doc_iters, verbose=False)
                self._gammas[index] = var_gamma
                # make indicator matrix in which: I[w_dn = j] = doc[w_dn] (count)
                Nd = len(doc)
                row = range(Nd)
                col = list(doc.keys())
                data = list(doc.values())
                ind = coo_matrix((data, (row, col)), shape=(Nd, self._V)) # (Nd, V)
                # cumulate for beta
                phi_sum += var_phi.T.dot(ind.toarray()) # (Nd, K).T x (Nd, V) = (K, V)

            lambda_t = self._eta + (D/batch_size) * phi_sum / np.sum(phi_sum, axis=1).reshape(self._K, 1) + 1e-100

            self._lambda = p_t * lambda_t + (1 - p_t) * self._lambda
            print("lambda ({}): {}".format(self._lambda.shape, self._lambda))
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
        K = self._K
        # init phi and gamma
        var_phi = np.ones((Nd, K)) * 1/K
        var_gamma = np.ones(K) * (np.sum(list(doc.values()))/K + self._alpha)
        old_gamma = 0  
        for i in range(iters):
            old_gamma = var_gamma
            E_log_theta = dirichlet_expectation(var_gamma) # (K,)
            
            w_indices = list(doc.keys())
            E_log_beta = psi(self._lambda[:,w_indices]) - psi(np.sum(self._lambda, axis=1))[:, np.newaxis] # (K, Nd)
            var_phi = np.exp(E_log_beta + E_log_theta.reshape(K, 1)).T # (K, Nd).T = (Nd, K)
            var_phi = var_phi / np.sum(var_phi, axis=1).reshape(Nd, 1) + 1e-100
            var_gamma = self._alpha + var_phi.T.dot(np.array(list(doc.values()))) # (Nd, K).T * (Nd, ) = (K, )
            
            meanchanged = np.mean(np.fabs(var_gamma - old_gamma))
            if (meanchanged <= self._doc_converge):
                if verbose: print("{} iterations, var_gamma: {}".format(i, var_gamma))
                break
        return var_phi, var_gamma

    def write_log(self, it, lb, changed, runtime):
        savedir = self._savedir
        if not os.path.exists(savedir):
            os.makedirs(savedir)
        with open(savedir + '/training.logs', 'a') as f:
            f.write(str(it) + '\t' + str(lb) + '\t' + str(changed) + '\t' + str(runtime) + '\n')

    def save(self, state):
        savedir = self._savedir
        if not os.path.exists(savedir):
            print("not exists")
            os.makedirs(savedir)
        print(savedir)
        pickle.dump(self._gammas, open(savedir + '/gamma.' + state, 'wb'))
        pickle.dump(self._lambda, open(savedir + '/lambda.' + state, 'wb'))
        pass

    def load(self, loops):
        savedir = self._savedir
        self._gammas = pickle.load(open(savedir + '/gamma.'+ str(loops), 'rb'))
        self._lambda = pickle.load(open(savedir + '/lambda.'+ str(loops), 'rb'))
        pass

    def print_topics(self, vocab, nwords = 20, verbose=False):
        indices = range(len(vocab))
        topic_no = 0
        savedir = self._savedir
        if not os.path.exists(savedir):
            os.makedirs(savedir)     
        f = open(savedir + '/topic_words.txt', 'w')
        for topic in self._lambda:
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
    with open(os.path.dirname(os.path.realpath(__file__)) + '/dataset/gro/voca_counts.txt', 'r') as f:
        line = f.readline()
        V = int(line.split()[-1])
    with open(os.path.dirname(os.path.realpath(__file__)) + '/dataset/gro/document_counts.txt', 'r') as f:
        line = f.readline()
        D = int(line.split()[-1])
        
    print("num docs: {}, num terms: {}".format(D, V))
    print("Loading sparse docs from file...")
    X = [{} for _ in range(D)]
    i = 0
    with open(os.path.dirname(os.path.realpath(__file__)) + '/dataset/gro/grolier-train.txt', 'r') as f:
        for line in f:
            if line == '': continue
            line = line.split(' ')
            for term in line[1:]:
                word_index, freq = term.split(':')
                X[i][int(word_index)] = int(freq)
            i += 1
    
    print("Running lda algorithm...")
    K = 100
    doc_iters = 20
    alpha = 0.01
    eta = 0.01
    batch_size = 500
    To = 64
    kappa = 0.7
    ldaModel = OnlineLDA(K = K, V = V, alpha = alpha, eta = eta, docs = X, method="random", run_em=True, batch_size = batch_size, To = To, kappa = kappa, doc_iters= doc_iters, isLogged = True)

    vocab = []
    with open(os.path.dirname(os.path.realpath(__file__)) + '/dataset/gro/grolier-voca.txt', 'r') as f:
        for line in f: vocab.append(line[:-1])

    ldaModel.print_topics(vocab = vocab, nwords = 20, verbose = False)

    time_stack.append(time.time())
    print("Done in {} seconds \n".format(time_stack[-1] - time_stack[0]))
    with open(savedir + '/logs', 'a') as f:
        f.write("{} terms, {} topics, {} EM iterations, {} max doc iterations, {} alpha: {} seconds\n".format(ldaModel._V, ldaModel._K, ldaModel._em_iters_ran, doc_iters, ldaModel._alpha, time_stack[-1] - time_stack[0]))
