import numpy as np
import ot
from ot.bregman import *
from ot.da import *

class RegularizedSinkhornTransport(BaseTransport):

    def __init__(self, reg_e=1., alpha=0.1, max_iter=1000, tol=10e-9, verbose=False, log=False,
                metric="sqeuclidean", norm=None, distribution_estimation=distribution_estimation_uniform,
                out_of_sample_map='ferradans', limit_max=np.infty):

        self.reg_e = reg_e
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.log = log
        self.metric = metric
        self.norm = norm
        self.limit_max = limit_max
        self.distribution_estimation = distribution_estimation
        self.out_of_sample_map = out_of_sample_map

    def fit(self, Xs=None, ys=None, Xt=None, yt=None, prev_gamma=None, iteration=1):

        super(RegularizedSinkhornTransport, self).fit(Xs=Xs, ys=ys, Xt=Xt, yt=yt)

        gamma = np.random.rand(Xs.shape[0], Xt.shape[0])
        gamma = gamma / np.sum(gamma, 1)[:, None]       
        gamma[~ np.isfinite(gamma)] = 0

        for k in range(5):

            print('Iter',k)
            grad_J = self.gradient_J(prev_gamma, Xs, gamma, Xt, iteration)
            
            log_gamma = np.log(gamma)
            log_gamma[~ np.isfinite(log_gamma)] = 0
            cost_k = self.alpha*self.cost_ + self.alpha*grad_J - log_gamma
            if self.log:
                gamma_t, log = sinkhorn(a=self.mu_s, b=self.mu_t, M=cost_k, reg=1+self.reg_e*self.alpha,
                                     numItermax=self.max_iter, stopThr=self.tol, verbose=self.verbose, log=self.log)

            else:
                gamma_t = sinkhorn(a=self.mu_s, b=self.mu_t, M=cost_k, reg=1+self.reg_e*self.alpha,
                                 numItermax=self.max_iter, stopThr=self.tol, verbose=self.verbose, log=self.log)

            gamma_t = gamma_t / np.sum(gamma_t, 1)[:, None]
            gamma_t[~ np.isfinite(gamma_t)] = 0
            #print(np.mean(np.square(gamma-gamma_t)))
            gamma = gamma_t

        
        if self.log:
            self.coupling_, self.log_ = gamma, log
        else:
            self.coupling_ = gamma
            self.log_ = dict()
        
        return self

    def gradient_J(self, gamma_prev, X_cur, gamma_cur, X_next, iter=1):
    
        if iter>0: print('Gamma prev', gamma_prev.shape)
        print('Gamma cur', gamma_cur.shape)
        print(X_cur.shape)
        print(X_next.shape)
        if iter==0:
            return 0
        else:
            #prev_samples = np.matmul(gamma_prev, X_cur)
            prev_samples = X_cur
            cur_samples = np.matmul(gamma_cur, X_next)

            print(prev_samples.shape)
            print(cur_samples.shape)

            # Downsample

            if (prev_samples.shape[0] > cur_samples.shape[0]):

                np.random.shuffle(prev_samples)
                prev_samples = prev_samples[:cur_samples.shape[0]]

            elif (prev_samples.shape[0] < cur_samples.shape[0]):

                indices = np.random.random_integers(0, prev_samples.shape[0]-1, cur_samples.shape[0])
                prev_samples = prev_samples[indices]

            print(prev_samples.shape, cur_samples.shape)
            grad = cur_samples - prev_samples
            grad = 2*np.matmul(grad,X_next.T)
            grad = grad/np.sum(grad)
            return grad





class RegularizedSinkhornTransportOTDA(BaseTransport):

    def __init__(self, reg_e=1., alpha=0.1, max_iter=1000, tol=10e-9, verbose=False, log=False,
                metric="sqeuclidean", norm=None, distribution_estimation=distribution_estimation_uniform,
                out_of_sample_map='ferradans', limit_max=np.infty):

        self.reg_e = reg_e
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.log = log
        self.metric = metric
        self.norm = norm
        self.limit_max = limit_max
        self.distribution_estimation = distribution_estimation
        self.out_of_sample_map = out_of_sample_map

    def fit(self, Xs=None, ys=None, Xt=None, yt=None, Xs_trans=None, ys_trans=None, prev_gamma=None, iteration=1):


        if check_params(Xs=Xs, Xt=Xt):

            # pairwise distance
            self.cost_ = dist(Xs_trans, Xt, metric=self.metric)
            if self.norm == 'mean':
                self.cost_ = self.cost_/np.linalg.norm(self.cost_, axis=1).reshape(-1, 1)
            else:
                self.cost_ = cost_normalization(self.cost_, self.norm)
            
            if (ys is not None) and (ys_trans is not None) and (yt is not None):

                if self.limit_max != np.infty:
                    self.limit_max = self.limit_max * np.max(self.cost_)

                # assumes labeled source samples occupy the first rows
                # and labeled target samples occupy the first columns
                classes = [c for c in np.unique(ys_trans) if c != -1]
                for c in classes:
                    idx_s = np.where((ys_trans != c) & (ys_trans != -1))
                    idx_t = np.where(yt == c)

                    # all the coefficients corresponding to a source sample
                    # and a target sample :
                    # with different labels get a infinite
                    for j in idx_t[0]:
                        self.cost_[idx_s[0], j] = self.limit_max

            # distribution estimation
            self.mu_s = self.distribution_estimation(Xs_trans)
            self.mu_t = self.distribution_estimation(Xt)

            # store arrays of samples
            self.xs_ = Xs_trans
            self.xt_ = Xt


        gamma = np.random.rand(Xs_trans.shape[0], Xt.shape[0])
        gamma = gamma / np.sum(gamma, 1)[:, None]       
        gamma[~ np.isfinite(gamma)] = 0

        for k in range(5):

            print('Iter',k)
            print(Xs_trans.shape)
            grad_J = self.gradient_J(prev_gamma, Xs, gamma, Xt, iteration)
            
            log_gamma = np.log(gamma)
            log_gamma[~ np.isfinite(log_gamma)] = 0
            cost_k = self.alpha*self.cost_ + self.alpha*grad_J - log_gamma
            if self.log:
                gamma_t, log = sinkhorn(a=self.mu_s, b=self.mu_t, M=cost_k, reg=1+self.reg_e*self.alpha,
                                     numItermax=self.max_iter, stopThr=self.tol, verbose=self.verbose, log=self.log)

            else:
                gamma_t = sinkhorn(a=self.mu_s, b=self.mu_t, M=cost_k, reg=1+self.reg_e*self.alpha,
                                 numItermax=self.max_iter, stopThr=self.tol, verbose=self.verbose, log=self.log)

            gamma_t = gamma_t / np.sum(gamma_t, 1)[:, None]
            gamma_t[~ np.isfinite(gamma_t)] = 0
            print(np.mean(np.square(gamma-gamma_t)))
            gamma = gamma_t

        
        if self.log:
            self.coupling_, self.log_ = gamma, log
        else:
            self.coupling_ = gamma
            self.log_ = dict()
        
        return self

    def gradient_J(self, gamma_prev, X_cur, gamma_cur, X_next, iter=1):
    
        if iter>0: print('Gamma prev', gamma_prev.shape)
        print('Gamma cur', gamma_cur.shape)
        print(X_cur.shape)
        print(X_next.shape)
        if iter==0:
            return 0
        else:
            prev_samples = np.matmul(gamma_prev, X_cur)
            cur_samples = np.matmul(gamma_cur, X_next)

            print(prev_samples.shape)
            print(cur_samples.shape)

            # Downsample

            if (prev_samples.shape[0] > cur_samples.shape[0]):

                np.random.shuffle(prev_samples)
                prev_samples = prev_samples[:cur_samples.shape[0]]

            elif (prev_samples.shape[0] < cur_samples.shape[0]):

                indices = np.random.random_integers(0, prev_samples.shape[0]-1, cur_samples.shape[0])
                prev_samples = prev_samples[indices]

            print(prev_samples.shape, cur_samples.shape)
            grad = cur_samples - prev_samples
            grad = 2*np.matmul(grad,X_next.T)
            grad = grad/np.sum(grad)
            return grad





