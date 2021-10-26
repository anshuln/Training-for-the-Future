import numpy as np
import ot
from ot.da import SinkhornTransport
from regularized_ot import RegularizedSinkhornTransport, RegularizedSinkhornTransportOTDA

def naive_transport(X_source, Y_source, X_aug_list, Y_aug_list, X_target, Y_target):

    ot_sinkhorn = SinkhornTransport(reg_e=0.5, max_iter=100, norm="median", verbose=False)

    X_domain = [X_source]
    X_domain = X_domain + X_aug_list + [X_target]
    Y_domain = [Y_source]
    Y_domain = Y_domain + Y_aug_list + [Y_target]

    gamma = []
    for i in range(len(X_domain) - 1):

        print('Domain: %d' %i)
        ot_sinkhorn.fit(Xs=X_domain[i]+1e-6, ys=Y_domain[i], Xt=X_domain[-1]+1e-6, yt=Y_domain[-1])
    
    X_domain[i] = ot_sinkhorn.transform(X_domain[i])

    return X_domain[0], X_domain[1:-1], X_domain[-1]



def transform_samples(X_source, Y_source, X_aug_list, Y_aug_list, X_target, Y_target, time_reg=False):

    ot_sinkhorn = RegularizedSinkhornTransport(reg_e=0.5, alpha=10, max_iter=100, norm="median", verbose=False)

    X_domain = [X_source]
    X_domain = X_domain + X_aug_list + [X_target]
    Y_domain = [Y_source]
    Y_domain = Y_domain + Y_aug_list + [Y_target]

    print(len(X_domain))

    i=0

    print('Domain %d' % i)
    print('Shape', X_domain[i].shape)
    gamma = []
    for j in range(i+1,len(X_domain)):

        print('Transforming to %d' % j)
        if time_reg:
            if j==i+1: ot_sinkhorn.fit(Xs=X_domain[i]+1e-6, ys=Y_domain[i], Xt=X_domain[j]+1e-6, yt=Y_domain[j], iteration=0)
            else: ot_sinkhorn.fit(Xs=X_domain[i]+1e-6, ys=Y_domain[i], Xt=X_domain[j]+1e-6, yt=Y_domain[j], prev_gamma=gamma, iteration=1)
        else:   
            ot_sinkhorn.fit(Xs=X_domain[i]+1e-6, ys=Y_domain[i], Xt=X_domain[j]+1e-6, yt=Y_domain[j], iteration=0)
        gamma = ot_sinkhorn.coupling_
        X_domain[j] = ot_sinkhorn.transform(X_domain[i])

    return X_domain[0], X_domain[1:-1], X_domain[-1]


def transform_samples_iter_reg(X_source, Y_source, X_aug_list, Y_aug_list, X_target, Y_target, time_reg=True):

    ot_sinkhorn_r = RegularizedSinkhornTransport(reg_e=0.5, alpha=10, max_iter=100, norm="median", verbose=False)

    X_domain = [X_source]
    X_domain = X_domain + X_aug_list + [X_target]
    Y_domain = [Y_source]
    Y_domain = Y_domain + Y_aug_list + [Y_target]

    i=0
    

    print('Domain %d' % i)

    gamma = []
    
    for i in range(len(X_domain)-1):

        X_temp = X_domain[i]

        for j in range(i+1,len(X_domain)):

            print('Transforming to %d' % j)

            if time_reg:
                if j==i+1: ot_sinkhorn_r.fit(Xs=X_domain[i]+1e-6, Xt=X_domain[j]+1e-6, ys=Y_domain[i], yt = Y_domain[j], iteration=0)
                else: ot_sinkhorn_r.fit(Xs=X_temp+1e-6, Xt=X_domain[j]+1e-6, ys=Y_domain[i], yt=Y_domain[j], prev_gamma=gamma, iteration=1)
            else:
                ot_sinkhorn_r.fit(Xs=X_temp+1e-6, Xt=X_domain[j]+1e-6, ys=Y_domain[i], yt = Y_domain[j], iteration=0)   
            gamma = ot_sinkhorn_r.coupling_
            X_temp = ot_sinkhorn_r.transform(X_temp)
        
        X_domain[i] = X_temp

    #X_domain[i] = X_temp
    return X_domain 
    
def transform_samples_reg_otda(X_source, Y_source, X_aug_list, Y_aug_list,  X_target, Y_target):

    ot_sinkhorn_r = RegularizedSinkhornTransportOTDA(reg_e=0.5, max_iter=50, norm="median", verbose=False)

    X_domain = [X_source]
    X_domain = X_domain + X_aug_list + [X_target]
    Y_domain = [Y_source]
    Y_domain = Y_domain + Y_aug_list + [Y_target]

    for i in range(len(X_domain) - 1):

        print('-'*100)
        print('Domain %d' % i)
        print('-'*100)
        
        gamma = []
        X_temp = X_domain[i]
        Y_temp = Y_domain[i]

        for j in range(i+1, len(X_domain) - 1):

            print('-'*100)
            print('Transforming to %d' % j)
            print('-'*100)

            if j==i+1: ot_sinkhorn_r.fit(Xs=X_domain[j-1]+1e-6, ys=Y_domain[j-1], Xt=X_domain[j]+1e-6, yt=Y_domain[j], 
                                            Xs_trans=X_temp+1e-6, ys_trans=Y_domain[i], iteration=0)
            else: ot_sinkhorn_r.fit(Xs=X_domain[j-1]+1e-6, ys=Y_domain[j-1], Xt=X_domain[j]+1e-6, yt=Y_domain[j],
                                            Xs_trans=X_temp+1e-6, ys_trans=Y_domain[i], prev_gamma=gamma, iteration=1)
            gamma = ot_sinkhorn_r.coupling_
            X_temp = ot_sinkhorn_r.transform(X_temp)

        X_domain[i] = X_temp

    return X_domain[0], X_domain[1:-1], X_domain[-1]
