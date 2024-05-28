import mkl
mkl.set_num_threads(1)
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tree import *
from cvar_tree_utilities import *
from sklearn.datasets import make_spd_matrix 
import gurobipy as gp
from gurobipy import GRB
#################################################################################################################################################################################################################
def solve_cvar(Y, A_mat = None, b_vec = None, alpha = None, verbose = False, weights = None, if_weight = False, impose_int = None): 

    n = Y.shape[0] 
    L = Y.shape[1] 
    if not if_weight:
        weights = np.ones(n)/n

    m = gp.Model()
    m.setParam("Threads", 1)
    if impose_int == 0:
        z = pd.Series(m.addVars(L, lb = 0, name = 'z'), index = range(L))
    elif impose_int == 1:
        z = pd.Series(m.addVars(L, vtype = GRB.BINARY, name = 'z'), index = range(L))
    w = m.addVar(lb = - GRB.INFINITY, name = 'auxiliary') 
    u = pd.Series(m.addVars(n, lb = 0, name = 'u'), index = range(n))
    m.update()

    risk = 1/(1 - alpha) * (weights * u).sum() + w 
    m.setObjective(risk, GRB.MINIMIZE)

    max_constraints = []
    for i in range(n):
        max_constraints.append(m.addConstr(u[i] >= Y[i, :].dot(z) - w)) 

    LP_constraints = []
    for i in range(A_mat.shape[0]):
        LP_constraints.append(m.addConstr(A_mat[i, :].dot(z) == b_vec[i]))

    m.update()

    if not verbose:
        m.setParam('OutputFlag', 0)

    m.optimize()

    try:
        decision = np.zeros(len(z)+1)
        decision[:-1] = np.array([zz.X for zz in z]) 
        decision[-1] = np.array(w.X) 
        obj = m.objVal

    except:
        if verbose:
            print("optimization error!")
        decision = None
        obj = np.inf

    return (decision, obj)
#################################################################################################################################################################################################################
def evaluate_cost(Y, Prob):

    n = Y.shape[0] 
    h = np.zeros(n)

    (dec_temp, _) = Prob
    dec = dec_temp[:-1]
    for scenario in range(n):
        h[scenario] = np.matmul(Y[scenario, :], dec)

    return (h)
#################################################################################################################################################################################################################
def solve_oracle(Y, A_mat = None, b_vec = None, verbose = False, impose_int = None):

    n = Y.shape[0] 
    h = np.zeros(n)

    for scenario in range(n):
        (_, obj_temp) = solve_hindsight(Y[scenario,:], A_mat = A_mat, b_vec = b_vec, verbose = verbose, impose_int = impose_int)
        h[scenario] = obj_temp

    return (h)
#################################################################################################################################################################################################################
def solve_hindsight(y, A_mat = None, b_vec = None, verbose = False, impose_int = None): 

    L = y.shape[0] 

    m = gp.Model()
    m.setParam("Threads", 1)
    z = pd.Series(m.addVars(L, lb = 0, name = 'z'), index = range(L))

    if impose_int == 0:
        z = pd.Series(m.addVars(L, lb = 0, name = 'z'), index = range(L))
    elif impose_int == 1:
        z = pd.Series(m.addVars(L, vtype = GRB.BINARY, name = 'z'), index = range(L))
    m.update()
    risk = y.dot(z)
    m.setObjective(risk, GRB.MINIMIZE)

    LP_constraints = []
    for i in range(A_mat.shape[0]):
        LP_constraints.append(m.addConstr(A_mat[i, :].dot(z) == b_vec[i]))
    m.update()

    if not verbose:
        m.setParam('OutputFlag', 0)

    m.optimize()

    try:
        decision = np.array([zz.X for zz in z])
        obj = m.objVal

    except:
        if verbose:
            print("optimization error!")
        decision = None
        obj = np.inf

    return (decision, obj)
#################################################################################################################################################################################################################
def get_gamma_star(Y, W, h_xhat, h_oracle, A_mat = None, b_vec = None, alpha = None, MaxIter = None, impose_int = None): 
        
    n = Y.shape[0] 
    gamma_l = 0
    gamma_r = 1

    lambda_gamma_l = get_lambda(h_xhat, h_oracle, gamma = gamma_l)
    lambda_gamma_r = get_lambda(h_xhat, h_oracle, gamma = gamma_r)

    CVaR_gamma_l = np.zeros(n)
    for scenario in range(n):
        (_, CVaR_gamma_l_temp) = solve_cvar_gamma(Y, lambda_gamma_l, A_mat = A_mat, b_vec = b_vec, alpha = alpha, verbose = False, weights = W[scenario,:], if_weight = True, impose_int = impose_int)
        CVaR_gamma_l[scenario] = CVaR_gamma_l_temp
    Qhat_gamma_l = CVaR_gamma_l.mean()

    CVaR_gamma_r = np.zeros(n)
    for scenario in range(n):
        (_, CVaR_gamma_r_temp) = solve_cvar_gamma(Y, lambda_gamma_r, A_mat = A_mat, b_vec = b_vec, alpha = alpha, verbose = False, weights = W[scenario,:], if_weight = True, impose_int = impose_int)
        CVaR_gamma_r[scenario] = CVaR_gamma_r_temp
    Qhat_gamma_r = CVaR_gamma_r.mean()
    
    if Qhat_gamma_l <= 0.0001:
        gamma_star = gamma_l
        Qhat_star = Qhat_gamma_l
        gamma = gamma_l*np.ones(MaxIter)
        Qhat = Qhat_gamma_l*np.ones(MaxIter)
    else:
        gamma_star = gamma_r
        Qhat_star = Qhat_gamma_r

        gamma = np.ones(MaxIter)
        gamma[0] = gamma_l
        gamma[1] = gamma_r 
        Qhat = 999*np.ones(MaxIter)
        Qhat[0] = Qhat_gamma_l 
        Qhat[1] = Qhat_gamma_r 

        index = 2
        Qhat_gamma_p = Qhat_gamma_l
            
        while (gamma_r - gamma_l) > 0.001 and abs(Qhat_gamma_p) > 0.0001: 
            
            gamma_p = (gamma_r + gamma_l)/2

            lambda_gamma_p = get_lambda(h_xhat, h_oracle, gamma = gamma_p)
            CVaR_gamma_p = np.zeros(n)
            for scenario in range(n): 
                (_, CVaR_gamma_p_temp) = solve_cvar_gamma(Y, lambda_gamma_p, A_mat = A_mat, b_vec = b_vec, alpha = alpha, verbose = False, weights = W[scenario,:], if_weight = True, impose_int = impose_int)
                CVaR_gamma_p[scenario] = CVaR_gamma_p_temp
            Qhat_gamma_p = CVaR_gamma_p.mean()

            if Qhat_gamma_p > 0:
                gamma_l = gamma_p
                Qhat_gamma_l = Qhat_gamma_p
            else:
                gamma_r = gamma_p
                Qhat_gamma_r = Qhat_gamma_p

            Qhat[index] = Qhat_gamma_p
            gamma[index] = gamma_p

            index = index + 1

        if abs(Qhat_gamma_p) <= 0.0001:
            gamma_star = gamma_p
            Qhat_star = Qhat_gamma_p 
        else:
            gamma_star = gamma_r
            Qhat_star = Qhat_gamma_r

    return (gamma_star, Qhat_star, gamma, Qhat) 
#################################################################################################################################################################################################################
def get_lambda(h_xhat, h_oracle, gamma = None):

    n = np.shape(h_oracle)[0]
    lambda_gamma = np.zeros(n)
    for scenario in range(n):
        lambda_gamma[scenario] = gamma*h_xhat[scenario] + (1 - gamma)*h_oracle[scenario]

    return (lambda_gamma) 
#################################################################################################################################################################################################################
def solve_cvar_gamma(Y, lambda_gamma, A_mat = None, b_vec = None, alpha = None, verbose = False, weights = None, if_weight = False, impose_int = None):

    n = Y.shape[0] 
    L = Y.shape[1]  

    if not if_weight:
        weights = np.ones(n)/n

    m = gp.Model()
    m.setParam("Threads", 1)
    if impose_int == 0:
        z = pd.Series(m.addVars(L, lb = 0, name = 'z'), index = range(L))
    elif impose_int == 1:
        z = pd.Series(m.addVars(L, vtype = GRB.BINARY, name = 'z'), index = range(L))
    u = pd.Series(m.addVars(n, lb = 0, name = 'u'), index = range(n)) 
    w = m.addVar(lb = - GRB.INFINITY, name = 'auxiliary') 
    m.update()

    risk = 1/(1 - alpha) * (weights * u).sum() + w 
    m.setObjective(risk, GRB.MINIMIZE)

    max_constraints = []
    for i in range(n):
        max_constraints.append(m.addConstr(u[i] >= Y[i, :].dot(z) - lambda_gamma[i] - w)) 

    LP_constraints = []
    for i in range(A_mat.shape[0]):
        LP_constraints.append(m.addConstr(A_mat[i, :].dot(z) == b_vec[i]))

    m.update()

    if not verbose:
        m.setParam('OutputFlag', 0)

    m.optimize()

    try:
        decision = np.zeros(len(z)+1)
        decision[:-1] = np.array([zz.X for zz in z]) 
        decision[-1] = np.array(w.X) 
        obj = m.objVal

    except:
        if verbose:
            print("optimization error!")
        decision = None
        obj = np.inf

    return (decision, obj)
#################################################################################################################################################################################################################
def get_cvar_gamma_dec_full(Y, W, h_xhat, h_oracle, A_mat = None, b_vec = None, alpha = None, gamma = None, impose_int = None):

    n = W.shape[0] 
    L = Y.shape[1]
    
    lambda_gamma = get_lambda(h_xhat, h_oracle, gamma = gamma)

    xstar_zeta = np.zeros((n,L))
    for scenario in range(n):
        (dec_temp, _) = solve_cvar_gamma(Y, lambda_gamma, A_mat = A_mat, b_vec = b_vec, alpha = alpha, verbose = False, weights = W[scenario,:], if_weight = True, impose_int = impose_int)
        xstar_zeta[scenario,:] = dec_temp[:-1]

    return (xstar_zeta)
#################################################################################################################################################################################################################
def get_cvar_dec_full(Y, W, A_mat = None, b_vec = None, alpha = None, if_weight = False, impose_int = None):

    n = W.shape[0] 
    L = Y.shape[1] 

    xstar_zeta = np.zeros((n,L))
    for scenario in range(n):
        (dec_temp, _) = solve_cvar(Y, A_mat = A_mat, b_vec = b_vec, alpha = alpha, verbose = False, weights = W[scenario,:], if_weight = True, impose_int = impose_int)
        xstar_zeta[scenario,:] = dec_temp[:-1]

    return (xstar_zeta)
#################################################################################################################################################################################################################
def evaluate_vhat(Y, SAA_tr, dec, A_mat = None, b_vec = None, impose_int = None):

    h_oracle_ts = solve_oracle(Y , A_mat = A_mat, b_vec = b_vec, verbose = False, impose_int = impose_int) 
    h_xhat = evaluate_cost(Y , SAA_tr) 

    n = Y.shape[0] 

    h = np.zeros(n)
    for scenario in range(n):
        h[scenario] = np.matmul(Y[scenario, :], dec[scenario, :])

    vhat = 1 - (h.mean() - h_oracle_ts.mean())/(h_xhat.mean() - h_oracle_ts.mean())

    return (vhat)
#################################################################################################################################################################################################################
def evaluate_cvar(h_init = None, alpha = 0.8):

    n = h_init.shape[0]
    p_init = np.ones(n)/n

    if alpha == 1: 
        cvar_arr = max(h_init[np.argwhere(p_init > 0)])
        cvar = cvar_arr.item() 

    else:
        ind = np.argsort(-h_init) 
        h = np.array([h_init[i] for i in ind])
        p = np.array([p_init[i] for i in ind])

        delta = 1 - alpha
        cump = np.cumsum(p)
        cump[-1] = 1

        i = np.argwhere(cump < delta)
        if i.size > 0:
            index_arr = max(i) + 1
            index = index_arr.item() 
        else:
            index = 0

        p[(index+1):] = 0

        if index == 0:
            p[index] = delta
        else:
            p[index] = delta - cump[index-1]

        cvar = np.matmul(np.transpose(p),h)/delta

    return (cvar)
#################################################################################################################################################################################################################
def train_tree(X_train_list, X_test_list, Y_train_list, Y_test_list, year_list = None, runs = None, alpha = None, A_mat = None, b_vec = None, seed = None):

    n_jobs = runs 
    np.random.seed(seed)  
    n_trees = 100
    weights_train = {}
    weights_test = {}
    for year in year_list:
        weights_train[year] = {} 
    for year in year_list:  
        weights_test[year] = {} 
    min_leaf_size = 10 
    max_depth = 100
    n_proposals = 365
    mtry = 65
    honesty = False
    balancedness_tol = 0.2
    lb = 0; ub = 1
    verbose = False
    bootstrap = True
    subsample_ratio = 1

    models_forest = {}
    for year in year_list:

        results_fit = Parallel(n_jobs=n_jobs, verbose = 3)(delayed(experiment_downtown_years)(Y_train_list[run], X_train_list[run], 
                Y_test_list[run], X_test_list[run], 
                A_mat = A_mat, b_vec = b_vec, alpha = alpha, ub = ub, lb = lb, 
                subsample_ratio = subsample_ratio, bootstrap = bootstrap, n_trees = n_trees, honesty = honesty, mtry = mtry, 
                min_leaf_size = min_leaf_size, max_depth = max_depth, n_proposals = n_proposals, 
                    balancedness_tol = balancedness_tol, verbose = verbose, seed = seed) for run in range(n_jobs))
        models_forest[year] = [res[0] for res in results_fit]

        N_train = np.shape(X_train_list)[1]
        N_test = np.shape(X_test_list)[1] 
        N_feutures = np.shape(X_train_list)[2]
        N_randomvar =  np.shape(Y_train_list)[2]

        for run in range(runs):
            weights_train[year][run] = {}
            for key in models_forest[year][run].keys():
                weights_train[year][run][key] = np.zeros((N_train,N_train)) 
                for i in range(N_train):
                    weights_train[year][run][key][i,:] = models_forest[year][run][key].get_weights(X_train_list[run][i,:]) 

        for run in range(runs):
            weights_test[year][run] = {}
            for key in models_forest[year][run].keys():
                weights_test[year][run][key] = np.zeros((N_test,N_train))      
                for i in range(N_test):
                    weights_test[year][run][key][i,:] = models_forest[year][run][key].get_weights(X_test_list[run][i,:]) 

    return(weights_train, weights_test)
######################################################################################################################################################################################################################
def generate_list(Y_list, no_covariates = None, train_size = None, validation_size = None, test_size = None, l_vl = None, h_vl = None, l_ts = None, h_ts = None, runs = None, seed = None):

    Y_list_np = pd.DataFrame(Y_list['year']).to_numpy()
    no_edges = Y_list_np.shape[1]

    mean_X = np.zeros(no_covariates) 
    std_X = np.ones(no_covariates) 

    mean_Y = np.zeros((no_edges, 3)) 
    for i in range(3): 
        if i == 0:
            mean_Y[:, 0] = np.mean(Y_list_np, axis = 0)
        elif i == 1:
            mean_Y[:, 1] = np.multiply(np.mean(Y_list_np, axis = 0), np.random.uniform(low = l_vl, high = h_vl, size = no_edges))
        elif i == 2:
            mean_Y[:, 2] = np.multiply(np.mean(Y_list_np, axis = 0), np.random.uniform(low = l_ts, high = h_ts, size = no_edges))
    std_Y = np.std(Y_list_np, axis = 0)

    n = no_edges + no_covariates
    covar = make_spd_matrix(n, random_state = seed)
    diag_covar = np.diag(np.diag(covar))
    sqrt_diag_covar = np.sqrt(diag_covar)
    inv_sqrt_diag_covar = np.linalg.inv(sqrt_diag_covar)
    cor = np.linalg.multi_dot([inv_sqrt_diag_covar, covar, inv_sqrt_diag_covar])
    info_std = np.concatenate((std_X[:, np.newaxis], std_Y[:, np.newaxis]), axis = 0) 
    std = np.diag(info_std[:,0])
    covar_var = np.linalg.multi_dot([std, cor, std])

    df = pd.DataFrame (info_std) 
    df.to_excel('ShortestPathProblem/info_std'+'.xlsx', index=False)
    df = pd.DataFrame (covar) 
    df.to_excel('ShortestPathProblem/covar'+'.xlsx', index=False)
    df = pd.DataFrame (cor) 
    df.to_excel('ShortestPathProblem/cor'+'.xlsx', index=False)
    df = pd.DataFrame (covar_var) 
    df.to_excel('ShortestPathProblem/covar_var'+'.xlsx', index=False)

    XY_list = []
    ###################################################################################################################
    for run in range(runs):
        for i in range(3): 

            info_mean = np.concatenate((mean_X[:, np.newaxis], mean_Y[:, i][:, np.newaxis]), axis = 0)
            
            if i == 0:
                XY_list_temp_i = np.random.multivariate_normal(info_mean[:,0], covar_var, size = train_size)
                XY_list_temp = XY_list_temp_i
            elif i == 1:
                XY_list_temp_i = np.random.multivariate_normal(info_mean[:,0], covar_var, size = validation_size)
                XY_list_temp = np.concatenate((XY_list_temp, XY_list_temp_i), axis = 0)
            elif i == 2:
                XY_list_temp_i = np.random.multivariate_normal(info_mean[:,0], covar_var, size = test_size)
                XY_list_temp = np.concatenate((XY_list_temp, XY_list_temp_i), axis = 0)

        XY_list.append(XY_list_temp)
    ###################################################################################################################
    X_tr_list = np.array(XY_list)[:, :train_size, :no_covariates]
    X_vl_list = np.array(XY_list)[:, train_size:train_size+validation_size, :no_covariates]
    X_ts_list = np.array(XY_list)[:, train_size+validation_size:, :no_covariates]
    X_vlts_list = np.array(XY_list)[:, train_size:, :no_covariates]

    Y_tr_list = np.array(XY_list)[:, :train_size, no_covariates:]
    Y_vl_list = np.array(XY_list)[:, train_size:train_size+validation_size, no_covariates:]
    Y_ts_list = np.array(XY_list)[:, train_size+validation_size:, no_covariates:]
    Y_vlts_list = np.array(XY_list)[:, train_size:, no_covariates:]

    return(X_tr_list, X_vl_list, X_ts_list, X_vlts_list, Y_tr_list, Y_vl_list, Y_ts_list, Y_vlts_list)
######################################################################################################################################################################################################################
def correlation_from_covariance(covariance):

    v = np.sqrt(np.diag(covariance))
    outer_v = np.outer(v, v)
    correlation = covariance / outer_v
    correlation[covariance == 0] = 0

    return (correlation)