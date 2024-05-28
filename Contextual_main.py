import mkl
mkl.set_num_threads(1)
from Contextual_functions import *
############################################################################################################
# User settings
max_shift = 0.6 # 60% shift case (can be alternatively set to 0%, 20%, 30%, 40%, 50%)
impose_int = 0 # setting to 1 imposes integrality constraints on x
runs = 50 # number of instances
train_size = 400 # size of the training set
validation_size = 400 # size of the validation set
test_size = 1000 # size of the test set
no_covariates = 200 # size of the side information vector
############################################################################################################
# Parameter initialization
seed = 0
np.random.seed(seed)
alpha_stochforest = 0  
MaxIter = 1000
l_vl = 1
l_ts = 1
h_vl = 1 + max_shift
h_ts = 1 + max_shift
############################################################################################################
# alpha discretization
alpha_discretization = 20
alpha_vec = np.logspace(np.log10(0.01), np.log10(0.99), num = alpha_discretization) 
alpha_vec = np.sort(np.unique(np.concatenate([alpha_vec, np.linspace(0, 1, alpha_discretization, endpoint = False)]))) 
alpha_list = np.append('alpha', alpha_vec)
alpha_discretization = len(alpha_vec)
############################################################################################################
# Read network structure and two-year historical travel times from Kallus and Mao (2020)
year_list = ["year"] 
A_mat = pd.read_csv("ShortestPathProblem/A_downtwon_1221to1256.csv").to_numpy()
b_vec = pd.read_csv("ShortestPathProblem/b_downtwon_1221to1256.csv").to_numpy()
Y_list = {y: pd.read_csv("ShortestPathProblem/Y_" + str(y) + ".csv") for y in year_list}
############################################################################################################
# Generate the dataset 
(X_train_list, X_validation_list, X_test_list, X_vlts_list, Y_train_list, Y_validation_list, Y_test_list, Y_vlts_list) = generate_list(Y_list, no_covariates = no_covariates, train_size = train_size, validation_size = validation_size, test_size = test_size, l_vl = l_vl, h_vl = h_vl, l_ts = l_ts, h_ts = h_ts, runs = runs, seed = seed)
############################################################################################################
# Train random forests
(weights_tr, weights_vlts) = train_tree(X_train_list, X_vlts_list, Y_train_list, Y_vlts_list, year_list = year_list, runs = runs, alpha = alpha_stochforest, A_mat = A_mat, b_vec = b_vec, seed = seed)
for year in year_list:
    pickle.dump(weights_tr[year], open("ShortestPathProblem/weights_tr_"+year+".pkl", "wb"))
    pickle.dump(weights_vlts[year], open("ShortestPathProblem/weights_vlts_"+year+".pkl", "wb"))

w_train = weights_tr["year"]
w_validation = {}
w_test = {}
for run in range(runs):
    w_validation[run] = {} 
    w_test[run] = {}
    w_validation[run]["rf_rf"] = weights_vlts["year"][run]["rf_rf"][:validation_size, :]
    w_test[run]["rf_rf"] = weights_vlts["year"][run]["rf_rf"][validation_size:, :]
############################################################################################################
# Solve SAA 
SAA_train = Parallel(n_jobs = runs, verbose = 3)(delayed(solve_cvar)(Y_train_list[run], A_mat = A_mat, b_vec = b_vec, alpha = 0, verbose = False, weights = None, if_weight = False, impose_int = impose_int) for run in range(runs))
h_xhat_train = Parallel(n_jobs = runs, verbose = 3)(delayed(evaluate_cost)(Y_train_list[run], SAA_train[run]) for run in range(runs))
h_oracle_train = Parallel(n_jobs = runs, verbose = 3)(delayed(solve_oracle)(Y_train_list[run], A_mat = A_mat, b_vec = b_vec, verbose = False, impose_int = impose_int) for run in range(runs))
############################################################################################################
# DRPCR

# find gamma* for all alpha values using the train set
gamma_star_outputs = {}
for alpha in alpha_vec:
    gamma_star_outputs[alpha] = Parallel(n_jobs = runs, verbose = 3)(delayed(get_gamma_star)(Y_train_list[run], w_train[run]["rf_rf"], h_xhat_train[run], h_oracle_train[run], A_mat = A_mat, b_vec = b_vec, alpha = alpha, MaxIter = MaxIter, impose_int = impose_int) for run in range(runs))                                                                             
gamma_star_train_v = {}
Qhat_star_train_v = {}
gamma_train_v = {}
Qhat_train_v = {}
for alpha in alpha_vec:
    gamma_star_train_v[alpha] = np.zeros(runs)
    Qhat_star_train_v[alpha] = np.zeros(runs)
    gamma_train_v[alpha] = np.zeros((MaxIter, runs))
    Qhat_train_v[alpha] = np.zeros((MaxIter, runs))
    for run in range(runs):
        (gamma_star_temp, Qhat_star_temp, gamma_temp, Qhat_temp) = gamma_star_outputs[alpha][run]
        gamma_star_train_v[alpha][run] = gamma_star_temp 
        Qhat_star_train_v[alpha][run] = Qhat_star_temp
        gamma_train_v[alpha][:, run] = gamma_temp
        Qhat_train_v[alpha][:, run] = Qhat_temp       
pickle.dump(gamma_star_train_v, open("ShortestPathProblem/gamma_star_train_v.pkl", "wb"))
pickle.dump(Qhat_star_train_v, open("ShortestPathProblem/Qhat_star_train_v.pkl", "wb"))
pickle.dump(gamma_train_v, open("ShortestPathProblem/gamma_train_v.pkl", "wb"))
pickle.dump(Qhat_train_v, open("ShortestPathProblem/Qhat_train_v.pkl", "wb"))


# get vhat decisions for the validation set
dec_DRPCR_alpha = {}
for alpha in alpha_vec:
    dec_DRPCR_alpha[alpha] = Parallel(n_jobs = runs, verbose = 3)(delayed(get_cvar_gamma_dec_full)(Y_train_list[run], w_validation[run]["rf_rf"], h_xhat_train[run], h_oracle_train[run], A_mat = A_mat, b_vec = b_vec, alpha = alpha, gamma = gamma_star_train_v[alpha][run], impose_int = impose_int) for run in range(runs))


# evaluate vhat over the validation set
vhat_validation = {}
for alpha in alpha_vec:
    vhat_validation[alpha] = Parallel(n_jobs = runs, verbose = 3)(delayed(evaluate_vhat)(Y_validation_list[run], SAA_train[run], dec_DRPCR_alpha[alpha][run], A_mat = A_mat, b_vec = b_vec, impose_int = impose_int) for run in range(runs))

matrix_vhat_validation = np.zeros((alpha_discretization, runs)) 
for alpha in range(alpha_discretization):
    for run in range(runs):
        matrix_vhat_validation[alpha, run] = vhat_validation[alpha_vec[alpha]][run] 
pickle.dump(matrix_vhat_validation, open("ShortestPathProblem/matrix_vhat_validation.pkl", "wb"))


# determine alpha*
win_index_vhat = matrix_vhat_validation.argmax(axis = 0)        
df = pd.DataFrame (win_index_vhat) 
df.to_excel('ShortestPathProblem/win_index_vhat.xlsx', index=False)

# get optimal vhat decisions
dec_DRPCR = Parallel(n_jobs = runs, verbose = 3)(delayed(get_cvar_gamma_dec_full)(Y_train_list[run], w_test[run]["rf_rf"], h_xhat_train[run], h_oracle_train[run], A_mat = A_mat, b_vec = b_vec, alpha = alpha_vec[win_index_vhat[run]], gamma = gamma_star_train_v[alpha_vec[win_index_vhat[run]]][run], impose_int = impose_int) for run in range(runs))
pickle.dump(dec_DRPCR, open("ShortestPathProblem/dec_DRPCR.pkl", "wb"))
############################################################################################################
# get optimal CSO decisions
dec_CSO = Parallel(n_jobs = runs, verbose = 3)(delayed(get_cvar_dec_full)(Y_train_list[run], w_test[run]["rf_rf"], A_mat = A_mat, b_vec = b_vec, alpha = 0, if_weight = True, impose_int = impose_int) for run in range(runs))
pickle.dump(dec_CSO, open("ShortestPathProblem/dec_CSO.pkl", "wb"))
############################################################################################################
# DRCSO  

# get DRCSO decisions for the validation set
dec_cvar_loss_alpha = {}
for alpha in alpha_vec:
    dec_cvar_loss_alpha[alpha] = Parallel(n_jobs = runs, verbose = 3)(delayed(get_cvar_dec_full)(Y_train_list[run], w_validation[run]["rf_rf"], A_mat = A_mat, b_vec = b_vec, alpha = alpha, if_weight = True, impose_int = impose_int) for run in range(runs))


# evaluate vhat over the validation set
cvar_loss_validation = {}
for alpha in alpha_vec:
    cvar_loss_validation[alpha] = Parallel(n_jobs = runs, verbose = 3)(delayed(evaluate_vhat)(Y_validation_list[run], SAA_train[run], dec_cvar_loss_alpha[alpha][run], A_mat = A_mat, b_vec = b_vec, impose_int = impose_int) for run in range(runs))

matrix_cvar_loss_validation = np.zeros((alpha_discretization, runs)) 
for alpha in range(alpha_discretization):
    for run in range(runs):
        matrix_cvar_loss_validation[alpha, run] = cvar_loss_validation[alpha_vec[alpha]][run] 
pickle.dump(matrix_cvar_loss_validation, open("ShortestPathProblem/matrix_cvar_loss_validation.pkl", "wb"))


# determine alpha*
win_index_cvar_loss = matrix_cvar_loss_validation.argmax(axis = 0)
df = pd.DataFrame (win_index_cvar_loss) 
df.to_excel('ShortestPathProblem/win_index_cvar_loss.xlsx', index=False)

# get optimal DRCSO decisions
dec_DRCSO = Parallel(n_jobs = runs, verbose = 3)(delayed(get_cvar_dec_full)(Y_train_list[run], w_test[run]["rf_rf"], A_mat = A_mat, b_vec = b_vec, alpha =  alpha_vec[win_index_cvar_loss[run]], if_weight = True, impose_int = impose_int) for run in range(runs))
pickle.dump(dec_DRCSO, open("ShortestPathProblem/dec_DRCSO.pkl", "wb"))
############################################################################################################
# DRCRO

# get DRCRO decisions for the validation set
dec_DRCRO_alpha = {}
for alpha in alpha_vec:
    dec_DRCRO_alpha[alpha] = Parallel(n_jobs = runs, verbose = 3)(delayed(get_cvar_gamma_dec_full)(Y_train_list[run], w_validation[run]["rf_rf"], h_xhat_train[run], h_oracle_train[run], A_mat = A_mat, b_vec = b_vec, alpha = alpha, gamma = 0, impose_int = impose_int) for run in range(runs))

# evaluate vhat over the validation set
cvar_reg_validation = {}
for alpha in alpha_vec:
    cvar_reg_validation[alpha] = Parallel(n_jobs = runs, verbose = 3)(delayed(evaluate_vhat)(Y_validation_list[run], SAA_train[run], dec_DRCRO_alpha[alpha][run], A_mat = A_mat, b_vec = b_vec, impose_int = impose_int) for run in range(runs))

matrix_cvar_reg_validation = np.zeros((alpha_discretization, runs)) 
for alpha in range(alpha_discretization):
    for run in range(runs):
        matrix_cvar_reg_validation[alpha, run] = cvar_reg_validation[alpha_vec[alpha]][run] 
pickle.dump(matrix_cvar_reg_validation, open("ShortestPathProblem/matrix_cvar_reg_validation.pkl", "wb"))

# determine alpha*
win_index_cvar_reg = matrix_cvar_reg_validation.argmax(axis = 0)
df = pd.DataFrame (win_index_cvar_reg) 
df.to_excel('ShortestPathProblem/win_index_cvar_reg.xlsx', index=False)

# get optimal DRCRO decisions
dec_DRCRO = Parallel(n_jobs = runs, verbose = 3)(delayed(get_cvar_gamma_dec_full)(Y_train_list[run], w_test[run]["rf_rf"], h_xhat_train[run], h_oracle_train[run], A_mat = A_mat, b_vec = b_vec, alpha = alpha_vec[win_index_cvar_reg[run]], gamma = 0, impose_int = impose_int) for run in range(runs))
pickle.dump(dec_DRCRO, open("ShortestPathProblem/dec_DRCRO.pkl", "wb"))
############################################################################################################
# Evaluate the decisions over the test set
vhat_CSO = Parallel(n_jobs = runs, verbose = 3)(delayed(evaluate_vhat)(Y_test_list[run], SAA_train[run], dec_CSO[run], A_mat = A_mat, b_vec = b_vec, impose_int = impose_int) for run in range(runs))
vhat_DRCSO = Parallel(n_jobs = runs, verbose = 3)(delayed(evaluate_vhat)(Y_test_list[run], SAA_train[run], dec_DRCSO[run], A_mat = A_mat, b_vec = b_vec, impose_int = impose_int) for run in range(runs))
vhat_DRCRO = Parallel(n_jobs = runs, verbose = 3)(delayed(evaluate_vhat)(Y_test_list[run], SAA_train[run], dec_DRCRO[run], A_mat = A_mat, b_vec = b_vec, impose_int = impose_int) for run in range(runs))
vhat_DRPCR = Parallel(n_jobs = runs, verbose = 3)(delayed(evaluate_vhat)(Y_test_list[run], SAA_train[run], dec_DRPCR[run], A_mat = A_mat, b_vec = b_vec, impose_int = impose_int) for run in range(runs))
pickle.dump(np.concatenate((np.array(vhat_CSO)[:,np.newaxis], np.array(vhat_DRCSO)[:,np.newaxis], np.array(vhat_DRCRO)[:,np.newaxis], np.array(vhat_DRPCR)[:,np.newaxis]), axis = 1), open("ShortestPathProblem/vhat.pkl", "wb"))

with open("ShortestPathProblem/vhat.pkl", 'rb') as pickle_file:
    vhat = pickle.load(pickle_file)
df = pd.DataFrame (vhat) 
df.to_excel('ShortestPathProblem/vhat.xlsx', index=False)