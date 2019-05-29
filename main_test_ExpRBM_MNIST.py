# test Exp-RBM on MNIST
#from __future__ import division
import pickle, gzip
import numpy
import restricted_boltzmann_machine
import classification as cl
import shutil
import os

workdir="/home/yifeng/research/deep/github/xdgm/"
os.chdir(workdir)

dir_data="./data/MNIST/"

parent_dir_save="./results/RBM/"
prefix="RBM_MNIST"

# Load the dataset
f = gzip.open(dir_data+"mnist.pkl.gz", "rb")
train_set_x, train_set_y, test_set_x, test_set_y = pickle.load(f, fix_imports=True, encoding="latin1", errors="strict")
f.close()

# train_set_x is a list of 28X28 matrices
train_set_x=numpy.array(train_set_x,dtype=int)
num_train_samples,num_rows,num_cols=train_set_x.shape
train_set_x=numpy.reshape(train_set_x,newshape=(num_train_samples,num_rows*num_cols))
train_set_y=numpy.array(train_set_y,dtype=int)

test_set_x=numpy.array(test_set_x,dtype=int)
num_test_samples,num_rows,num_cols=test_set_x.shape
test_set_x=numpy.reshape(test_set_x,newshape=(num_test_samples,num_rows*num_cols))
test_set_y=numpy.array(test_set_y,dtype=int)

train_set_x=train_set_x.transpose()
test_set_x=test_set_x.transpose()

print(train_set_x.shape)
print(train_set_y.shape)
print(test_set_x.shape)
print(test_set_y.shape)

# limit the number of training set
#train_set_x=train_set_x[:,0:10000]
#train_set_y=train_set_y[0:10000]

num_train=train_set_x.shape[1]
num_test=test_set_x.shape[1]

# convert train_set_y to binary codes
train_set_y01,z_unique=cl.membership_vector_to_indicator_matrix(z=train_set_y, z_unique=range(10))
train_set_y01=train_set_y01.transpose()
test_set_y01,_=cl.membership_vector_to_indicator_matrix(z=test_set_y, z_unique=range(10))
test_set_y01=test_set_y01.transpose()


num_feat=train_set_x.shape[0]
visible_type="Bernoulli"
hidden_type="Bernoulli"
hidden_type_fixed_param=1
rng=numpy.random.RandomState(100)
M=num_feat
normalization_method="None"

if visible_type=="Bernoulli":
    normalization_method="scale"
    # parameter setting
    K=16*40
    learn_rate_a=0.02
    learn_rate_b=0.02
    learn_rate_W=0.02
    change_rate=0.9
    adjust_change_rate_at=[3600]
    adjust_coef=1.02
    
    reg_lambda_a=0#0.5
    reg_alpha_a=1
    reg_lambda_b=0#0.5
    reg_alpha_b=1
    reg_lambda_W=0
    reg_alpha_W=1
    
    batch_size=100
    pcdk=10
    NS=100
    maxiter=12000
    change_every_many_iters=120
    init_chain_time=10
    visible_type_fixed_param=0
    reinit_a_use_data_stat=True
    
elif visible_type=="Poisson":
    normalization_method="None"
    # parameter setting
    K=500
    learn_rate_a=0.001
    learn_rate_b=0.001
    learn_rate_W=0.00001
    change_rate=0.95
    adjust_change_rate_at=[3600]
    adjust_coef=1.02
    
    reg_lambda_a=0#0.5
    reg_alpha_a=1
    reg_lambda_b=0#0.5
    reg_alpha_b=1
    reg_lambda_W=0
    reg_alpha_W=1
    
    batch_size=100
    pcdk=3
    NS=100
    maxiter=6000
    change_every_many_iters=120
    init_chain_time=10
    visible_type_fixed_param=0
    reinit_a_use_data_stat=True   
    
elif visible_type=="Gaussian":
    # normalize data
    normalization_method="scale"
    # parameter setting
    K=500
    learn_rate_a=[0.02,0.02]
    learn_rate_b=0.02
    learn_rate_W=0.02
    change_rate=0.95
    adjust_change_rate_at=[6000]
    adjust_coef=1.02
    
    reg_lambda_a=0#0.5
    reg_alpha_a=1
    reg_lambda_b=0#0.5
    reg_alpha_b=1
    reg_lambda_W=0
    reg_alpha_W=1    
    
    batch_size=100
    pcdk=10
    NS=100
    maxiter=12000
    change_every_many_iters=120

    init_chain_time=10
    visible_type_fixed_param=0
    reinit_a_use_data_stat=True

# normalization method
if normalization_method=="binary":
    # discret data
    threshold=0
    ind=train_set_x<=threshold
    train_set_x[ind]=0
    train_set_x[numpy.logical_not(ind)]=1
    ind=test_set_x<=threshold
    test_set_x[ind]=0
    test_set_x[numpy.logical_not(ind)]=1

if normalization_method=="scale":
    train_set_x=train_set_x/255
    test_set_x=test_set_x/255


# initialize a model
model_rbm=restricted_boltzmann_machine.restricted_boltzmann_machine(M=M,K=K,visible_type=visible_type, visible_type_fixed_param=visible_type_fixed_param, hidden_type=hidden_type, hidden_type_fixed_param=hidden_type_fixed_param, tie_W_for_pretraining_DBM_bottom=False, tie_W_for_pretraining_DBM_top=False, if_fix_vis_bias=False, a=None, rng=rng)
# create a folder to save the results
dir_save=model_rbm.make_dir_save(parent_dir_save, prefix, learn_rate_a, learn_rate_b, learn_rate_W, reg_lambda_W, reg_alpha_W, visible_type_fixed_param, hidden_type_fixed_param, maxiter, normalization_method)
# make a copy of this script in dir_save
shutil.copy(workdir+"main_test_ExpRBM_MNIST.py", dir_save)
shutil.copy(workdir+"restricted_boltzmann_machine.py", dir_save)
# plot some some training samples
train_set_x_3way=numpy.reshape(train_set_x[:,0:100],newshape=(28,28,100))
cl.plot_image_subplots(dir_save+"fig_"+prefix+"_train_samples.pdf", data=train_set_x_3way[:,:,0:100], figwidth=6, figheight=6, colormap="gray", num_col=10, wspace=0.01, hspace=0.001)

# train RBM
model_rbm.train(X=train_set_x, X_validate=test_set_x, batch_size=batch_size, pcdk=pcdk, NS=NS, maxiter=maxiter, learn_rate_a=learn_rate_a, learn_rate_b=learn_rate_b, learn_rate_W=learn_rate_W, change_rate=change_rate, adjust_change_rate_at=adjust_change_rate_at, adjust_coef=adjust_coef, reg_lambda_a=reg_lambda_a, reg_alpha_a=reg_alpha_a, reg_lambda_b=reg_lambda_b, reg_alpha_b=reg_alpha_b, reg_lambda_W=reg_lambda_W, reg_alpha_W=reg_alpha_W, change_every_many_iters=change_every_many_iters, init_chain_time=init_chain_time, train_subset_size_for_compute_error=100, valid_subset_size_for_compute_error=100, track_reconstruct_error=True, track_free_energy=False, reinit_a_use_data_stat=reinit_a_use_data_stat, if_plot_error_free_energy=True, dir_save=dir_save, prefix=prefix, figwidth=5, figheight=3)


# sampling using Gibbs sampling
_,_=model_rbm.generate_samples(NS=100, sampling_time=4, reinit=False, pcdk=1000, rand_init=True, dir_save=dir_save, prefix=prefix)

# get z code for 100 training samples
num_sample_per_cl=10
train_set_x100,train_set_y100,_,_=cl.truncate_sample_size(train_set_x.transpose(),train_set_y,max_size_given=num_sample_per_cl)
train_set_x100=train_set_x100.transpose()
sample_set_x_3way=numpy.reshape(train_set_x100,newshape=(28,28,100))
cl.plot_image_subplots(dir_save+"fig_"+prefix+"_actual_samples.pdf", data=sample_set_x_3way, figwidth=6, figheight=6, colormap="gray", aspect="equal", num_col=int(numpy.ceil(numpy.sqrt(NS))), wspace=0.01, hspace=0.001)
HM,HM_sorted=model_rbm.generate_samples_given_x(train_set_x_sub=train_set_x100, dir_save=dir_save, prefix=prefix)

# given z code to generate samples
num_cl=10
model_rbm.generate_samples_given_h(H0=HM, NS=num_sample_per_cl*num_cl, dir_save=dir_save, prefix=prefix+"_given_HM")

for s in range(num_cl):
    H0=numpy.tile(HM[:,s*num_sample_per_cl:(s+1)*num_sample_per_cl], num_sample_per_cl)
    model_rbm.generate_samples_given_h(H0=H0, NS=num_sample_per_cl*num_cl, sampling_time=1, dir_save=dir_save, prefix=prefix+"_"+str(s))


# get z code for 100 training samples
num_sample_per_cl=10
train_set_x100,train_set_y100,_,_=cl.truncate_sample_size(train_set_x.transpose(),train_set_y,max_size_given=num_sample_per_cl)
train_set_x100=train_set_x100.transpose()
sample_set_x_3way=numpy.reshape(train_set_x100,newshape=(28,28,100))
cl.plot_image_subplots(dir_save+"fig_"+prefix+"_actual_samples.pdf", data=sample_set_x_3way, figwidth=6, figheight=6, colormap="gray", aspect="equal", num_col=int(numpy.ceil(numpy.sqrt(NS))), wspace=0.01, hspace=0.001)


# estimate log-likelihood
T=10000
S=10
loglh_test,logZ_test,Z_std_test,mfe_test=model_rbm.estimate_log_likelihood(X=test_set_x, base_rate_type="prior", beta=None, step=0.999, T=T, S=S, reuse_mfe=False, train_or_test="test", dir_save=dir_save, prefix=prefix+"_priormethod")
loglh_train,logZ_train,Z_std_train,mfe_train=model_rbm.estimate_log_likelihood(X=train_set_x, base_rate_type="prior", beta=None, step=0.999, T=T, S=S, reuse_mfe=False, train_or_test="train", dir_save=dir_save, prefix=prefix+"_priormethod")
loglh_test,logZ_test,Z_std_test,mfe_test=model_rbm.estimate_log_likelihood(X=test_set_x, base_rate_type="uniform", beta=None, step=0.999, T=T, S=S, reuse_mfe=True, train_or_test="test", dir_save=dir_save, prefix=prefix+"_uniformmethod")
loglh_train,logZ_train,Z_std_train,mfe_train=model_rbm.estimate_log_likelihood(X=train_set_x, base_rate_type="uniform", beta=None, step=0.999, T=T, S=S, reuse_mfe=True, train_or_test="train", dir_save=dir_save, prefix=prefix+"_uniformmethod")


print("result saved in: " + dir_save)
    

