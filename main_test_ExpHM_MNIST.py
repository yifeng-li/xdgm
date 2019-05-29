# test Exp-HM on MNIST
from __future__ import division
import pickle, gzip
import numpy
import helmholtz_machine
import classification as cl
import shutil
import os

workdir="/home/yifeng/research/deep/github/xdgm/"
os.chdir(workdir)

dir_data="./data/MNIST/"

parent_dir_save="./results/HM/"
prefix="HM_MNIST"

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
hidden_type_fixed_param=0
rng=numpy.random.RandomState(100)
M=num_feat
normalization_method="None"

if visible_type=="Bernoulli":  
    # normalization method
    normalization_method="scale"
    # parameter setting
    learn_rate_a_pretrain=0.1
    learn_rate_b_pretrain=[0.1,0.1,0.1] # can be a list
    learn_rate_W_pretrain=[0.1,0.1,0.1] # can be a list
    learn_rate_a_train=0.001
    learn_rate_b_train=[0.001,0.001,0.001] # can be a list
    learn_rate_W_train=[0.001,0.001,0.001] # can be a list
    change_rate_pretrain=0.9
    change_rate_train=0.85
    adjust_change_rate_at_pretrain=[6000,12000,15000]
    adjust_coef_pretrain=1.02
    adjust_change_rate_at_train=[6000,12000,15000]
    adjust_coef_train=1.02
    reg_lambda_a=0#0.5
    reg_alpha_a=1
    reg_lambda_b=0#0.5
    reg_alpha_b=1
    reg_lambda_W=0
    reg_alpha_W=1
    
    K=[500,500,1000]
    batch_size=100
    pcdk=20
    NS=100
    maxiter_pretrain=6000
    maxiter_train=3600
    change_every_many_iters=120
    init_chain_time=100
    visible_type_fixed_param=0
    reinit_a_use_data_stat=True
    
elif visible_type=="Poisson":
    # normalization method
    normalization_method="samecount"
    count=1000
    # parameter setting
    # for unnormalized data
    # for Bernoulli hidden type, use 0.00001
    # for Binomial hidden type,  use 0.0000001
    if hidden_type=="Bernoulli":
        K=[500,500]
        sumout="auto"
        learn_rate_a_pretrain=0.001
        learn_rate_b_pretrain=[0.001,0.01] 
        learn_rate_W_pretrain=[0.001,0.01]
        learn_rate_a_train=0.0003
        learn_rate_b_train=[0.0003,0.003] # can be a list
        learn_rate_W_train=[0.0003,0.003] # can be a list
        reg_lambda_a=0
        reg_alpha_a=1
        reg_lambda_b=0
        reg_alpha_b=1
        reg_lambda_W=0
        reg_alpha_W=1
    if hidden_type=="Binomial":
        K=250 # for Binomial K=20, for Bernoulli K=200
        sumout="auto"
        learn_rate_a=0.001
        learn_rate_b=0.001 
        learn_rate_W=0.001
        reg_lambda_a=0
        reg_alpha_a=1
        reg_lambda_b=0
        reg_alpha_b=1
        reg_lambda_W=0
        reg_alpha_W=1
        
    batch_size=100
    NMF=100
    pcdk=20
    NS=100
    maxiter_pretrain=6600
    maxiter_train=4400
    change_rate_pretrain=0.9
    change_rate_train=0.9
    change_every_many_iters=110
    init_chain_time=100
    visible_type_fixed_param=0
    reinit_a_use_data_stat=True

elif visible_type=="Gaussian":
    # normalization method
    normalization_method="tfidf"
    # parameter setting
    K=500
    learn_rate_a=[1,10]
    learn_rate_b=0.01
    learn_rate_W=0.1
    reg_lambda_a=[0,0]
    reg_alpha_a=[0,0]
    reg_lambda_b=0
    reg_alpha_b=0
    reg_lambda_W=0
    reg_alpha_W=0

    batch_size=100
    NMF=100
    pcdk=20
    NS=100
    maxiter_pretrain=300
    maxiter_train=300
    change_rate=0.8
    change_every_many_iters=20
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
    

# creat the object
model_hm=helmholtz_machine.helmholtz_machine(features=None, M=M, K=K, visible_type=visible_type, visible_type_fixed_param=visible_type_fixed_param, tol_poisson_max=8, rng=rng)
# create a folder to save the results
dir_save=model_hm.make_dir_save(parent_dir_save, prefix, learn_rate_a_pretrain, learn_rate_b_pretrain, learn_rate_W_pretrain, maxiter_pretrain+maxiter_train, normalization_method)
# make a copy of this script in dir_save
shutil.copy(workdir+"main_test_ExpHM_MNIST.py", dir_save)
shutil.copy(workdir+"restricted_boltzmann_machine.py", dir_save)
shutil.copy(workdir+"helmholtz_machine.py", dir_save)

# pretrain
model_hm.pretrain(X=train_set_x, batch_size=batch_size, pcdk=pcdk, NS=NS, maxiter=maxiter_pretrain, learn_rate_a=learn_rate_a_pretrain, learn_rate_b=learn_rate_b_pretrain, learn_rate_W=learn_rate_W_pretrain, change_rate=change_rate_pretrain, adjust_change_rate_at=adjust_change_rate_at_pretrain, adjust_coef=adjust_coef_pretrain, change_every_many_iters=change_every_many_iters, init_chain_time=init_chain_time, train_subset_size_for_compute_error=100, valid_subset_size_for_compute_error=100, track_reconstruct_error=True, track_free_energy=True, reinit_a_use_data_stat=reinit_a_use_data_stat, if_plot_error_free_energy=True, dir_save=dir_save, prefix="HM_pretrain", figwidth=5, figheight=3)

# sampling
sampling_time=10
sampling_NS=100
for s in range(sampling_time):
    Xg,XMg,Hg,HMr=model_hm.sample_xh_sleep(sampling_NS)
    # plot sampled data
    sample_set_x_3way=numpy.reshape(XMg,newshape=(28,28,100))
    print(s)
    cl.plot_image_subplots(dir_save+"/fig_"+prefix+"_pretrain_generated_samples_randinit_"+str(s)+".pdf", data=sample_set_x_3way, figwidth=6, figheight=6, colormap=None, num_col=10, wspace=0.01, hspace=0.001)

sampling_time=10
sampling_NS=100
sampling_num_iter=1000
Xg,XMg=model_hm.generate_x(NS=sampling_NS, num_iter=10*sampling_num_iter, init=True)
for s in range(sampling_time):
    Xg,XMg=model_hm.generate_x(num_iter=sampling_num_iter, init=False)
    # plot sampled data
    sample_set_x_3way=numpy.reshape(XMg,newshape=(28,28,100))
    print(s)
    cl.plot_image_subplots(dir_save+"/fig_"+prefix+"_Gibbs_pretrain_generated_samples_randinit_"+str(s)+".pdf", data=sample_set_x_3way, figwidth=6, figheight=6, colormap=None, num_col=10, wspace=0.01, hspace=0.001)



# train
model_hm.train(X=train_set_x, X_validate=None, batch_size=batch_size, maxiter=maxiter_train, learn_rate_a=learn_rate_a_train, learn_rate_b=learn_rate_b_train, learn_rate_W=learn_rate_W_train, change_rate=change_rate_train, adjust_change_rate_at=adjust_change_rate_at_train, adjust_coef=adjust_coef_train, change_every_many_iters=change_every_many_iters, init_chain_time=init_chain_time, train_subset_size_for_compute_error=100, valid_subset_size_for_compute_error=100, track_reconstruct_error=True, track_free_energy=True, if_plot_error_free_energy=True, dir_save=dir_save, prefix="HM_train", figwidth=5, figheight=3)

# sampling
sampling_time=10
sampling_NS=100
for s in range(sampling_time):
    Xg,XMg,Hg,HMr=model_hm.sample_xh_sleep(sampling_NS)
    # plot sampled data
    sample_set_x_3way=numpy.reshape(XMg,newshape=(28,28,100))
    print(s)
    cl.plot_image_subplots(dir_save+"/fig_"+prefix+"_finetune_generated_samples_randinit_"+str(s)+".pdf", data=sample_set_x_3way, figwidth=6, figheight=6, colormap=None, num_col=10, wspace=0.01, hspace=0.001)

sampling_time=10
sampling_NS=100
sampling_num_iter=1000
Xg,XMg=model_hm.generate_x(NS=sampling_NS, num_iter=10*sampling_num_iter, init=True)
for s in range(sampling_time):
    Xg,XMg=model_hm.generate_x(num_iter=sampling_num_iter, init=False)
    # plot sampled data
    sample_set_x_3way=numpy.reshape(XMg,newshape=(28,28,100))
    print(s)
    cl.plot_image_subplots(dir_save+"/fig_"+prefix+"_Gibbs_finetune_generated_samples_randinit_"+str(s)+".pdf", data=sample_set_x_3way, figwidth=6, figheight=6, colormap=None, num_col=10, wspace=0.01, hspace=0.001)



