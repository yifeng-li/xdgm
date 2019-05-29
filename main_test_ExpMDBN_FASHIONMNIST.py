# test Exp-MDBN on Fashion-MNIST
#from __future__ import division
import numpy
import multimodaldbn
import classification as cl
import shutil
import os

workdir="/home/yifeng/research/deep/github/xdgm/"
os.chdir(workdir)

dir_data="./data/FASHIONMNIST/"

parent_dir_save="./results/MDBM/"
prefix="MDBM_FASHIONMNIST"

# load data
train_set_x=numpy.loadtxt(dir_data+"fashion-mnist_train.csv", dtype=int, delimiter=",",skiprows=1)
train_set_y=train_set_x[:,0]
train_set_x=train_set_x[:,1:]
train_set_x=train_set_x.transpose()

test_set_x=numpy.loadtxt(dir_data+"fashion-mnist_test.csv", dtype=int, delimiter=",",skiprows=1)
test_set_y=test_set_x[:,0]
test_set_x=test_set_x[:,1:]
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
visible_types=["Bernoulli","Multinomial"]
hidden_type="Bernoulli"
hidden_type_fixed_param=0
rng=numpy.random.RandomState(100)
M=[num_feat,10]
features=numpy.array(range(num_feat),dtype=str)
unique_cl=numpy.array(range(10),dtype=str)
features_allviews=[features,unique_cl]
normalization_method="None"

if visible_types[0]=="Bernoulli":  
    # normalization method
    normalization_method="scale"
    # parameter setting
    learn_rate_a_pretrain=[0.1,0.1,0.1]
    learn_rate_b_pretrain=[0.1,0.1,0.1] 
    learn_rate_W_pretrain=[0.1,0.1,0.1]
    learn_rate_a_train=[0.02,0.02,0.02]
    learn_rate_b_train=[0.02,0.02,0.02] 
    learn_rate_W_train=[0.02,0.02,0.02]
    change_rate_pretrain=0.95
    change_rate_train=0.95
    adjust_change_rate_at_pretrain=[6000,12000,15000]
    adjust_coef_pretrain=1.02
    adjust_change_rate_at_train=[6000,9000,12000]
    adjust_coef_train=1.02
    reg_lambda_a=0#0.5
    reg_alpha_a=1
    reg_lambda_b=0#0.5
    reg_alpha_b=1
    reg_lambda_W=0
    reg_alpha_W=1
    
    K_view=[[500,500],None]
    K_joint=[1000]
    sumout="auto"
    batch_size=100
    NMF=20
    pcdk_pretrain=20
    pcdk_train=5
    NS=100
    maxiter_pretrain=24#000
    maxiter_train=24#000
    change_every_many_iters=120#20#600
    init_chain_time=100
    visible_type_fixed_param=[0,1]
    reinit_a_use_data_stat=[True,False]
    fix_a_view=[False,False]
elif visible_types[0]=="Poisson":
    # normalization method
    normalization_method="None"
    # parameter setting
    # for unnormalized data
    # for Bernoulli hidden type, use 0.00001
    # for Binomial hidden type,  use 0.0000001
    K_view=[[100],None]
    K_joint=[110]
    sumout="auto"
    learn_rate_a_pretrain=[0.0001,0.1,0.01]
    learn_rate_b_pretrain=[0.0001,0.1,0.01]
    learn_rate_W_pretrain=[0.0001,0.1,0.01]
    learn_rate_a_train=[0.00003,0.01,0.003]
    learn_rate_b_train=[0.00003,0.01,0.003]
    learn_rate_W_train=[0.00003,0.01,0.003]
    reg_lambda_a=0#0.001
    reg_alpha_a=1
    reg_lambda_b=0#0.001
    reg_alpha_b=1
    reg_lambda_W=0#0.001
    reg_alpha_W=1
        
    batch_size=100
    NMF=100
    pcdk=20
    NS=100
    maxiter_pretrain=12000
    maxiter_train=12000
    change_rate_pretrain=0.9
    change_rate_train=0.9
    change_every_many_iters=600
    init_chain_time=100
    visible_type_fixed_param=[0,1]
    reinit_a_use_data_stat=[True,False]
    fix_a_view=[False,False]

elif visible_types[0]=="Gaussian":
    # normalization method
    normalization_method="tfidf"
    # parameter setting
    K_view=[None,None]
    K_joint=[500]
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
    visible_type_fixed_param=[0,1]
    reinit_a_use_data_stat=[True,False]


# normalization method
if normalization_method=="binary":
    # discret data
    ind=train_set_x<=50
    train_set_x[ind]=0
    train_set_x[numpy.logical_not(ind)]=1
    ind=test_set_x<=50
    test_set_x[ind]=0
    test_set_x[numpy.logical_not(ind)]=1

if normalization_method=="scale":
    train_set_x=train_set_x/255
    test_set_x=test_set_x/255

# initialize a model
model_mdbn=multimodaldbn.multimodaldbn(num_views=2, features=features_allviews, M=M, K_view=K_view, K_joint=K_joint, visible_types=visible_types, visible_type_fixed_param=visible_type_fixed_param, fix_a_view=fix_a_view, tol_poisson_max=8, rng=rng)

# create a folder to save the results
dir_save=model_mdbn.make_dir_save(parent_dir_save, prefix)
# make a copy of this script in dir_save
shutil.copy(workdir+"main_test_ExpMDBN_FASHIONMNIST.py", dir_save)
shutil.copy(workdir+"restricted_boltzmann_machine.py", dir_save)
shutil.copy(workdir+"helmholtz_machine.py", dir_save)
shutil.copy(workdir+"deep_boltzmann_machine.py", dir_save)
shutil.copy(workdir+"multimodaldbm.py", dir_save)
shutil.copy(workdir+"multimodaldbn.py", dir_save)

# pretrain
pretrain_time=model_mdbn.pretrain(X=[train_set_x,train_set_y01], just_pretrain_HMDBM=True, batch_size=batch_size, NMF=NMF, pcdk=pcdk_pretrain, NS=NS, maxiter=maxiter_pretrain, learn_rate_a=learn_rate_a_pretrain, learn_rate_b=learn_rate_b_pretrain, learn_rate_W=learn_rate_W_pretrain, change_rate=change_rate_pretrain, adjust_change_rate_at=adjust_change_rate_at_pretrain, adjust_coef=adjust_coef_pretrain, change_every_many_iters=change_every_many_iters, init_chain_time=init_chain_time, train_subset_size_for_compute_error=100, valid_subset_size_for_compute_error=100, track_reconstruct_error=True, track_free_energy=True, reinit_a_use_data_stat=reinit_a_use_data_stat, if_plot_error_free_energy=True, dir_save=dir_save, prefix="pretrain_HM", figwidth=5, figheight=3)

##############################################################
# sampling
rand_init=False
sampling_time=3
num_col=10
num_sampled_points=10*num_col
pcdk_sampling=100#0
NMF_sampling=20#0
prefix="pretrain_MDBN_FASHIONMINST"

chainX,chainXM=model_mdbn.generate_x(pcdk=10*pcdk_sampling, NMF=NMF_sampling, NS=num_sampled_points, X0=None, init_sampling=True)

for s in range(sampling_time):
    chainX,chainXM=model_mdbn.generate_x(pcdk=pcdk_sampling, init_sampling=False)
    # plot sampled data
    sample_set_x_3way=numpy.reshape(chainXM[0],newshape=(28,28,num_sampled_points))
    filename=dir_save+prefix+"_generated_samples_randinit_"+str(rand_init)+"_"+str(s)+"_V0.pdf"
    cl.plot_image_subplots(filename, data=sample_set_x_3way, figwidth=6, figheight=6, colormap=None, num_col=num_col, wspace=0.01, hspace=0.001)
    #filename=dir_save+"mdbn_MNIST_generated_samples_randinit_"+str(rand_init)+"_"+str(s)+"_V1_1D.txt"
    #sample_set_y_1way=numpy.argmax(chainXM[1],axis=0)
    #numpy.savetxt(filename,sample_set_y_1way,fmt="%s",delimiter="\t")    
    filename=dir_save+prefix+"_generated_samples_randinit_"+str(rand_init)+"_"+str(s)+"_V1_2D.txt"
    sample_set_y_2way=numpy.reshape(numpy.argmax(chainXM[1],axis=0),newshape=(num_col,int(num_sampled_points/num_col)))
    numpy.savetxt(filename,sample_set_y_2way,fmt="%s",delimiter="\t")
    filename=dir_save+prefix+"_generated_samples_randinit_"+str(rand_init)+"_"+str(s)+"_V1_prob.txt"
    numpy.savetxt(filename,chainXM[1].transpose(),fmt="%0.4f",delimiter="\t")

#######################################################################
# given one view, sample another view: given classes, generate image
method="Gibbs_sampling" # "mean_field", "Gibbs_sampling"
num_col=10
num_sampled_points=10*num_col
pcdk_sampling=1000
NMF_sampling=200
 
test_set_x_sim,test_set_y_sim,_,_=cl.sampling(train_set_x.transpose(),train_set_y,max_size_given=num_col,rng=rng)
test_set_x_sim=test_set_x_sim.transpose()
test_set_y_sim01,_=cl.membership_vector_to_indicator_matrix(z=test_set_y_sim, z_unique=range(10))
test_set_y_sim01=test_set_y_sim01.transpose()
filename=dir_save+prefix+"_given_class_sample_image_V0_real_initial.pdf"
test_set_x_sim_3way=numpy.reshape(test_set_x_sim, newshape=(28,28,num_sampled_points))
cl.plot_image_subplots(filename, data=test_set_x_sim_3way, figwidth=6, figheight=6, colormap=None, num_col=num_col, wspace=0.01, hspace=0.001)

X_view,XM_view=model_mdbn.generate_missing_x(pcdk=pcdk_sampling, NMF=NMF_sampling, X=[test_set_x_sim,test_set_y_sim01], missing_view=[True,False], method=method, init_sampling=True) # "mean_field" or "Gibbs_sampling"

filename=dir_save+prefix+"_given_class_sample_image_"+method+"_V0.pdf"
test_set_x_sim_3way=numpy.reshape(XM_view[0], newshape=(28,28,num_sampled_points))
cl.plot_image_subplots(filename, data=test_set_x_sim_3way, figwidth=6, figheight=6, colormap=None, num_col=num_col, wspace=0.01, hspace=0.001)
#filename=dir_save+"mdbn_MNIST_given_class_sample_image_"+method+"_V1_1D.txt"
#numpy.savetxt(filename,test_set_y_sim,fmt="%s",delimiter="\t")
filename=dir_save+prefix+"_given_class_sample_image_"+method+"_V1_2D.txt"
test_subset_y_sim_2d=numpy.reshape(test_set_y_sim,newshape=(num_col,num_sampled_points/num_col))
numpy.savetxt(filename,test_subset_y_sim_2d,fmt="%s",delimiter="\t")
filename=dir_save+prefix+"_given_class_sample_image_"+method+"_V1_prob.txt"
numpy.savetxt(filename,XM_view[1].transpose(),fmt="%0.4f",delimiter="\t")
filename=dir_save+prefix+"_given_class_sample_image_"+method+"_V1_prob_rank.txt"
XM_view1_sort=numpy.argsort(XM_view[1].transpose(),axis=1)
XM_view1_sort=XM_view1_sort[:,::-1]
numpy.savetxt(filename,XM_view1_sort,fmt="%s",delimiter="\t")

#####################################################################
# given one view, sample another view: given picture, predict classes
method="mean_field" # "mean_field", "Gibbs_sampling"
num_col=10
num_sampled_points=10*num_col
pcdk_sampling=1000
NMF_sampling=200

X_view,XM_view=model_mdbn.generate_missing_x(pcdk=pcdk_sampling, NMF=NMF_sampling, X=[test_set_x,None], missing_view=[False,True], method=method, init_sampling=True) # "mean_field" or "Gibbs_sampling"

filename=dir_save+prefix+"_given_image_sample_class_"+method+"_V0.pdf"
test_set_x_3way=numpy.reshape(test_set_x[:,0:num_sampled_points], newshape=(28,28,num_sampled_points))
cl.plot_image_subplots(filename, data=test_set_x_3way, figwidth=6, figheight=6, colormap=None, num_col=num_col, wspace=0.01, hspace=0.001)
#filename=dir_save+"mdbn_MNIST_given_image_sample_class_"+method+"_V1_1D.txt"
test_subset_y_1d=numpy.argmax(XM_view[1],axis=0)
#numpy.savetxt(filename,test_subset_y_1d,fmt="%s",delimiter="\t")
filename=dir_save+prefix+"_given_image_sample_class_"+method+"_V1_2D.txt"
test_subset_y_2d=numpy.reshape(numpy.argmax(XM_view[1][:,0:num_sampled_points],axis=0),newshape=(num_col,num_sampled_points/num_col))
numpy.savetxt(filename,test_subset_y_2d,fmt="%s",delimiter="\t")
filename=dir_save+prefix+"_given_image_sample_class_"+method+"_V1_prob.txt"
numpy.savetxt(filename,XM_view[1].transpose(),fmt="%0.4f",delimiter="\t")
filename=dir_save+prefix+"_given_image_sample_class_"+method+"_V1_prob_rank.txt"
XM_view1_sort=numpy.argsort(XM_view[1].transpose(),axis=1)
XM_view1_sort=XM_view1_sort[:,::-1]
numpy.savetxt(filename,XM_view1_sort,fmt="%s",delimiter="\t")
# calculate performance
perf,conf_mat=cl.perform(test_set_y,test_subset_y_1d,unique_classes=z_unique)
# save performance
cl.save_perform(path=dir_save, filename=prefix+"_mean_performances_"+method+".txt", create_new_file=True, perf=perf, std=None, auroc=None, auroc_std=None, auprc=None, auprc_std=None, conf_mat=conf_mat, classes_unique=z_unique, pretraining_time=pretrain_time, training_time=None, test_time=None, stat_test=None)

#################################################################
#T=10000
#S=10
#model_mdbn.estimate_log_likelihood(X=[test_set_x,test_set_y01], NMF=100, base_rate_type="prior", beta=None, step_base=0.999, T=T, stepdist="even", S=S, sumout="auto", dir_save=dir_save, prefix=prefix+"_test_")    
#model_mdbn.estimate_log_likelihood(X=[train_set_x,train_set_y01], NMF=100, base_rate_type="prior", beta=None, step_base=0.999, T=T, stepdist="even", S=S, sumout="auto", dir_save=dir_save, prefix=prefix+"_train_")
#
#print("result saved in: " + dir_save)


# train
train_time=model_mdbn.train(X=[train_set_x,train_set_y01], X_validate=None, batch_size=batch_size, NMF=NMF, pcdk=pcdk_train, NS=NS, maxiter=maxiter_train, learn_rate_a=learn_rate_a_train, learn_rate_b=learn_rate_b_train, learn_rate_W=learn_rate_W_train, change_rate=change_rate_train, adjust_change_rate_at=adjust_change_rate_at_train, adjust_coef=adjust_coef_train, change_every_many_iters=change_every_many_iters, init_chain_time=init_chain_time, track_reconstruct_error=True, valid_subset_size_for_compute_error=100, track_free_energy=True, if_plot_error_free_energy=True, dir_save=dir_save, prefix="train_MDBN", figwidth=5, figheight=3)

##############################################################
# sampling
rand_init=False
sampling_time=3
num_col=10
num_sampled_points=10*num_col
pcdk_sampling=1000
NMF_sampling=200
prefix="finetune_MDBN_FASHIONMINST"

chainX,chainXM=model_mdbn.generate_x(pcdk=10*pcdk_sampling, NMF=NMF_sampling, NS=num_sampled_points, X0=None, init_sampling=True)

for s in range(sampling_time):
    chainX,chainXM=model_mdbn.generate_x(pcdk=pcdk_sampling, init_sampling=False)
    # plot sampled data
    sample_set_x_3way=numpy.reshape(chainXM[0],newshape=(28,28,num_sampled_points))
    filename=dir_save+prefix+"_generated_samples_randinit_"+str(rand_init)+"_"+str(s)+"_V0.pdf"
    cl.plot_image_subplots(filename, data=sample_set_x_3way, figwidth=6, figheight=6, colormap=None, num_col=num_col, wspace=0.01, hspace=0.001)
    #filename=dir_save+"mdbn_MNIST_generated_samples_randinit_"+str(rand_init)+"_"+str(s)+"_V1_1D.txt"
    #sample_set_y_1way=numpy.argmax(chainXM[1],axis=0)
    #numpy.savetxt(filename,sample_set_y_1way,fmt="%s",delimiter="\t")    
    filename=dir_save+prefix+"_generated_samples_randinit_"+str(rand_init)+"_"+str(s)+"_V1_2D.txt"
    sample_set_y_2way=numpy.reshape(numpy.argmax(chainXM[1],axis=0),newshape=(num_col,int(num_sampled_points/num_col)))
    numpy.savetxt(filename,sample_set_y_2way,fmt="%s",delimiter="\t")
    filename=dir_save+prefix+"_generated_samples_randinit_"+str(rand_init)+"_"+str(s)+"_V1_prob.txt"
    numpy.savetxt(filename,chainXM[1].transpose(),fmt="%0.4f",delimiter="\t")

#######################################################################
# given one view, sample another view: given classes, generate image
method="Gibbs_sampling" # "mean_field", "Gibbs_sampling"
num_col=10
num_sampled_points=10*num_col
pcdk_sampling=1000
NMF_sampling=200
 
test_set_x_sim,test_set_y_sim,_,_=cl.sampling(train_set_x.transpose(),train_set_y,max_size_given=num_col,rng=rng)
test_set_x_sim=test_set_x_sim.transpose()
test_set_y_sim01,_=cl.membership_vector_to_indicator_matrix(z=test_set_y_sim, z_unique=range(10))
test_set_y_sim01=test_set_y_sim01.transpose()
filename=dir_save+prefix+"_given_class_sample_image_V0_real_initial.pdf"
test_set_x_sim_3way=numpy.reshape(test_set_x_sim, newshape=(28,28,num_sampled_points))
cl.plot_image_subplots(filename, data=test_set_x_sim_3way, figwidth=6, figheight=6, colormap=None, num_col=num_col, wspace=0.01, hspace=0.001)

X_view,XM_view=model_mdbn.generate_missing_x(pcdk=pcdk_sampling, NMF=NMF_sampling, X=[test_set_x_sim,test_set_y_sim01], missing_view=[True,False], method=method, init_sampling=True) # "mean_field" or "Gibbs_sampling"

filename=dir_save+prefix+"_given_class_sample_image_"+method+"_V0.pdf"
test_set_x_sim_3way=numpy.reshape(XM_view[0], newshape=(28,28,num_sampled_points))
cl.plot_image_subplots(filename, data=test_set_x_sim_3way, figwidth=6, figheight=6, colormap=None, num_col=num_col, wspace=0.01, hspace=0.001)
#filename=dir_save+"mdbn_MNIST_given_class_sample_image_"+method+"_V1_1D.txt"
#numpy.savetxt(filename,test_set_y_sim,fmt="%s",delimiter="\t")
filename=dir_save+prefix+"_given_class_sample_image_"+method+"_V1_2D.txt"
test_subset_y_sim_2d=numpy.reshape(test_set_y_sim,newshape=(num_col,num_sampled_points/num_col))
numpy.savetxt(filename,test_subset_y_sim_2d,fmt="%s",delimiter="\t")
filename=dir_save+prefix+"_given_class_sample_image_"+method+"_V1_prob.txt"
numpy.savetxt(filename,XM_view[1].transpose(),fmt="%0.4f",delimiter="\t")
filename=dir_save+prefix+"_given_class_sample_image_"+method+"_V1_prob_rank.txt"
XM_view1_sort=numpy.argsort(XM_view[1].transpose(),axis=1)
XM_view1_sort=XM_view1_sort[:,::-1]
numpy.savetxt(filename,XM_view1_sort,fmt="%s",delimiter="\t")

#####################################################################
# given one view, sample another view: given picture, predict classes
method="mean_field" # "mean_field", "Gibbs_sampling"
num_col=10
num_sampled_points=10*num_col
pcdk_sampling=1000
NMF_sampling=200

X_view,XM_view=model_mdbn.generate_missing_x(pcdk=pcdk_sampling, NMF=NMF_sampling, X=[test_set_x,None], missing_view=[False,True], method=method, init_sampling=True) # "mean_field" or "Gibbs_sampling"

filename=dir_save+prefix+"_given_image_sample_class_"+method+"_V0.pdf"
test_set_x_3way=numpy.reshape(test_set_x[:,0:num_sampled_points], newshape=(28,28,num_sampled_points))
cl.plot_image_subplots(filename, data=test_set_x_3way, figwidth=6, figheight=6, colormap=None, num_col=num_col, wspace=0.01, hspace=0.001)
#filename=dir_save+"mdbn_MNIST_given_image_sample_class_"+method+"_V1_1D.txt"
test_subset_y_1d=numpy.argmax(XM_view[1],axis=0)
#numpy.savetxt(filename,test_subset_y_1d,fmt="%s",delimiter="\t")
filename=dir_save+prefix+"_given_image_sample_class_"+method+"_V1_2D.txt"
test_subset_y_2d=numpy.reshape(numpy.argmax(XM_view[1][:,0:num_sampled_points],axis=0),newshape=(num_col,num_sampled_points/num_col))
numpy.savetxt(filename,test_subset_y_2d,fmt="%s",delimiter="\t")
filename=dir_save+prefix+"_given_image_sample_class_"+method+"_V1_prob.txt"
numpy.savetxt(filename,XM_view[1].transpose(),fmt="%0.4f",delimiter="\t")
filename=dir_save+prefix+"_given_image_sample_class_"+method+"_V1_prob_rank.txt"
XM_view1_sort=numpy.argsort(XM_view[1].transpose(),axis=1)
XM_view1_sort=XM_view1_sort[:,::-1]
numpy.savetxt(filename,XM_view1_sort,fmt="%s",delimiter="\t")
# calculate performance
perf,conf_mat=cl.perform(test_set_y,test_subset_y_1d,unique_classes=z_unique)
# save performance
cl.save_perform(path=dir_save, filename=prefix+"_mean_performances_"+method+".txt", create_new_file=True, perf=perf, std=None, auroc=None, auroc_std=None, auprc=None, auprc_std=None, conf_mat=conf_mat, classes_unique=z_unique, pretraining_time=pretrain_time, training_time=None, test_time=None, stat_test=None)

################################################################
T=10000
S=10
model_mdbn.estimate_log_likelihood(X=[test_set_x,test_set_y01], NMF=100, base_rate_type="prior", beta=None, step_base=0.999, T=T, stepdist="even", S=S, sumout="auto", dir_save=dir_save, prefix=prefix+"_test_")    
model_mdbn.estimate_log_likelihood(X=[train_set_x,train_set_y01], NMF=100, base_rate_type="prior", beta=None, step_base=0.999, T=T, stepdist="even", S=S, sumout="auto", dir_save=dir_save, prefix=prefix+"_train_")

print("result saved in: " + dir_save)



