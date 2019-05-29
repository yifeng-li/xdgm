# test DBM and DBM on MNIST
import pickle, gzip
import numpy
import deep_boltzmann_machine
import classification as cl
import shutil
import os

workdir="/home/yifeng/research/deep/github/xdgm/"
os.chdir(workdir)

dir_data="./data/MNIST/"

parent_dir_save="./results/DBM/"
prefix="DBM_MNIST"

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
visible_type_fixed_param=None
rng=numpy.random.RandomState(100)
normalization_method="scale"
M=num_feat
K=[500,500]
batch_size=100
pcdk=20
NS=100
maxiter_pretrain=6000
maxiter_train=6000
learn_rate_a=0.8
learn_rate_b=0.8
learn_rate_W=0.8
change_rate=0.9
change_every_many_iters=240
init_chain_time=1000
NMF=100

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
model_dbm=deep_boltzmann_machine.deep_boltzmann_machine(M=M,K=K,visible_type=visible_type, visible_type_fixed_param=visible_type_fixed_param, rng=rng)
# create a folder to save the results
dir_save=model_dbm.make_dir_save(parent_dir_save,prefix,learn_rate_a, learn_rate_b, learn_rate_W, maxiter=maxiter_train, normalization_method="None")
# make a copy of this script in dir_save
shutil.copy(workdir+"main_test_ExpRBM_MNIST.py", dir_save)
shutil.copy(workdir+"main_test_DBM_MNIST.py", dir_save)

# pretraining
model_dbm.pretrain(X=train_set_x, batch_size=batch_size, pcdk=pcdk, NS=NS ,maxiter=maxiter_pretrain, learn_rate_a=learn_rate_a, learn_rate_b=learn_rate_b, learn_rate_W=learn_rate_W, change_rate=change_rate, change_every_many_iters=change_every_many_iters, init_chain_time=init_chain_time)

# train RBM
model_dbm.train(X=train_set_x, X_validate=test_set_x, batch_size=batch_size, NMF=NMF, pcdk=pcdk, NS=NS, maxiter=maxiter_train, learn_rate_a=learn_rate_a, learn_rate_b=learn_rate_b, learn_rate_W=learn_rate_W, change_rate=change_rate, change_every_many_iters=change_every_many_iters, init_chain_time=init_chain_time)

# sampling
rand_init=False
sampling_time=10

model_dbm.pcd_sampling(pcdk=20000, NS=100, X0=None, clamp_visible=False, persistent=True, rand_init_X=rand_init, rand_init_H=False, init_sampling=True)
for s in range(sampling_time):
    chainX,chainH,chainXM,chainXP,chainHM,chain_length_=model_dbm.pcd_sampling(pcdk=1000, init_sampling=False)
    # plot sampled data
    sample_set_x_3way=numpy.reshape(chainXM,newshape=(28,28,100))
    print(s)
    cl.plot_image_subplots(dir_save+"/fig_DBM_MNIST_generated_samples_randinit_"+str(rand_init)+"_"+str(s)+".pdf", data=sample_set_x_3way, figwidth=6, figheight=6, colormap=None, num_col=10, wspace=0.01, hspace=0.001)

# estimate log-likelihood
loglh_test,logZ_test,Z_std_test,energy_test,entropy_test=model_dbm.estimate_log_likelihood(X=test_set_x, NMF=100, base_type="prior", beta=None, step_base=0.999, T=10000, S=10)
loglh_train,logZ_train,Z_std_train,energy_train,entropy_train=model_dbm.estimate_log_likelihood(X=train_set_x, NMF=100, base_type="prior", beta=None, step_base=0.999, T=10000, S=10)
loglh_test,logZ_test,Z_std_test,energy_test,entropy_test=model_dbm.estimate_log_likelihood(X=test_set_x, NMF=100, base_type="uniform", beta=None, step_base=0.999, T=10000, S=10)
loglh_train,logZ_train,Z_std_train,energy_train,entropy_train=model_dbm.estimate_log_likelihood(X=train_set_x, NMF=100, base_type="uniform", beta=None, step_base=0.999, T=10000, S=10)
