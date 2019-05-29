
#from __future__ import division
import numpy
import math
import restricted_boltzmann_machine
import classification as cl
import copy
import os
import time

class deep_boltzmann_machine:
    def __init__(self, features=None, M=None, K=None, visible_type="Bernoulli", visible_type_fixed_param=100, hidden_type="Bernoulli", hidden_type_fixed_param=0, if_fix_vis_bias=False, a=None, fix_a_log_ind=None, tol_poisson_max=numpy.log(255), rng=numpy.random.RandomState(100)):
        """
        M: scalar integer, the dimension of input, i.e. the number of input features.
        K: list of integers, the numbers of hidden units in each hidden layer. 
        hidden_type can be Bernoulli, Poisson, Binomial, NegativeBinomial,Multinomial, or Gaussian_FixPrecision1 or Gaussian_FixPrecision2.
        """
        self.features=features
        self.M=M
        self.K=K
        self.NK=len(K) # number of hidden layers
        self.visible_type=visible_type
        self.visible_type_fixed_param=visible_type_fixed_param
        if numpy.isscalar(hidden_type):
            hidden_type=[hidden_type]*self.NK
        self.hidden_type=hidden_type
        if numpy.isscalar(hidden_type_fixed_param):
            hidden_type_fixed_param=[hidden_type_fixed_param]*self.NK
        self.hidden_type_fixed_param=hidden_type_fixed_param
        self.a=[]
        self.b=[]
        self.W=[]
        self.rbms=[]
        self.rng=rng
        
        self.tol_poisson_max=tol_poisson_max
        
        if self.visible_type=="Bernoulli":
            #self.a=self.rng.normal(loc=0, scale=0.01, size=(self.M,1)) # M X 1
            #self.a=self.rng.uniform(low=-0.01, high=0.01, size=(self.M,1)) # M X 1
            self.a=numpy.zeros(shape=(self.M,1))
            self.W=[]
            self.b=[]
            for nk in range(self.NK):
                if nk==0:
                    nrow_W_nk=self.M
                else:
                    nrow_W_nk=self.K[nk-1]
                ncol_W_nk=self.K[nk]
                self.W.append( self.rng.normal(loc=0, scale=0.0001, size=(nrow_W_nk,ncol_W_nk)) ) # M by K[n], initialize weight matrices
                #self.W.append( numpy.zeros(shape=(nrow_W_nk,ncol_W_nk), dtype=float) )
                #self.b.append( self.rng.normal(loc=0, scale=0.01, size=(ncol_W_nk,1)) ) # K[n] X 1
                self.b.append( numpy.zeros(shape=(ncol_W_nk,1)) )
                #self.W.append( self.rng.uniform(low=-0.001, high=0.001, size=(nrow_W_nk,ncol_W_nk)) ) # M by K[n], initialize weight matrices
                #self.b.append( self.rng.uniform(low=-0.001, high=0.001, size=(ncol_W_nk,1)) ) # K[n] X 1
                
        elif self.visible_type=="Gaussian":
            self.a=[None]*2
            self.a[0]=self.rng.normal(loc=0, scale=1, size=(self.M,1)) # M X 1
            self.a[1]=-0.5*numpy.ones(shape=(self.M,1),dtype=float)  # M X 1, -precision/2, beta>0f.M,1)) # M X 1
            self.W=[]
            self.b=[]
            for nk in range(self.NK):
                if nk==0:
                        nrow_W_nk=self.M
                else:
                    nrow_W_nk=self.K[nk-1]
                ncol_W_nk=self.K[nk]
                #self.W.append( self.rng.normal(loc=0, scale=0.001, size=(nrow_W_nk,ncol_W_nk)) ) # M by K[n], initialize weight matrices
                self.W.append( numpy.zeros(shape=(nrow_W_nk,ncol_W_nk),dtype=float) )
                self.b.append( numpy.zeros(shape=(ncol_W_nk,1)) ) # K[n] X 1
                
        elif self.visible_type=="Gaussian_Hinton":
            self.a=[None]*2
            self.a[0]=self.rng.normal(loc=0, scale=0.001, size=(self.M,1)) # M X 1
            self.a[1]=10*numpy.ones(shape=(self.M,1),dtype=float)  # M X 1, precision, beta>0f.M,1)) # M X 1
            self.W=[]
            self.b=[]
            for nk in range(self.NK):
                if nk==0:
                        nrow_W_nk=self.M
                else:
                    nrow_W_nk=self.K[nk-1]
                ncol_W_nk=self.K[nk]
                self.W.append( self.rng.normal(loc=0, scale=0.001, size=(nrow_W_nk,ncol_W_nk)) ) # M by K[n], initialize weight matrices
                self.b.append( numpy.zeros(shape=(ncol_W_nk,1)) ) # K[n] X 1
                
        elif self.visible_type=="Gaussian_FixPrecision1":
            self.a=numpy.zeros(shape=(self.M,1),dtype=float)
            self.W=[]
            self.b=[]
            for nk in range(self.NK):
                if nk==0:
                    nrow_W_nk=self.M
                else:
                    nrow_W_nk=self.K[nk-1]
                ncol_W_nk=self.K[nk]
                self.W.append( numpy.zeros(shape=(nrow_W_nk,ncol_W_nk), dtype=float) ) # M by K[n], initialize weight matrices
                #self.b.append( self.rng.normal(loc=0, scale=0.01, size=(ncol_W_nk,1)) ) # K[n] X 1
                self.b.append( numpy.zeros(shape=(ncol_W_nk,1)) )
                
            if self.visible_type_fixed_param is None:
                self.visible_type_fixed_param=1*numpy.ones(shape=(self.M,1),dtype=float)
            if numpy.isscalar(self.visible_type_fixed_param):
                self.visible_type_fixed_param=self.visible_type_fixed_param*numpy.ones(shape=(self.M,1),dtype=float)
                
        elif self.visible_type=="Gaussian_FixPrecision2":
            self.a=numpy.ones(shape=(self.M,1),dtype=float)
            self.W=[]
            self.b=[]
            for nk in range(self.NK):
                if nk==0:
                    nrow_W_nk=self.M
                else:
                    nrow_W_nk=self.K[nk-1]
                ncol_W_nk=self.K[nk]
                self.W.append( numpy.zeros(shape=(nrow_W_nk,ncol_W_nk), dtype=float) ) # M by K[n], initialize weight matrices
                #self.b.append( self.rng.normal(loc=0, scale=0.01, size=(ncol_W_nk,1)) ) # K[n] X 1
                self.b.append( numpy.zeros(shape=(ncol_W_nk,1)) )
                
            if self.visible_type_fixed_param is None:
                self.visible_type_fixed_param=1*numpy.ones(shape=(self.M,1),dtype=float)
            if numpy.isscalar(self.visible_type_fixed_param):
                self.visible_type_fixed_param=self.visible_type_fixed_param*numpy.ones(shape=(self.M,1),dtype=float)
                
        elif self.visible_type=="Poisson":
            self.a=self.rng.normal(loc=0, scale=0.0001, size=(self.M,1))
            self.W=[]
            self.b=[]
            for nk in range(self.NK):
                if nk==0:
                    nrow_W_nk=self.M
                else:
                    nrow_W_nk=self.K[nk-1]
                ncol_W_nk=self.K[nk]
                #self.W.append( self.rng.normal(loc=0, scale=0.0001, size=(nrow_W_nk,ncol_W_nk)) ) # M by K[n], initialize weight matrices
                self.W.append( numpy.zeros(shape=(nrow_W_nk,ncol_W_nk), dtype=float) )
                #self.b.append( self.rng.normal(loc=0, scale=0.01, size=(ncol_W_nk,1)) ) # K[n] X 1
                self.b.append( numpy.zeros(shape=(ncol_W_nk,1)) )
        elif self.visible_type=="NegativeBinomial":
            self.a=math.log(0.5)*numpy.ones(shape=(self.M,1))
            self.W=[]
            self.b=[]
            for nk in range(self.NK):
                if nk==0:
                    nrow_W_nk=self.M
                else:
                    nrow_W_nk=self.K[nk-1]
                ncol_W_nk=self.K[nk]
                if nk==0:
                    self.W.append( numpy.abs(self.rng.normal(loc=0, scale=0.001, size=(nrow_W_nk,ncol_W_nk) ) ) )
                else:
                    self.W.append( numpy.zeros(shape=(nrow_W_nk,ncol_W_nk), dtype=float) )
                #self.W.append( -0.01*numpy.ones(shape=(nrow_W_nk,ncol_W_nk), dtype=float ) ) # M by K[n], initialize weight matrices
                #self.b.append( self.rng.normal(loc=0, scale=0.01, size=(ncol_W_nk,1)) ) # K[n] X 1
                self.b.append( numpy.zeros(shape=(ncol_W_nk,1)) )
            
            if self.visible_type_fixed_param is None:
                self.visible_type_fixed_param=100*numpy.ones(shape=(self.M,1),dtype=float)
            if numpy.isscalar(self.visible_type_fixed_param):
                self.visible_type_fixed_param=self.visible_type_fixed_param*numpy.ones(shape=(self.M,1),dtype=float)
        elif self.visible_type=="Multinomial":
            self.a=numpy.zeros(shape=(self.M,1),dtype=float)
            self.W=[]
            self.b=[]
            for nk in range(self.NK):
                if nk==0:
                    nrow_W_nk=self.M
                else:
                    nrow_W_nk=self.K[nk-1]
                ncol_W_nk=self.K[nk]
                self.W.append( numpy.zeros(shape=(nrow_W_nk,ncol_W_nk), dtype=float) ) # M by K[n], initialize weight matrices
                #self.b.append( self.rng.normal(loc=0, scale=0.01, size=(ncol_W_nk,1)) ) # K[n] X 1
                self.b.append( numpy.zeros(shape=(ncol_W_nk,1)) )
        elif self.visible_type=="Multinoulli":
            self.Ms=visible_type_fixed_param
            self.a=math.log(1/self.M)*numpy.ones(shape=(self.M,1))
            self.W=[]
            self.b=[]
            for nk in range(self.NK):
                if nk==0:
                    W_input=[None]*self.M
                    ncol_W_nk=self.K[nk]
                    for m in range(self.M):
                        W_input[m]=self.rng.normal(loc=0, scale=0.001, size=(self.Ms[m],ncol_W_nk))
                    self.W.append(W_input)
                else:
                    nrow_W_nk=self.K[nk-1]
                    ncol_W_nk=self.K[nk]
                    self.W.append( self.rng.normal(loc=0, scale=0.001, size=(nrow_W_nk,ncol_W_nk)) ) # M by K[n], initialize weight matrices
                #self.b.append( self.rng.normal(loc=0, scale=0.01, size=(ncol_W_nk,1)) ) # K[n] X 1
                self.b.append( numpy.zeros(shape=(ncol_W_nk,1)) )
        elif self.visible_type=="Gamma":
            self.a=[None]*2
            self.a[0]=1*numpy.ones(shape=(self.M,1), dtype=float)
            self.a[1]=-numpy.ones(shape=(self.M,1),dtype=float)  # M X 1, -precision/2, beta>0f.M,1)) # M X 1
            self.W=[]
            self.b=[]
            for nk in range(self.NK):
                if nk==0:
                        nrow_W_nk=self.M
                else:
                    nrow_W_nk=self.K[nk-1]
                ncol_W_nk=self.K[nk]
                self.W.append( numpy.ones(shape=(nrow_W_nk,ncol_W_nk),dtype=float) ) # M by K[n], initialize weight matrices
                self.b.append( numpy.zeros(shape=(ncol_W_nk,1)) ) # K[n] X 1
        else:
            print("Error! Please select a correct data type for visible variables from {Bernoulli,Gaussian,Multinoulli,Poisson}.")
            exit()
        # whether fix a if this DBM is a joint component in multimodal DBM
        self.if_fix_vis_bias=if_fix_vis_bias
        self.fix_a_log_ind=fix_a_log_ind
        if if_fix_vis_bias:
            self.fix_vis_bias(a,fix_a_log_ind)

    
    def fix_vis_bias(self,a=None,fix_a_log_ind=None):
        """
        Fix the visible bias. Do not update them in learning.
        a: a numpy array of shape M by 1.
        fix_a_log_ind: a bool numpy vector of length M, fixed_log_ind[m]==True means fix self.a[m]
        """
        if a is not None:
            self.a=a # reset a
            if len(self.rbms)>0:
                self.rbms[0].a=a # reset the rbm's visiable bias
            self.if_fix_vis_bias=True
            self.fix_a_log_ind=fix_a_log_ind
            if self.fix_a_log_ind is None:
                self.fix_a_log_ind=numpy.array([True]*self.M)
        else: # do not reset a
            self.if_fix_vis_bias=True
            self.fix_a_log_ind=fix_a_log_ind
            if self.fix_a_log_ind is None:
                self.fix_a_log_ind=numpy.array([True]*self.M)
        

    def pretrain(self, X=None, batch_size=10, pcdk=20, NS=20 ,maxiter=100, learn_rate_a=0.01, learn_rate_b=0.01, learn_rate_W=0.01, change_rate=0.8, adjust_change_rate_at=None, adjust_coef=1.02, change_every_many_iters=10, init_chain_time=100, train_subset_size_for_compute_error=100, valid_subset_size_for_compute_error=100, track_reconstruct_error=True, track_free_energy=True, reinit_a_use_data_stat=False, if_plot_error_free_energy=False, dir_save="./", prefix="RBM", figwidth=5, figheight=3): 
        """
        Pretraining DBM using RBMs.
        Different layers have different learning rate.
        """
        start_time=time.clock()
        # different layers can have different learning rates         
        if numpy.isscalar(learn_rate_b):
            learn_rate_b=[learn_rate_b]*self.NK
        if numpy.isscalar(learn_rate_W):
            learn_rate_W=[learn_rate_W]*self.NK        
            
        self.X=X
        rbm_X=self.X
        visible_type=self.visible_type
        #self.rbms=[] # define it in initialization
        self.H_pretrain=[]
        print("Start pretraining DBM...")
        for nk in range(self.NK):
            print("the {0}-th hidden layer...".format(nk+1))
            if nk==0 and self.NK>1: # bottom RBM
                tie_W_for_pretraining_DBM_bottom=True
                tie_W_for_pretraining_DBM_top=False
                rbm_visible_length=self.M
                rbm_hidden_length=self.K[nk]
                visible_type=self.visible_type
                visible_type_fixed_param=self.visible_type_fixed_param
                hidden_type=self.hidden_type[nk]
                hidden_type_fixed_param=self.hidden_type_fixed_param[nk]
                rbm_if_fix_vis_bias=self.if_fix_vis_bias
                a=self.a
                rbm_fix_a_log_ind=self.fix_a_log_ind
                rbm_track_reconstruct_error=track_reconstruct_error
                rbm_track_free_energy=track_free_energy
                rbm_reinit_a_use_data_stat=reinit_a_use_data_stat
                rbm_learn_rate_a=learn_rate_a
            elif nk==self.NK-1 and self.NK>1: # top RBM
                tie_W_for_pretraining_DBM_bottom=False
                tie_W_for_pretraining_DBM_top=True
                rbm_visible_length=self.K[nk-1]
                rbm_hidden_length=self.K[nk]
                visible_type=self.hidden_type[nk-1]
                visible_type_fixed_param=self.hidden_type_fixed_param[nk-1]
                hidden_type=self.hidden_type[nk]
                hidden_type_fixed_param=self.hidden_type_fixed_param[nk]
                rbm_if_fix_vis_bias=True
                rbm_fix_a_log_ind=None
                rbm_track_reconstruct_error=track_reconstruct_error
                rbm_track_free_energy=track_free_energy
                rbm_reinit_a_use_data_stat=False
                rbm_learn_rate_a=learn_rate_b[nk-1]
            elif nk==0 and self.NK==1: # there is only one hidden layer
                tie_W_for_pretraining_DBM_bottom=False
                tie_W_for_pretraining_DBM_top=False
                rbm_visible_length=self.M
                rbm_hidden_length=self.K[nk]
                visible_type=self.visible_type
                visible_type_fixed_param=self.visible_type_fixed_param
                hidden_type=self.hidden_type[nk]
                hidden_type_fixed_param=self.hidden_type_fixed_param[nk]
                rbm_if_fix_vis_bias=self.if_fix_vis_bias
                a=self.a
                rbm_fix_a_log_ind=self.fix_a_log_ind
                rbm_track_reconstruct_error=track_reconstruct_error
                rbm_track_free_energy=track_free_energy
                rbm_reinit_a_use_data_stat=reinit_a_use_data_stat
                rbm_learn_rate_a=learn_rate_a
            else: # middle RBMs
                tie_W_for_pretraining_DBM_bottom=False
                tie_W_for_pretraining_DBM_top=False
                rbm_visible_length=self.K[nk-1]
                rbm_hidden_length=self.K[nk]
                visible_type=self.hidden_type[nk-1]
                visible_type_fixed_param=self.hidden_type_fixed_param[nk-1]
                hidden_type=self.hidden_type[nk]
                hidden_type_fixed_param=self.hidden_type_fixed_param[nk]
                rbm_if_fix_vis_bias=True
                rbm_fix_a_log_ind=None
                rbm_track_reconstruct_error=track_reconstruct_error
                rbm_track_free_energy=track_free_energy
                rbm_reinit_a_use_data_stat=False
                rbm_learn_rate_a=learn_rate_b[nk-1]
            # initialize RBM
            rbm_model=restricted_boltzmann_machine.restricted_boltzmann_machine(M=rbm_visible_length, K=rbm_hidden_length, visible_type=visible_type, visible_type_fixed_param=visible_type_fixed_param, hidden_type=hidden_type, hidden_type_fixed_param=hidden_type_fixed_param, tie_W_for_pretraining_DBM_bottom=tie_W_for_pretraining_DBM_bottom, tie_W_for_pretraining_DBM_top=tie_W_for_pretraining_DBM_top, if_fix_vis_bias=rbm_if_fix_vis_bias, a=a, fix_a_log_ind=rbm_fix_a_log_ind, tol_poisson_max=self.tol_poisson_max, rng=self.rng)
            # train RBM
            #print "The shape of rbm_X is{0}".format(rbm_X.shape)
            rbm_model.train(X=rbm_X, batch_size=batch_size, pcdk=pcdk, NS=NS, maxiter=maxiter, learn_rate_a=rbm_learn_rate_a, learn_rate_b=learn_rate_b[nk], learn_rate_W=learn_rate_W[nk], change_rate=change_rate, adjust_change_rate_at=adjust_change_rate_at, adjust_coef=adjust_coef, change_every_many_iters=change_every_many_iters, init_chain_time=init_chain_time, train_subset_size_for_compute_error=train_subset_size_for_compute_error, valid_subset_size_for_compute_error=valid_subset_size_for_compute_error, track_reconstruct_error=rbm_track_reconstruct_error, track_free_energy=rbm_track_free_energy, reinit_a_use_data_stat=rbm_reinit_a_use_data_stat, if_plot_error_free_energy=if_plot_error_free_energy, dir_save=dir_save, prefix=prefix+"_RBM_"+str(nk), figwidth=figwidth, figheight=figheight)
            # assign parameters to corresponding layers
            if nk==0 and self.NK>1: # bottom RBM
                a_nk,b_nk,W_nk=rbm_model.get_param()
                self.a=a_nk
                self.W[nk]=W_nk
                self.b[nk]=b_nk
            elif nk==self.NK-1 and self.NK>1: # top RBM
                a_nk,b_nk,W_nk=rbm_model.get_param()
                self.W[nk]=W_nk
                self.b[nk]=b_nk
            elif nk==0 and self.NK==1: # there is only one hidden layer
                a_nk,b_nk,W_nk=rbm_model.get_param()
                self.a=a_nk
                self.W[nk]=W_nk
                self.b[nk]=b_nk
            else: # middle RBMs
                a_nk,b_nk,W_nk=rbm_model.get_param()
                self.W[nk]=0.5*W_nk
                self.b[nk]=b_nk

            #rbm_X,_=rbm_model.sample_h_given_x(rbm_X) # the output of this layer is used as input of the next layer
            _,rbm_X=rbm_model.sample_h_given_x(rbm_X) # Hinton suggested to use probabilities
            a=b_nk # the bias of the nk-th hidden layer is used as the bias of visible notes of the nk+1-th layer 

            # save the trained rbms for initialize mean-filed approximation and Gibbs sampling.
            self.rbms.append(rbm_model)
            self.H_pretrain.append(rbm_X) # H of each RBM, for the purpose of (1) initializing mean-field approximation inference, (2) Gibbs sampling, and (3) building multi-modal DBM.

        print("Finished pretraining of DBM!")
        end_time = time.clock()
        self.pretrain_time=end_time-start_time
        return self.pretrain_time
        print("It took {0} seconds.".format(self.pretrain_time))


    def train(self,X=None, X_validate=None, batch_size=10, NMF=20, pcdk=20, NS=20, maxiter=100, learn_rate_a=0.01, learn_rate_b=0.01, learn_rate_W=0.01, change_rate=0.8, adjust_change_rate_at=None, adjust_coef=1.02, change_every_many_iters=10, init_chain_time=100, train_subset_size_for_compute_error=100, valid_subset_size_for_compute_error=100, track_reconstruct_error=True, track_free_energy=True, sumout="auto", if_plot_error_free_energy=False, dir_save="./", prefix="DBM", figwidth=5, figheight=3):
        """
        Train DBM.
        Different layers have different learning rate.
        """
        start_time=time.clock()
        print("Start training DBM...")
        if self.visible_type=="Multinoulli": # convert to binary
            self.X=[None]*self.M
            self.X_validate=[None]*self.M
            for m in range(self.M):
                Z,_=cl.membership_vector_to_indicator_matrix(X[m,:],z_unique=list(range(self.Ms[m])))
                self.X[m]=Z.transpose()
                self.N=self.X[0].shape[1]
                if X_validate is not None:
                    Z,_=cl.membership_vector_to_indicator_matrix(X_validate[m,:],z_unique=list(range(self.Ms[m])))
                    self.X_validate[m]=Z.transpose()
                    self.N_validate=self.X_validate[0].shape[1] # number of validation samples
        else: # not multinoulli variables
            self.X=X
            self.N=self.X.shape[1] # number of training samples
            self.X_validate=X_validate
            if X_validate is not None:
                self.N_validate=self.X_validate.shape[1] # number of validation samples
            else:
                self.N_validate=0

        self.batch_size=batch_size
        if self.batch_size>self.N:
            self.batch_size=1
                
        if self.NK==1:
            print("There is only one hidden layer. This is just a RBM, a pretraining is thus enough. I decide to exit.")
            return 0
    
        # different layers have different learning rates         
        if numpy.isscalar(learn_rate_b):
            learn_rate_b=[learn_rate_b]*self.NK
        if numpy.isscalar(learn_rate_W):
            learn_rate_W=[learn_rate_W]*self.NK        
            
        self.pcdk=pcdk
        self.NMF=NMF
        self.NS=NS
        self.maxiter=maxiter
        self.learn_rate_a=learn_rate_a
        self.learn_rate_b=learn_rate_b
        self.learn_rate_W=learn_rate_W
        self.change_rate=change_rate
        self.change_every_many_iters=change_every_many_iters
        
        self.rec_errors_train=[]
        self.rec_errors_valid=[]
        self.mfes_train=[]
        self.mfes_valid=[]

        print("initializing Markov chains ...")
        # initialize Markov chains
        _,_,_,_,_,_=self.pcd_sampling(pcdk=init_chain_time*pcdk,NS=NS,persistent=True, init_sampling=True, rand_init_X=False, rand_init_H=False) # initialize pcd

        for i in range(self.maxiter):
            
            if adjust_change_rate_at is not None:
                if i==adjust_change_rate_at[0]:
                    change_rate=change_rate*adjust_coef # increast change_rate
                    change_rate=1.0 if change_rate>1.0 else change_rate # make sure not greater than 1
                    if len(adjust_change_rate_at)>1:
                        adjust_change_rate_at=adjust_change_rate_at[1:] # delete the first element
                    else:
                        adjust_change_rate_at=None
                        
            self.learn_rate_a=self.change_learning_rate(current_learn_rate=self.learn_rate_a, change_rate=change_rate, current_iter=i, change_every_many_iters=change_every_many_iters)
            self.learn_rate_b=self.change_learning_rate(current_learn_rate=self.learn_rate_b, change_rate=change_rate, current_iter=i, change_every_many_iters=change_every_many_iters)
            self.learn_rate_W=self.change_learning_rate(current_learn_rate=self.learn_rate_W, change_rate=change_rate, current_iter=i, change_every_many_iters=change_every_many_iters)
            #print "starting the {0}-th iteration, the learning rate of a, b, W: {1}, {2}, {3}".format(i,self.learn_rate_a,self.learn_rate_b,self.learn_rate_W)
            # get mini-batch
            Xbatch=self.sample_minibatch(self.batch_size)
            
            # mean-field approximation, Hbatch is prob
            _,XbatchRM,_,_,Hbatch=self.mean_field_approximate_inference(Xbatch, NMF=self.NMF, rand_init_H=False)

            # pcd sampling
            XS,HS,_,_,_,_=self.pcd_sampling(pcdk=pcdk,init_sampling=False)
            
            # compute gradient
            self.compute_gradient(Xbatch,Hbatch,XS,HS)
            
            # update parameters
            self.update_param()
            
            # update the parameters for RBMs
            self.update_rbms()
            
            # compute reconstruction error of the training samples
            # sample some training samples, rather than use all training samples which is time-consuming
            if track_reconstruct_error:
                rec_error_train=self.compute_reconstruction_error(X0=Xbatch, X0RM=XbatchRM)
                self.rec_errors_train.append(rec_error_train)
            if track_free_energy:
                mfe_train,_=self.compute_free_energy(X=Xbatch, H=Hbatch)
                self.mfes_train.append(mfe_train)
                
            if self.X_validate is not None:
                if valid_subset_size_for_compute_error is not None:
                    valid_subset_ind=self.rng.choice(numpy.arange(self.N_validate,dtype=int),size=valid_subset_size_for_compute_error)
                    if self.visible_type=="Multinoulli":
                        X_validate_subset=[None]*self.M
                        for m in range(self.M):
                            X_validate_subset[m]=self.X_validate[m][:,valid_subset_ind]
                    else:
                        X_validate_subset=self.X_validate[:,valid_subset_ind]
                    if track_reconstruct_error:
                        rec_error_valid=self.compute_reconstruction_error(X0=X_validate_subset, X0RM=None)
                        self.rec_errors_valid.append(rec_error_valid)
                    if track_free_energy:
                        mfe_validate,_=self.compute_free_energy(X=X_validate_subset, H=None, sumout=sumout)
                        self.mfes_valid.append(mfe_validate)
                else:
                    if track_reconstruct_error:                    
                        rec_error_valid=self.compute_reconstruction_error(X0=self.X_validate, X0RM=None)
                        self.rec_errors_valid.append(rec_error_valid)
                    if track_free_energy:
                        mfe_validate,_=self.compute_free_energy(X=self.X_validate, H=None, sumout=sumout) 
                        self.mfes_valid.append(mfe_validate)
                # compute difference of free energy between training set and validation  set
                # the log-likelihood(train_set) - log-likelihood(validate_set) = F(validate_set) - F(train_set), the log-partition function, logZ is cancelled out
                if track_reconstruct_error and track_free_energy:
                    free_energy_dif=mfe_train - mfe_validate
                    print("{0}-th iteration, learn_rate_W: {1}, train_rec_error: {2}, valid_rec_error: {3}, free_energy_train: {4}, free_energy_valid: {5}, free_energy_dif: {6}".format(i, self.learn_rate_W, rec_error_train, rec_error_valid, mfe_train, mfe_validate, free_energy_dif))
                elif not track_reconstruct_error and track_free_energy:
                    free_energy_dif=mfe_train - mfe_validate
                    print("{0}-th iteration, learn_rate_W: {1}, free_energy_train: {2}, free_energy_valid: {3}, free_energy_dif: {4}".format(i, self.learn_rate_W, mfe_train, mfe_validate, free_energy_dif))
                elif track_reconstruct_error and not track_free_energy:
                    print("{0}-th iteration, learn_rate_W: {1}, train_rec_error: {2}, valid_rec_error: {3}".format(i, self.learn_rate_W, rec_error_train, rec_error_valid))
                elif not track_reconstruct_error and not track_free_energy:
                    print("{0}-th iteration, learn_rate_W: {1}".format(i, self.learn_rate_W))
            else:
                if track_reconstruct_error and track_free_energy:
                    print("{0}-th iteration, learn_rate_W: {1}, train_rec_error: {2}, free_energy_train: {3}".format(i, self.learn_rate_W, rec_error_train, mfe_train))
                elif not track_reconstruct_error and track_free_energy:
                    print("{0}-th iteration, learn_rate_W: {1}, free_energy_train: {2}".format(i, self.learn_rate_W, mfe_train))
                elif track_reconstruct_error and not track_free_energy:
                    print("{0}-th iteration, learn_rate_W: {1}, train_rec_error: {2}".format(i, self.learn_rate_W, rec_error_train))
                elif not track_reconstruct_error and not track_free_energy:
                    print("{0}-th iteration, learn_rate_W: {1}".format(i, self.learn_rate_W))
                    
        if if_plot_error_free_energy:
            self.plot_error_free_energy(dir_save, prefix=prefix, figwidth=figwidth, figheight=figheight)

        print("The (fine-tuning) training of DBM is finished!")
        end_time = time.clock()
        self.train_time=end_time-start_time
        return self.train_time
        print("It took {0} seconds.".format(self.train_time))
        

    def smooth(self,x, mean_over=5):
        """
        Smooth a vector of numbers.
        x: list of vector.
        mean_over: scalar, the range of taking mean.
        """
        num=len(x)
        x_smooth=numpy.zeros((num,))
        for n in range(num):
            start=n-mean_over+1
            if start<0:
                start=0
            x_smooth[n]=numpy.mean(x[start:n+1])
        return x_smooth


    def plot_error_free_energy(self, dir_save="./", prefix="DBM", mean_over=5, figwidth=5, figheight=3):
        import matplotlib as mpl
        mpl.use("pdf")
        import matplotlib.pyplot as plt
        
        if len(self.rec_errors_train)>0:
            num_iters=len(self.rec_errors_train)
            if mean_over>0:
                self.rec_errors_train=self.smooth(self.rec_errors_train, mean_over=mean_over)
            else:
                self.rec_errors_train=numpy.array(self.rec_errors_train)

        if len(self.rec_errors_valid)>0:
            num_iters=len(self.rec_errors_valid)
            if mean_over>0:
                self.rec_errors_valid=self.smooth(self.rec_errors_valid, mean_over=mean_over)
            else:
                self.rec_errors_valid=numpy.array(self.rec_errors_valid)

        if len(self.mfes_train)>0:
            num_iters=len(self.mfes_train)
            if mean_over>0:
                self.mfes_train=self.smooth(self.mfes_train, mean_over=mean_over)
            else:
                self.mfes_train=numpy.array(self.mfes_train)

        if len(self.mfes_valid)>0:
            num_iters=len(self.mfes_valid)
            if mean_over>0:
                self.mfes_valid=self.smooth(self.mfes_valid, mean_over=mean_over)
            else:
                self.mfes_valid=numpy.array(self.mfes_valid)

        iters=numpy.array(range(num_iters),dtype=int)
        
        # ignore the first five results as they are not stable
        iters=iters[5:]
        if len(self.rec_errors_train)>0:
            self.rec_errors_train=self.rec_errors_train[5:]
        if len(self.rec_errors_valid)>0:
            self.rec_errors_valid=self.rec_errors_valid[5:]
        if len(self.mfes_train)>0:
            self.mfes_train=self.mfes_train[5:]
        if len(self.mfes_valid)>0:
            self.mfes_valid=self.mfes_valid[5:]
        
        #plt.ion()
        fig=plt.figure(num=1,figsize=(figwidth,figheight))
        ax=fig.add_subplot(1,1,1)
        if len(self.mfes_train)>0:
            ax.plot(iters,self.mfes_train,linestyle="-", color="blue", linewidth=0.5, label="FE:Train")
        if len(self.mfes_valid)>0:
            ax.plot(iters,self.mfes_valid,linestyle=":",color="blueviolet",linewidth=0.5, label="FE:Test")
        ax.set_xlabel("Iteration",fontsize=8)
        ax.set_ylabel("Free Energy (FE)",color="blue",fontsize=8)
        for tl in ax.get_yticklabels():
            tl.set_color("b")
        plt.setp(ax.get_yticklabels(), fontsize=8)
        plt.setp(ax.get_xticklabels(), fontsize=8)
            
        #ax.legend(loc="lower left",fontsize=8)

        ax2=ax.twinx()
        if len(self.rec_errors_train)>0:
            ax2.plot(iters,self.rec_errors_train,linestyle="-",color="red",linewidth=0.5, label="RCE:Train")
        if len(self.rec_errors_valid)>0:
            ax2.plot(iters,self.rec_errors_valid,linestyle=":",color="darkgoldenrod",linewidth=0.5, label="RCE:Test")
        ax2.set_ylabel("Reconstruction Error (RCE)", color="red",fontsize=8)
        for tl in ax2.get_yticklabels():
            tl.set_color("r")
        plt.setp(ax2.get_yticklabels(), fontsize=8)
        plt.setp(ax2.get_xticklabels(), fontsize=8)
        # legend
        ax.legend(loc="lower left",fontsize=8)
        ax2.legend(loc="upper right",fontsize=8)
        filename=dir_save+prefix+"_error_free_energy.pdf"
        plt.tight_layout()
        fig.savefig(filename,bbox_inches='tight',format="pdf",dpi=600)
        plt.close(fig)
        #plt.close("all")
        

    def compute_reconstruction_error(self, X0, X0RM=None):
        """
        Compute the difference between the real sample X0 and the recoverd sample X0RM by mean-field.
        """
        if X0RM is None:
            _,X0RM,_,_,_=self.mean_field_approximate_inference(X0,NMF=self.NMF,rand_init_H=False)
        if self.visible_type=="Multinoulli":
            self.rec_error=0
            for m in range(self.M):
                self.rec_error=self.rec_error+numpy.mean(numpy.abs(X0RM[m]-X0[m]))
        else:
            self.rec_error=numpy.mean(numpy.abs(X0RM-X0))
        return self.rec_error


    def compute_free_energy(self,X=None, H=None, sumout="auto"): ##################### need update for Gaussian visible type
        """
        Compute "free" energy E() with some layers summed out. 
        """
        if X is None:
            X=self.X
        if H is None:
            _,_,_,_,H=self.mean_field_approximate_inference(X,NMF=self.NMF,rand_init_H=False)
        
        if sumout=="auto":
            if self.NK%2==1: # NK is odd, such as x, h1, h2, h3, h4, h5
                sumout="odd" # sum out h1, h3, h5; equlvalently, x, h2, h4 can be summed out
            if self.NK%2==0: # NK is even, such as x, h1, h2, h3, h4
                sumout="even" # sum out x, h2, h4
            
        z=self.zeta(X=X, H=H, a=self.a, b=self.b, visible_type=self.visible_type, visible_type_fixed_param=self.visible_type_fixed_param, sumout=sumout)
        
        a_hat,b_hat=self.compute_posterior_bias(visible_type=self.visible_type, visible_type_fixed_param=self.visible_type_fixed_param, a=self.a, b=self.b, W=self.W, X=X, H=H, scale=1)
        logPar=self.A(a_hat=a_hat, b_hat=b_hat, visible_type=self.visible_type, visible_type_fixed_param=self.visible_type_fixed_param, hidden_type=self.hidden_type, hidden_type_fixed_param=self.hidden_type_fixed_param, sumout=sumout )
        fes = -z - logPar
        
        mfe=numpy.mean(fes) # average over N samples
        return mfe,fes
        

    def change_learning_rate(self, current_learn_rate, change_rate, current_iter, change_every_many_iters):
        if current_iter!=0 and current_iter%change_every_many_iters==0:
            if numpy.isscalar(current_learn_rate):
                return current_learn_rate * change_rate
            else:
                new_learn_rate=[c*change_rate for c in current_learn_rate]
                #R=len(current_learn_rate)
                #new_learn_rate=[None]*R
                #for r in range(R):
                #    new_learn_rate[r]=current_learn_rate[r]*change_rate
                return new_learn_rate
        else:
            return current_learn_rate


    def sample_minibatch(self, batch_size=20):
        ind_batch=self.rng.choice(self.N,size=batch_size,replace=False)
        if self.visible_type=="Multinoulli":
            Xbatch=[None]*self.M
            for m in range(self.M):
                Xbatch[m]=self.X[m][:,ind_batch]
                if batch_size==1:
                    Xbatch[m].shape=(self.Ms[m],1)
        else:
            Xbatch=self.X[:,ind_batch]
            if batch_size==1:
                Xbatch.shape=(self.M,1)
        return Xbatch


    def estimate_log_likelihood(self, X=None, NMF=100, base_rate_type="prior", beta=None, step_base=0.999, T=10000, stepdist="even", S=100, sumout="auto", dir_save="/.", prefix="DBM"):
        """
        Estimate the log-likelihood of DBM.
        """
        if X is None:
            X=self.X
        # trivial case, just a RBM, set NMF=1
        if self.NK==1:
            NMF=1 
            
        print("I am estimating the log-likelihood...")
        # E_q[-E(x,h^1, h^2,..., h^L)] 
        _,_,_,_,HP=self.mean_field_approximate_inference(Xbatch=X, NMF=NMF, rand_init_H=False)
        mean_energy,_=self.compute_energy(X,HP)
        # estimate logZ
        logZ,logZ_std,logws,log_ratio_AIS_mean,log_ratio_AIS_std=self.estimate_logZ(base_rate_type=base_rate_type, beta=beta, step_base=step_base, T=T, stepdist=stepdist, S=S, sumout=sumout)
        # compute entropy of approximate distributions
        mean_entropy,_=self.compute_entropy(HP)
        # compute log-likelihood 
        loglh= -mean_energy - logZ + mean_entropy
        print("log-likelihood:{0}, logZ:{1}, logZ_std:{2}, energy: {3}, entropy: {4}".format(loglh, logZ, logZ_std, mean_energy, mean_entropy))
        
        # save results
        result=numpy.zeros(shape=(5,2),dtype=object)
        result[:,0]=numpy.array(["log-likelihood","logZ","logZ_std","energy","entropy"])
        result[:,1]=numpy.array([loglh,logZ,logZ_std,mean_energy,mean_entropy])
        filename=dir_save + prefix + "_estimated_log_likelihood.txt"
        numpy.savetxt(filename, result, delimiter="\t", fmt="%s")
        
        return loglh,logZ,logZ_std,mean_energy,mean_entropy
      
    
    def compute_energy(self,X,H):
        """
        Compute energy E(x,h) given x and h. 
        """
        print("I am computing energy E(X,h)...")
        num_samples=X.shape[1]
        Es=0
        for n in range(num_samples):
            En=-self.rbms[0].zeta(theta=self.a, X=X[:,[n]], fixed_param=self.visible_type_fixed_param, distribution=self.visible_type)
            #En=-numpy.dot(self.a.transpose(),X[:,[n]])
            for l in range(self.NK):
                if l==0:
                    if self.visible_type=="Multinoulli":
                        for m in range(self.M):
                            En=En - numpy.dot(self.b[l].transpose(),H[l][:,[n]]) - X[m][:,[n]].transpose().dot(self.W[l][m]).dot(H[l][:,[n]])
                    else:
                        En=En - numpy.dot(self.b[l].transpose(),H[l][:,[n]]) - X[:,[n]].transpose().dot(self.W[l]).dot(H[l][:,[n]])
                else: # not first hidden layer
                    En=En - numpy.dot(self.b[l].transpose(),H[l][:,[n]]) - H[l-1][:,[n]].transpose().dot(self.W[l]).dot(H[l][:,[n]])
            Es=Es+En
        Es=Es[0,0] # take off [[]]
        ME=Es/num_samples # mean energy
        return ME,Es
        
    
#    def compute_entropy(self,HP=0):
#        """
#        Compute the entropy of approximate distribution q(h).
#        Only work for Bernoulli and Multinoulli distributions.
#        HP: each column of HP[l] is a sample.
#        """
#        print "I am computing entropy..."
#        entropies=0
#        num_samples=HP[0].shape[1]
#        for n in range(num_samples):
#            entropies_n=0
#            for l in range(self.NK):
#                HPln=HP[l][:,n]
#                for h in HPln:
#                    if h!=0 and h!=1:
#                        entropies_n= entropies_n -h*numpy.log(h) - (1-h)*numpy.log(1-h)
#            entropies= entropies + entropies_n
#        mean_entropy=entropies/num_samples
#        return mean_entropy,entropies
        

    def compute_entropy(self,HP):
        """
        Compute the entropy of approximate distribution q(h).
        Only work for Bernoulli and Multinoulli distributions.
        HP: each column of HP[l] is a sample.
        """
        print("I am computing entropy...")
        entropies=0
        num_samples=HP[0].shape[1]
        #print "There are {} samples".format(num_samples)
        for n in range(num_samples):
            entropies_n=0
            #print "there are {} hidden layers".format(self.NK)
            for l in range(self.NK):
                if self.hidden_type[l]=="Bernoulli":
                    HPln=HP[l][:,n]
                    for h in HPln:
                        if h!=0 and h!=1:
                            entropies_n= entropies_n -h*numpy.log(h) - (1-h)*numpy.log(1-h)
                elif self.hidden_type[l]=="Multinomial": # only applicable for count is 1, = multinoulli
                    HPln=HP[l][:,n]
                    for h in HPln:
                        if h!=0 and h!=1:
                            entropies_n= entropies_n - h*numpy.log(h)
                elif self.hidden_type[l]=="Binomial":
                    HPln=HP[l][:,n]
                    for h in HPln:
                        if h!=0 and h!=1:
                            entropies_n= entropies_n + 0.5*numpy.log( 2*numpy.pi*numpy.e*self.hidden_type_fixed_param[l]*h*(1-h))
                elif self.hidden_type[l]=="Gaussian_FixPrecision1" or self.hidden_type[l]=="Gaussian_FixPrecision2":
                    for k in range(self.K[l]):
                        entropies_n= entropies_n + 0.5*numpy.log( 2*numpy.pi*numpy.e/self.hidden_type_fixed_param[l])
                elif self.hidden_type[l]=="Gaussian":
                    for k in range(self.K[l]):
                        entropies_n= entropies_n + 0.5*numpy.log( 2*numpy.pi*numpy.e/(-2*self.b[l][1]))
                else:
                    print("The entropy for {0} distrubution is not implemented yet.".format(self.hidden_type[l]))
                    entropies_n= entropies_n + 0
            entropies= entropies + entropies_n
        mean_entropy=entropies/num_samples
        #print "The mean entropy is {}".format(mean_entropy)
        return mean_entropy,entropies
        
      
    def estimate_logZ(self, base_rate_type="prior", beta=None, step_base=0.999, T=10000, stepdist="even", S=100, sumout="auto"):
        """
        Estimate the (log) partition function.
        base_type: string, the type of model A, either "uniform" or "prior".
        beta: list or numpy vector, the inceasing sequence.
        step_base: scalar, used to compute beta: beta=1-step_base^k.
        T: scalar, number of steps.
        S: scalar, number of repeats for taking avarage.
        """
        print("I am estimating the log-partition function log(Z) ...")
        #num_samples=X.shape[1] # number of samples
        if beta is None:
            #step1=0.001
            #setp2=0.0001
            #step3=0.00001
            #beta=numpy.concatenate( (numpy.arange(0,0.5,step1), numpy.arange(0.5,0.9,step2), numpy.arange(0.9,1+0.1*step3,step3)))
            beta=numpy.arange(0,T,step=1,dtype=int)
            if stepdist=="exp":        
                beta=1-step_base**beta
            if stepdist=="even":
                beta=beta/T
            beta=numpy.concatenate((beta,[1]))
            T=len(beta) # actually, there are T+1 elements in beta
        
        a_A,b_A,logZ_A=self.compute_logZ_A(self.a, self.b, visible_type=self.visible_type, visible_type_fixed_param=self.visible_type_fixed_param, base_rate_type=base_rate_type)
        
        print("logZ_A={0}".format(logZ_A))
        
        print("I need to run AIS for {0} times.".format(S))
        logws=numpy.zeros(shape=(S,),dtype=float)
        #ws=numpy.zeros(shape=(S,),dtype=float)
        for s in range(S):
            print("I am running the {0}-th AIS...".format(s))
            # Markov chain
            x_t=numpy.zeros(shape=(self.M,1),dtype=float) # used to initialize, not really used anyway
            h_A_t=[None]*self.NK
            h_B_t=[None]*self.NK # I need to initialize it using zeros as they are not used actually when beta_t=0
            log_p_star_diff=numpy.zeros(shape=(T-1,),dtype=float)
            
            # just initialize h_B_t, for the first iteration where beta_t=0, it is not use            
            h_B_t,_=self.sample_hidden(visible_type=self.visible_type, visible_type_fixed_param=self.visible_type_fixed_param, hidden_type=self.hidden_type, hidden_type_fixed_param=self.hidden_type_fixed_param, NS=1, X=None, H=None, a=self.a, b=self.b, W=None, beta_t=0, even_or_odd_first="even", rng=self.rng)
            x_t,_,_=self.sample_visible(visible_type=self.visible_type, a=self.a, W=None, H=None, visible_type_fixed_param=self.visible_type_fixed_param, tie_W_for_pretraining_DBM_top=False, NS=1, rng=self.rng)
            for t in range(T-1):
                                
                # sample hidden
                h_A_t,_=self.sample_hidden(visible_type=self.visible_type, visible_type_fixed_param=self.visible_type_fixed_param, hidden_type=self.hidden_type, hidden_type_fixed_param=self.hidden_type_fixed_param, NS=1, X=None, H=None, a=a_A, b=b_A, W=None, beta_t=1-beta[t], even_or_odd_first="even", rng=self.rng)
                h_B_t,_=self.sample_hidden(visible_type=self.visible_type, visible_type_fixed_param=self.visible_type_fixed_param, hidden_type=self.hidden_type, hidden_type_fixed_param=self.hidden_type_fixed_param, X=x_t, H=h_B_t, a=self.a, b=self.b, W=self.W, beta_t=beta[t], even_or_odd_first="even", rng=self.rng)
                
                # sample visible
                a_B_hat=self.rbms[0].compute_posterior_bais(self.a, self.W[0], h_B_t[0], visible_or_hidden="visible", distribution=self.visible_type)
                a_hat_t=self.rbms[0].combine_a_A_a_B_hat(a_A, a_B_hat=a_B_hat, beta_t=beta[t], visible_type=self.visible_type)
                x_t,_,_=self.sample_visible(visible_type=self.visible_type, a=a_hat_t, W=None, H=None, visible_type_fixed_param=self.visible_type_fixed_param, tie_W_for_pretraining_DBM_top=False, NS=1, rng=self.rng)                
                
                # log p*()
#                if self.visible_type=="NegativeBinomial":
#                    if t==0:
#                        log_p_star_t_xh_t=0
#                        log_p_star_tplus1_xh_t=self.log_p_star_t(x=x_t, h_A=h_A_t, h_B=h_B_t, beta_t=beta[t+1], a_A=a_A, b_A=b_A, a_B=self.a, b_B=self.b, W_B=self.W, sumout=sumout)
#                    elif t==T-2:
#                        log_p_star_t_xh_t=self.log_p_star_t(x=x_t, h_A=h_A_t, h_B=h_B_t, beta_t=beta[t], a_A=a_A, b_A=b_A, a_B=self.a, b_B=self.b, W_B=self.W, sumout=sumout)
#                        log_p_star_tplus1_xh_t=0
#                    else:
#                        log_p_star_t_xh_t=self.log_p_star_t(x=x_t, h_A=h_A_t, h_B=h_B_t, beta_t=beta[t], a_A=a_A, b_A=b_A, a_B=self.a, b_B=self.b, W_B=self.W, sumout=sumout)
#                    log_p_star_tplus1_xh_t=self.log_p_star_t(x=x_t, h_A=h_A_t, h_B=h_B_t, beta_t=beta[t+1], a_A=a_A, b_A=b_A, a_B=self.a, b_B=self.b, W_B=self.W, sumout=sumout)
#                else:
#                    log_p_star_t_xh_t=self.log_p_star_t(x=x_t, h_A=h_A_t, h_B=h_B_t, beta_t=beta[t], a_A=a_A, b_A=b_A, a_B=self.a, b_B=self.b, W_B=self.W, sumout=sumout)
#                    log_p_star_tplus1_xh_t=self.log_p_star_t(x=x_t, h_A=h_A_t, h_B=h_B_t, beta_t=beta[t+1], a_A=a_A, b_A=b_A, a_B=self.a, b_B=self.b, W_B=self.W, sumout=sumout)
                log_p_star_t_xh_t=self.log_p_star_t(x=x_t, h_A=h_A_t, h_B=h_B_t, beta_t=beta[t], a_A=a_A, b_A=b_A, a_B=self.a, b_B=self.b, W_B=self.W, sumout=sumout)
                log_p_star_tplus1_xh_t=self.log_p_star_t(x=x_t, h_A=h_A_t, h_B=h_B_t, beta_t=beta[t+1], a_A=a_A, b_A=b_A, a_B=self.a, b_B=self.b, W_B=self.W, sumout=sumout)
#                if self.visible_type=="NegativeBinomial":
#                    if numpy.isinf(log_p_star_t_xh_t):
#                        log_p_star_t_xh_t=0
#                    if numpy.isinf(log_p_star_tplus1_xh_t):
#                        log_p_star_tplus1_xh_t=0
                    
                log_p_star_diff[t]=log_p_star_tplus1_xh_t-log_p_star_t_xh_t
            
            #print log_p_star_diff
            logws[s]=numpy.sum(log_p_star_diff)
            log_ratio_AIS_mean=numpy.mean(logws)
            #ws[s]=numpy.exp(logws[s])
            #ratio_AIS_mean=numpy.mean(ws)
            if S>1: # multiple runs of AIS
                log_ratio_AIS_std=numpy.std(logws)
            else:
                log_ratio_AIS_std=0
        #logZ_B=numpy.log(ratio_AIS_mean) + logZ_A
        logZ_B=log_ratio_AIS_mean + logZ_A
        print("log_ratio_AIS_mean={0}".format(log_ratio_AIS_mean))
        logZ_B_std= numpy.std(logws+logZ_A)
        #print logws
        #print logZ_A
        #print logZ_B
        #print logZ_B_std
        #print log_ratio_AIS_mean
        #print log_ratio_AIS_std
        return logZ_B,logZ_B_std,logws,log_ratio_AIS_mean,log_ratio_AIS_std
            

    def compute_logZ_A(self, a_B, b_B, visible_type="Bernoulli", visible_type_fixed_param=None, hidden_type="Bernoulli", hidden_type_fixed_param=None, base_rate_type="prior"):
        """
        Compute the log-partition function of the prior or uniform base-rate model A.
        a_B, b_B: the biases of model B.
        """
        # define the parameters of DBM_A first
        NK=len(b_B)
        if base_rate_type=="prior":
            a_A=a_B
            b_A=b_B
        elif base_rate_type=="uniform":    
            a_A=numpy.zeros(shape=a_B.shape,dtype=float)
            b_A=[None]*NK
            for nk in range(NK):
                b_A[nk]=numpy.zeros(shape=b_B[nk].shape,dtype=float)

            if visible_type=="Gaussian":
                a_A1=numpy.zeros(shape=a_B[0].shape,dtype=float)
                a_A2=math.pi*numpy.ones(shape=a_B[1].shape,dtype=float)
                a_A=[a_A1,a_A2]
            elif visible_type=="Gaussian_FixPrecision1" or visible_type=="Gaussian_FixPrecision2":
                a_A=numpy.zeros(shape=a_B.shape,dtype=float)
            elif visible_type=="NegativeBinomial":
                a_A=numpy.log(0.5)*numpy.ones(shape=a_B.shape,dtype=float)
            elif visible_type=="Multinoulli":
                M=len(a_B)
                a_A=[None]*M
                for m in range(M):
                    a_A[m]=numpy.zeros(shape=a_B[m].shape,dtype=float)
            elif visible_type=="Gamma":
                a_A1=numpy.ones(shape=a_B[0].shape,dtype=float)
                a_A2=-numpy.ones(shape=a_B[1].shape,dtype=float)
                a_A=[a_A1,a_A2]
            
            if hidden_type=="Gaussian":
                for nk in range(NK):
                    b_Ank1=numpy.zeros(shape=b_B[nk][0].shape,dtype=float)
                    b_Ank2=-math.pi*numpy.ones(shape=b_B[nk][1].shape,dtype=float)
                b_A[nk]=[b_Ank1,b_Ank2]
            elif hidden_type=="Gaussian_FixPrecision1" or hidden_type=="Gaussian_FixPrecision2":
                for nk in range(NK):
                    b_Ank=numpy.zeros(shape=b_B[nk].shape,dtype=float)
                    b_A[nk]=b_Ank
            elif hidden_type=="NegativeBinomial":
                for nk in range(NK):
                    b_Ank=numpy.log(0.5)*numpy.ones(shape=b_B[nk].shape,dtype=float)
                    b_A[nk]=b_Ank
            elif hidden_type=="Multinoulli":
                for nk in range(NK):
                    b_Ank=[]
                    M=len(b_B[nk])
                    for m in range(M):
                        b_Ank[m]=numpy.zeros(shape=b_B[nk][m].shape,dtype=float)
                    b_A[nk]=b_Ank
            elif hidden_type=="Gamma":
                for nk in range(NK):
                    b_Ank1=numpy.ones(shape=b_B[nk][0].shape,dtype=float)
                    b_Ank2=-numpy.ones(shape=b_B[nk][1].shape,dtype=float)
                b_A[nk]=[b_Ank1,b_Ank2]
                
        # master equation for log-partition function of model A
        rbm=restricted_boltzmann_machine.restricted_boltzmann_machine(features=None, M=self.M, K=self.K[0], visible_type=visible_type, visible_type_fixed_param=visible_type_fixed_param, hidden_type="Bernoulli", hidden_type_fixed_param=1, tie_W_for_pretraining_DBM_bottom=False, tie_W_for_pretraining_DBM_top=False, if_fix_vis_bias=False, a=None, fix_a_log_ind=None, rng=numpy.random.RandomState(100))
        logZ_A=rbm.A(a_A,fixed_param=visible_type_fixed_param, distribution=visible_type) 
        for nk in range(NK): 
            logZ_Bnk=rbm.A(b_A[nk],fixed_param=hidden_type_fixed_param, distribution=hidden_type)
            logZ_A=logZ_A + logZ_Bnk
        return a_A,b_A,logZ_A
 
 
    def combine_a_A_a_B_hat(self, a_A, a_B_hat, beta_t, visible_type):
        if visible_type=="Bernoulli" or visible_type=="Poisson" or visible_type=="Guassian_FixPrecision" or visible_type=="Guassian_FixPrecision2" or visible_type=="NegativeBinomial" or visible_type=="Multinomial":
            a_hat_t=(1-beta_t)*a_A + beta_t*a_B_hat
        elif visible_type=="Gaussian" or visible_type=="Gamma":
            a_hat_t=[None,None]*2
            a_hat_t[0]=(1-beta_t)*a_A[0] + beta_t*a_B_hat[0]
            a_hat_t[1]=(1-beta_t)*a_A[1] + beta_t*a_B_hat[1]
        return a_hat_t
        
               
    def log_p_star_t(self, x, h_A, h_B, beta_t, a_A, b_A, a_B, b_B, W_B, sumout="auto"):
        """
        Computer the un-normalized intermediate log distribution log p*_t(x,h^(2),h^(4),...) .
        """
        if sumout=="auto":
            if self.NK%2==1: # NK is odd, such as x, h1, h2, h3, h4, h5
                sumout="odd" # sum out h1, h3, h5; equlvalently, x, h2, h4 can be summed out
            if self.NK%2==0: # NK is even, such as x, h1, h2, h3, h4
                sumout="even" # sum out x, h2, h4
        
        if beta_t==1:
            zeta_A=0
        else:
            zeta_A=(1-beta_t) * self.zeta(X=x, H=h_A, a=a_A, b=b_A, visible_type=self.visible_type, visible_type_fixed_param=self.visible_type_fixed_param, sumout=sumout)

        if beta_t==0:
            zeta_B=0
        else:
            zeta_B=beta_t * self.zeta(X=x, H=h_B, a=a_B, b=b_B, visible_type=self.visible_type, visible_type_fixed_param=self.visible_type_fixed_param, sumout=sumout)
        
        if beta_t==1:
            logPar_A=0
        else:
            a_hat_A,b_hat_A=self.compute_posterior_bias(visible_type=self.visible_type, visible_type_fixed_param=self.visible_type_fixed_param, a=a_A, b=b_A, W=None, X=None, H=None, scale=1-beta_t )
            logPar_A=self.A(a_hat=a_hat_A, b_hat=b_hat_A, visible_type=self.visible_type, visible_type_fixed_param=self.visible_type_fixed_param, hidden_type=self.hidden_type, hidden_type_fixed_param=self.hidden_type_fixed_param, sumout=sumout )
        
        if beta_t==0:
            logPar_B=0
        else:
            a_hat_B,b_hat_B=self.compute_posterior_bias(visible_type=self.visible_type, visible_type_fixed_param=self.visible_type_fixed_param, a=a_B, b=b_B, W=W_B, X=x, H=h_B, scale=beta_t )
            logPar_B=self.A(a_hat=a_hat_B, b_hat=b_hat_B, visible_type=self.visible_type, visible_type_fixed_param=self.visible_type_fixed_param, hidden_type=self.hidden_type, hidden_type_fixed_param=self.hidden_type_fixed_param, sumout=sumout )
            
        log_p_star_t_xh = zeta_A + logPar_A + zeta_B + logPar_B
         
        return log_p_star_t_xh


    def zeta(self, X=None, H=None, a=None, b=None, visible_type="Bernoulli", visible_type_fixed_param=None, NK=None, sumout="auto"):
        """
        sumout: string, can be one from {"auto", "odd", "even", or None} 
        NK: the number of hidden layer. When NK==0, this function is probably called from MDBM, in which case, only compute zeta for the visible layer.
        """
        if NK is None:
            NK=len(b)
            
        if sumout=="auto":
            if NK%2==1: # NK is odd, such as x, h1, h2, h3, h4, h5
                sumout="odd" # sum out h1, h3, h5; equlvalently, x, h2, h4 can be summed out
            if NK%2==0: # NK is even, such as x, h1, h2, h3, h4
                sumout="even" # sum out x, h2, h4
        
        rbm=restricted_boltzmann_machine.restricted_boltzmann_machine(M=100, K=100) # create a rbm for call its zeta function, the initial parameters of this RBM does not matter
        if sumout=="odd": # keep x, h2, h4
            z=rbm.zeta(a, X, fixed_param=visible_type_fixed_param, distribution=visible_type)
            layer_inds=list(range(1,NK,2)) # even
            for nk in layer_inds:
                z = z + rbm.zeta(b[nk], H[nk], fixed_param=None, distribution="Bernoulli")
        if sumout=="even": # keep h1, h2, h5
            layer_inds=list(range(0,NK,2))
            z=0
            for nk in layer_inds:
                z = z + rbm.zeta(b[nk], H[nk], fixed_param=None, distribution="Bernoulli")
        if sumout is None: # keep all hidden layers
            z=rbm.zeta(a, X, fixed_param=visible_type_fixed_param, distribution=visible_type)
            layer_inds=list(range(NK)) # all hidden layers
            for nk in layer_inds:
                z = z + rbm.zeta(b[nk], H[nk], fixed_param=None, distribution="Bernoulli")
        return z


    def compute_posterior_bias(self, visible_type="Bernoulli", visible_type_fixed_param=None, a=None, b=None, W=None, X=None, H=None, scale=1):
        """
        Compute the posterior bias given parameters and data.
        """
        
        # visible
        if visible_type=="Bernoulli" or visible_type=="Poisson" or visible_type=="NegativeBinomial" or visible_type=="Multinomial" or visible_type=="Gaussian_FixPrecision1" or visible_type=="Gaussian_FixPrecision2":
            if H is not None:
                a_hat=a + numpy.dot(W[0],H[0])
            else:
                a_hat=a
            a_hat= scale*a_hat
        elif visible_type=="Gaussian":
            if H is not None:
                a_hat1=a[0] + numpy.dot(W[0],H[0])
            else:
                a_hat1=a[0]
            a_hat2=a[1]
            a_hat=[scale*a_hat1, scale*a_hat2]
        elif visible_type=="Gamma":
            a_hat1=a[0]
            if H is not None:
                a_hat2=a[1] + numpy.dot(W[0],H[0])
            else:
                a_hat2=a[1]            
            a_hat=[scale*a_hat1, scale*a_hat2]
        elif visible_type=="Multinoulli":
            M=len(a)
            a_hat=[None]*M
            for m in range(M):
                if H is not None:
                    a_hat[m]=scale* (a[m] + numpy.dot(W[0][m],H[0]) )
                else:
                    a_hat[m]=scale* a[m]
        
        # hidden
        if X is not None and visible_type=="Gaussian_FixPrecision1":        
            X=visible_type_fixed_param * X         
        NK=len(b)
        b_hat=[None]*NK
        for nk in range(NK):
            if NK>1:
                if nk==0:
                    if visible_type=="Multinoulli":
                        if X is not None:
                            b_hat[nk]= scale*( b[nk] + numpy.dot(W[nk+1], H[nk+1] ) )
                            for m in range(M):
                                b_hat[nk]= b_hat[nk] + scale*( numpy.dot(W[nk][m].transpose(), X[m] ) )
                        else: # H is None
                            b_hat[nk]= scale*b[nk]
                    else: # not Multinoulli
                        if X is not None:
                            b_hat[nk]= scale*( b[nk] + numpy.dot(W[nk].transpose(), X ) + numpy.dot(W[nk+1], H[nk+1] ) )
                        else:
                            b_hat[nk]= scale* b[nk]
                elif nk==NK-1:
                    if X is not None:
                        b_hat[nk]= scale*( b[nk] + numpy.dot(W[nk].transpose(), H[nk-1] ) )
                    else:
                        b_hat[nk]= scale*b[nk]
                else: # middle
                    if X is not None:
                        b_hat[nk]= scale*( b[nk] + numpy.dot(W[nk].transpose(), H[nk-1] ) + numpy.dot(W[nk+1], H[nk+1]) )
                    else:
                        b_hat[nk]= scale* b[nk]
            else: # NK=1
                if visible_type=="Multinoulli":
                    b_hat[nk]= scale* b[nk]
                    if X is not None:
                        for m in range(M):
                            b_hat[nk]= b_hat[nk] + scale*( numpy.dot(W[nk][m].transpose(), X[m] ) )
                else:
                    b_hat[nk]= scale* b[nk]
                    if X is not None:
                        b_hat[nk]= b_hat[nk] + scale*( numpy.dot(W[nk].transpose(), X ) )
        
        return a_hat,b_hat
        
        
    def A(self, a_hat=None, b_hat=None, visible_type="Bernoulli", visible_type_fixed_param=None, hidden_type="Bernoulli", hidden_type_fixed_param=None, sumout="auto"):
        """
        Compute the sum of log-partition functions of specific layers.
        """
        NK=len(b_hat)
        
        if numpy.isscalar(hidden_type):
            hidden_type=[hidden_type]*NK
        if numpy.isscalar(hidden_type_fixed_param):
            hidden_type_fixed_param=[hidden_type_fixed_param]*NK
        
        if sumout=="auto":
            if NK%2==1: # NK is odd, such as x, h1, h2, h3, h4, h5
                sumout="odd" # sum out h1, h3, h5; equlvalently, x, h2, h4 can be summed out
            if NK%2==0: # NK is even, such as x, h1, h2, h3, h4
                sumout="even" # sum out x, h2, h4
                
        rbm=restricted_boltzmann_machine.restricted_boltzmann_machine(M=100,K=100) # call the A() function from RBM, the initial parameters of this RBM does not matter
        if sumout=="odd": # add A() for h1, h3, h5
            logPar=0
            for nk in range(0,NK,2):
                logPar = logPar + rbm.A(theta=b_hat[nk], fixed_param=hidden_type_fixed_param[nk], distribution=hidden_type[nk])
        if sumout=="even": # add A() for x, h2, h4
            if a_hat is None:
                logPar=0
            else:
                logPar=rbm.A(theta=a_hat,fixed_param=visible_type_fixed_param, distribution=visible_type)
            for nk in range(1,NK,2):
                logPar = logPar + rbm.A(theta=b_hat[nk], fixed_param=hidden_type_fixed_param[nk], distribution=hidden_type[nk])
                
        return logPar
    

    def mean_field_approximate_inference(self,Xbatch=None,NMF=20,rand_init_H=False):
        """
        Mean-field method to approximate the data dependent expectation. This function can also be used for infere p(h|v).
        Xbatch: list of numpy matrices with each sample in a row, a batch of training samples.
        NMF: number of iterations of the mean-field approximation.
        rand_init_H: bool, whether randomly initialize the hidden layers or use marginals from RBMs.
        """

        #self.NMF=NMF # number of iterations in mean-field approximation

#        # randomly initialize Hbatch
        Hbatch=[None]*self.NK
        HbatchP=[None]*self.NK
#        for nk in range(self.NK):
#            if not rand_init_H: # initialize using marginals
#                if nk==0:
#                    rbm_X=Xbatch
#                else:
#                    rbm_X,_=self.rbms[nk-1].sample_h_given_x(rbm_X)
#                _,Hrand=self.rbms[nk].sample_h_given_x(rbm_X)
#            else: # initialize randomly
#                Hrand=self.rng.random_sample(size=(self.K[nk],self.batch_size)) # random numbers from [0,1)
#            HbatchP[nk]=Hrand

        # get number of samples in this mean-field approximation
        if self.visible_type=="Multinoulli":
            NS=Xbatch[0].shape[1]
        else:
            NS=Xbatch.shape[1]
        if rand_init_H: # randomly initialize H
            for nk in range(self.NK):
                Hrand=self.rng.random_sample(size=(self.K[nk],self.batch_size)) # random numbers from [0,1)
                HbatchP[nk]=Hrand
        else: # marginal to initialize
            Hbatch,HbatchP=self.sample_hidden(visible_type=self.visible_type, visible_type_fixed_param=self.visible_type_fixed_param, hidden_type=self.hidden_type, hidden_type_fixed_param=self.hidden_type_fixed_param, NS=NS, X=None, H=None, a=None, b=self.b, W=None, beta_t=1, even_or_odd_first="even", rng=self.rng)
        
        # if there is only one layer, just run it once
        if self.NK==1:
            NMF=1

        for i in range(NMF):
            Hbatch,HbatchP=self.sample_hidden(visible_type=self.visible_type, visible_type_fixed_param=self.visible_type_fixed_param, hidden_type=self.hidden_type, hidden_type_fixed_param=self.hidden_type_fixed_param, X=Xbatch, H=HbatchP, a=self.a, b=self.b, W=self.W, beta_t=1, even_or_odd_first="even", rng=self.rng)

            # in the last iteration, compute the recovered observed visible too.
            if i==NMF-1:
                XbatchR,XbatchRM,XbatchRP=self.sample_visible(visible_type=self.visible_type, a=self.a, W=self.W[0], H=HbatchP[0], visible_type_fixed_param=self.visible_type_fixed_param, tie_W_for_pretraining_DBM_top=False, NS=None, rng=self.rng )

        # return mean: XbatchR: value, XbatchRM: mean, XbatchRP: prob, Hbatch: value, HbatchP: probabilities
        return XbatchR,XbatchRM,XbatchRP,Hbatch,HbatchP

    
    def sample_hidden(self, visible_type="Bernoulli", visible_type_fixed_param=0, hidden_type="Bernoulli", hidden_type_fixed_param=0, NS=100, X=None, H=None, a=None, b=None, W=None, beta_t=1, even_or_odd_first="even", rng=numpy.random.RandomState(100)):
        """
        Sample the hidden variables.
        If X is None, it will sample NS samples using marginals.
        """

        NK=len(b)
        
        if numpy.isscalar(hidden_type):
            hidden_type=[hidden_type]*NK
        if numpy.isscalar(hidden_type_fixed_param):
            hidden_type_fixed_param=[hidden_type_fixed_param]*NK            
            
        # get indices for hidden layers
        if even_or_odd_first=="even":
            layer_inds=list(range(1,NK,2)) # even
            layer_inds.extend(list(range(0,NK,2))) # +odd
        else:
            layer_inds=list(range(0,NK,2)) # odd
            layer_inds.extend(list(range(1,NK,2))) # +even
        if H is None:
            H=[None]*NK
        HP=[None]*NK
        if X is not None and visible_type=="Gaussian_FixPrecision1":
            X=visible_type_fixed_param*X
        for nk in layer_inds:
            if nk==0 and NK>1: # first hidden layer
                if visible_type=="Bernoulli" or visible_type=="Gaussian" or visible_type=="Gaussian_FixPrecision1" or visible_type=="Gaussian_FixPrecision2" or visible_type=="Poisson" or visible_type=="NegativeBinomial" or visible_type=="Multinomial" or visible_type=="Gamma":
                    if X is not None:
                        b_hat= b[nk] + numpy.dot(numpy.transpose(W[nk]),X) + numpy.dot(W[nk+1],H[nk+1])
                    else:
                        b_hat=numpy.repeat(b[nk], NS, axis=1)
                    b_hat=beta_t * b_hat
                    Hnk,Pnk=self.sample_h_given_b_hat(b_hat=b_hat, visible_type=visible_type, visible_type_fixed_param=visible_type_fixed_param, hidden_type=hidden_type[nk], hidden_type_fixed_param=hidden_type_fixed_param[nk], hidden_value_or_meanfield="value")
                elif visible_type=="Multinoulli":
                    if X is not None:
                        b_hat=b[nk] + numpy.dot(W[nk+1],H[nk+1])
                        M=len(X)
                        for m in range(M):
                            b_hat=b_hat + numpy.dot(numpy.transpose(W[nk][m]),X[m])
                    else:
                        b_hat=numpy.repeat(b[nk], NS, axis=1)
                    b_hat=beta_t * b_hat
                    Hnk,Pnk=self.sample_h_given_b_hat(b_hat=b_hat, visible_type=visible_type, visible_type_fixed_param=visible_type_fixed_param, hidden_type=hidden_type[nk], hidden_type_fixed_param=hidden_type_fixed_param[nk], hidden_value_or_meanfield="value")
                elif visible_type=="Gaussian_Hinton":
                    if X is not None:
                        b_hat=b[nk] + numpy.dot(numpy.transpose(W[nk]),numpy.diag(a[1]).dot(X)) + numpy.dot(W[nk+1],H[nk+1])
                    else:
                        b_hat=numpy.repeat(b[nk], NS, axis=1)
                    b_hat=beta_t * b_hat
                    Hnk,Pnk=self.sample_h_given_b_hat(b_hat=b_hat, visible_type=visible_type, visible_type_fixed_param=visible_type_fixed_param, hidden_type=hidden_type[nk], hidden_type_fixed_param=hidden_type_fixed_param[nk], hidden_value_or_meanfield="value")
            elif nk==NK-1 and NK>1: # last but not the first hidden layer
                if X is not None:
                    b_hat=b[nk] +  numpy.dot(numpy.transpose(W[nk]),H[nk-1])
                else:
                    b_hat=numpy.repeat(b[nk], NS, axis=1)
                b_hat=beta_t * b_hat
                Hnk,Pnk=self.sample_h_given_b_hat(b_hat=b_hat, visible_type=visible_type, visible_type_fixed_param=visible_type_fixed_param, hidden_type=hidden_type[nk], hidden_type_fixed_param=hidden_type_fixed_param[nk], hidden_value_or_meanfield="value")
            elif nk==0 and NK==1: # there is only one hidden layer in this DBM, actually it is a RBM!
                if visible_type=="Bernoulli" or visible_type=="Gaussian" or visible_type=="Gaussian_FixPrecision1" or visible_type=="Gaussian_FixPrecision2" or visible_type=="Poisson" or visible_type=="NegativeBinomial" or visible_type=="Multinomial":
                    if X is not None:
                        b_hat=b[nk] + numpy.dot(numpy.transpose(W[nk]),X)
                    else:
                        b_hat=numpy.repeat(b[nk], NS, axis=1)
                    b_hat=beta_t * b_hat
                    Hnk,Pnk=self.sample_h_given_b_hat(b_hat=b_hat, visible_type=visible_type, visible_type_fixed_param=visible_type_fixed_param, hidden_type=hidden_type[nk], hidden_type_fixed_param=hidden_type_fixed_param[nk], hidden_value_or_meanfield="value")
                elif visible_type=="Multinoulli":
                    if X is not None:
                        b_hat=b[nk]
                        for m in range(M):
                            b_hat= b_hat + numpy.dot(numpy.transpose(W[nk][m]),X[m])
                    else:
                        b_hat=numpy.repeat(b[nk], NS, axis=1)    
                    b_hat=beta_t * b_hat
                    Hnk,Pnk=self.sample_h_given_b_hat(b_hat=b_hat, visible_type=visible_type, visible_type_fixed_param=visible_type_fixed_param, hidden_type=hidden_type[nk], hidden_type_fixed_param=hidden_type_fixed_param[nk], hidden_value_or_meanfield="value")
                elif visible_type=="Gaussian_Hinton":
                    if X is not None:
                        b_hat=b[nk] + numpy.dot(numpy.transpose(W[nk]),numpy.diag(a[1]).dot(X))
                    else:
                        b_hat=numpy.repeat(b[nk], NS, axis=1)
                    b_hat=beta_t * b_hat
                    Hnk,Pnk=self.sample_h_given_b_hat(b_hat=b_hat, visible_type=visible_type, visible_type_fixed_param=visible_type_fixed_param, hidden_type=hidden_type[nk], hidden_type_fixed_param=hidden_type_fixed_param[nk], hidden_value_or_meanfield="value")
            else: # in the middle
                if X is not None:
                    b_hat=b[nk] + numpy.dot(numpy.transpose(W[nk]),H[nk-1]) + numpy.dot(W[nk+1],H[nk+1])
                else:
                    b_hat=numpy.repeat(b[nk], NS, axis=1)
                b_hat=beta_t * b_hat
                Hnk,Pnk=self.sample_h_given_b_hat(b_hat=b_hat, visible_type=visible_type, visible_type_fixed_param=visible_type_fixed_param, hidden_type=hidden_type[nk], hidden_type_fixed_param=hidden_type_fixed_param[nk], hidden_value_or_meanfield="value")
            HP[nk]=Pnk
            H[nk]=Hnk
                
        return H,HP


    def sample_h_given_b_hat(self, b_hat=None, visible_type="Bernoulli", visible_type_fixed_param=0, hidden_type="Bernoulli", hidden_type_fixed_param=0, hidden_value_or_meanfield="value"):
        """
        In a shallow set.
        """
        #H=X # initialize H
        #num=X.shape[1]
        #for n in range(num):
        #    b=numpy.copy(self.b)
        #    b.shape=(self.K,)
        #    h_prob=cl.sigmoid(b + numpy.dot(self.W.transpose(),X[:,n])) 
        #    h=numpy.zeros(shape=(self.K,),dtype=int)
        #    for k in range(self.K):
        #        h[k]=self.rng.binomial(n=1,h_prob[k],size=1)
        #    H[:,n]=h
            
        # sampling
        if hidden_type=="Bernoulli":
            P=cl.sigmoid(b_hat)
            HM=P # mean of hidden variables
            if hidden_value_or_meanfield=="value":
                H=cl.Bernoulli_sampling(P,rng=self.rng)
            elif hidden_value_or_meanfield=="meanfield":
                H=HM
        elif hidden_type=="Binomial":
            P=cl.sigmoid(b_hat) # probability
            HM=hidden_type_fixed_param*P # mean
            if hidden_value_or_meanfield=="value":
                # hidden_type_fixed_param is the number of trials
                H=cl.Binomial_sampling(hidden_type_fixed_param, P, rng=self.rng)
            elif hidden_value_or_meanfield=="meanfield":
                H=HM
        elif hidden_type=="Multinomial":
            P=numpy.exp(b_hat) # probability
            P=cl.normalize_probability(P)
            HM=hidden_type_fixed_param*P # mean
            if hidden_value_or_meanfield=="value":
                # hidden_type_fixed_param is the number of trials
                H=cl.Multinomial_sampling(hidden_type_fixed_param, P=P, rng=self.rng)
            elif hidden_value_or_meanfield=="meanfield":
                H=HM
                
        return H,HM


    def sample_visible(self, visible_type="Bernoulli", a=None, W=None, H=None, visible_type_fixed_param=10, tie_W_for_pretraining_DBM_top=False, NS=None, rng=numpy.random.RandomState(100)):
        if H is not None:
            if visible_type=="Bernoulli":
                if tie_W_for_pretraining_DBM_top:
                    a_hat=a + 2*numpy.dot(W,H)
                else:
                    a_hat=a + numpy.dot(W,H)
                P=cl.sigmoid( a_hat )
                XM=P
                X=cl.Bernoulli_sampling(P,rng=rng)
            elif visible_type=="Gaussian":
                a1=a[0]
                a2=a[1]
                if tie_W_for_pretraining_DBM_top:
                    a_hat1=a1 + 2*numpy.dot(W,H) # mean
                else:
                    a_hat1=a1 + numpy.dot(W,H) 
                a_hat2=a2
                XM=-a_hat1/(2*a_hat2)
                P=None
                X=cl.Gaussian_sampling(XM,-2*a_hat2,rng=rng)
            elif visible_type=="Gaussian_FixPrecision1":
                if tie_W_for_pretraining_DBM_top:
                    a_hat=a + 2*numpy.dot(W,H)
                else:
                    a_hat=a + numpy.dot(W,H)
                XM=a_hat
                P=None
                X=cl.Gaussian_sampling(XM,visible_type_fixed_param,rng=rng)
            elif visible_type=="Gaussian_FixPrecision2":
                if tie_W_for_pretraining_DBM_top:
                    a_hat=a + 2*numpy.dot(W,H)
                else:
                    a_hat=a + numpy.dot(W,H)
                XM=a_hat/visible_type_fixed_param
                P=None
                X=cl.Gaussian_sampling(XM,visible_type_fixed_param,rng=rng)
            elif visible_type=="Gaussian_Hinton":
                a1=a[0]
                a2=a[1]
                if tie_W_for_pretraining_DBM_top:
                    a_hat1=a1 + 2/a2*numpy.dot(W,H) # mean
                else:
                    a_hat1=a1 + 1/a2*numpy.dot(W,H) 
                XM=a_hat1
                P=None
                X=cl.Gaussian_sampling(a_hat1,a2,rng=rng)
            elif visible_type=="Poisson": 
                tol_poisson_max=self.tol_poisson_max
                if tie_W_for_pretraining_DBM_top:
                    a_hat=a + 2*numpy.dot(W,H)
                else:
                    a_hat=a + numpy.dot(W,H)
                a_hat[a_hat>tol_poisson_max]=tol_poisson_max
                XM=numpy.exp(a_hat)
                P=None
                X=cl.Poisson_sampling(XM,rng=rng)
            elif visible_type=="NegativeBinomial": 
                if tie_W_for_pretraining_DBM_top:
                    a_hat= a + 2*numpy.dot(W,H) 
                else:
                    a_hat= a + numpy.dot(W,H) # a_hat should be negative
                tol_negbin_max=-1e-8
                tol_negbin_min=-100
                a_hat[a_hat>=0]=tol_negbin_max
                a_hat[a_hat<tol_negbin_min]=tol_negbin_min
                P_failure=numpy.exp(a_hat)
                P=P_failure
                P_success=1-P_failure
                #print "max: {}".format(numpy.max(P_failure))
                #print "min: {}".format(numpy.min(P_failure))
                XM=visible_type_fixed_param*(P_failure/P_success)
                X=cl.NegativeBinomial_sampling(K=visible_type_fixed_param,P=P_success,rng=rng)
            elif visible_type=="Multinomial":
                if tie_W_for_pretraining_DBM_top:
                    a_hat= a + 2*numpy.dot(W,H) 
                else:
                    a_hat= a + numpy.dot(W,H)
                P=numpy.exp(a_hat)
                P=cl.normalize_probability(P)
                #print "max: {}".format(numpy.max(P))
                #print "min: {}".format(numpy.min(P))
                XM=visible_type_fixed_param*(P)
                X=cl.Multinomial_sampling(N=visible_type_fixed_param,P=P,rng=rng)
            elif visible_type=="Multinoulli":
                P=[None]*self.M
                XM=[None]*self.M
                X=[None]*self.M
                for m in range(self.M):
                    if tie_W_for_pretraining_DBM_top:
                        a_hat= a[m] + 2*numpy.dot(W[m],H) 
                    else:
                        a_hat= a[m] + numpy.dot(W[m],H)
                    P[m]=numpy.exp(a_hat)
                    P[m]=cl.normalize_probability(P[m])
                    #print "max: {}".format(numpy.max(P))
                    #print "min: {}".format(numpy.min(P))
                    XM[m]=P[m]
                    X[m]=cl.Multinomial_sampling(N=1,P=P[m],rng=rng)
            elif visible_type=="Gamma":
                a1=a[0]
                a2=a[1]
                a_hat1=a1
                if tie_W_for_pretraining_DBM_top:
                    a_hat2=a2 + 2*numpy.dot(W,H) # mean
                else:
                    a_hat2=a2 + numpy.dot(W,H)
                P=None
                X=cl.Gamma_sampling(a_hat1+1, -a_hat2,rng=rng)
            else:
                print("Please choose a correct visible type!")
                
        if H is None: # randomly generate some visible samples without using H and W
            if NS is None:
                NS=1
            if visible_type=="Bernoulli":
                P=cl.sigmoid(numpy.repeat(a,NS,axis=1))
                XM=P
                X=cl.Bernoulli_sampling(P, rng=rng)
            elif visible_type=="Gaussian":
                a1=a[0]
                a2=a[1]
                XM=-numpy.repeat(a1,NS,axis=1)/(2*numpy.repeat(a2,NS,axis=1))
                P=None
                X=cl.Gaussian_sampling(XM, -2*a2, rng=rng)
            elif visible_type=="Gaussian_FixPrecision1":
                XM=a
                P=None
                X=cl.Gaussian_sampling(XM,visible_type_fixed_param,rng=rng)
            elif visible_type=="Gaussian_FixPrecision2":
                XM=a/visible_type_fixed_param
                P=None
                X=cl.Gaussian_sampling(XM,visible_type_fixed_param,rng=rng)
            elif visible_type=="Gaussian_Hinton":
                a1=a[0]
                a2=a[1]
                XM=numpy.repeat(a1,NS,axis=1)
                P=None
                X=cl.Gaussian_sampling(XM, a2, rng=rng)
            elif visible_type=="Poisson":
                tol_poisson_max=self.tol_poisson_max
                a[a>tol_poisson_max]=tol_poisson_max
                XM=numpy.exp(numpy.repeat(a,NS,axis=1))
                P=None
                X=cl.Poisson_sampling(XM, rng=rng)
            elif visible_type=="NegativeBinomial":
                P_failure=numpy.exp(numpy.repeat(a,NS,axis=1))
                P=P_failure
                P_success=1-P_failure
                XM=visible_type_fixed_param * (1-P_success)/P_success
                X=cl.NegativeBinomial_sampling(visible_type_fixed_param, P_success, rng=rng)
            elif visible_type=="Multinomial":
                p_normalized=cl.normalize_probability(numpy.exp(a))
                P=numpy.repeat(p_normalized,NS,axis=1) # DO I NEED TO NORMALIZE IT?
                XM=visible_type_fixed_param * P
                X=cl.Multinomial_sampling(N=visible_type_fixed_param,P=P,rng=rng)
            elif visible_type=="Multinoulli":
                P=[None]*self.M
                XM=[None]*self.M
                X=[None]*self.M
                for m in range(self.M):
                    p_normalized=cl.normalize_probability(numpy.exp(a[m]))
                    P[m]=numpy.repeat(p_normalized,NS,axis=1) # DO I NEED TO NORMALIZE IT?
                XM[m]=P[m]
                X[m]=cl.Multinomial_sampling(N=1,P=P[m],rng=rng)
            elif visible_type=="Gamma":
                a1=a[0]
                a2=a[1]
                a2_rep=numpy.repeat(a2,NS,axis=1)
                P=None
                X=cl.Gamma_sampling(a1+1, -a2_rep,rng=rng)
            else:
                print("Please choose a correct visible type!")
                                        
        return X,XM,P


    def pcd_sampling(self, pcdk=20, NS=20, X0=None, H0=None, clamp_visible=False, persistent=True, init_sampling=False, rand_init_X=True, rand_init_H=False): 
        """
        Persistent contrastive divergence sampling. This function can be used for learning and sampling after learning.
        INPUT: 
        pcdk: integer, steps of a Markov chain to generate a sample.
        NS: integer, number of Markov chains.
        rand_init_X: bool, whether randomly initialize the visible variables or sample some training samples as initial points.
        init_sampling: bool, whether call this function to initialize the Markov chains.
        rand_init_H: bool, whether randomly initialize the hidden layers or use the marginals from RBMs.
        OUTPUT: 
        self.chainX: list of numpy arrays, the final states of the visible samples of the Markov chains; chainX[nk] of size M by NS: the final states of visible samples.
        self.chainH: list of numpy arrays, the final states of the latent samples of the Markov chains; chainH[nk] of size K[nk] by NS: the final states of the nk-th hidden layer.
        self.chain_length: the length of Markov chains.
        """

        if not persistent:
            if X0 is not None: # X0 should be the current mini-batch
                if self.visible_type=="Multinoulli":
                    NS=X0[0].shape[1]
                else:
                    NS=X0.shape[1]
                init_sampling=True
            else:
                print("Error! You want to use CD-k sampling, but you did not give me a batch of training samples.")
                exit()

        if init_sampling: # This is the procedure in the beginning of PCD-sampling.
            # initialize Markov chains
        
            if clamp_visible:
                if X0 is None:
                    print("Error! You want to fix the input during sampling, but you did not give me an initial value for X0.")
                    exit()
                    
            #self.pcdk=pcdk
            self.NS=NS
            # randomly initialize H0 from Bernoulli distributions
            #self.chainX=self.rng.binomial(n=1,p=0.5,size=(self.M,self.NS)) # randomly initialize data
            X,XM,XP=self.sample_visible(visible_type=self.visible_type, a=self.a, W=None, H=None, visible_type_fixed_param=self.visible_type_fixed_param, tie_W_for_pretraining_DBM_top=False, NS=self.NS, rng=self.rng )
            if rand_init_X: # random initialize X
                X0=X
            else: # use training samples to initialize X
                if X0 is None:
                    X0=self.sample_minibatch(self.NS)
            self.chainX=copy.deepcopy(X0) #numpy.zeros(shape=(self.M,self.NS),dtype=float)
            self.chainXM=[] # not used here, so just set is empty
            self.chainXP=[] # not used here, so just set is empty


            if not rand_init_H: # initialize using marginals
                if H0 is None: # if H0 is not given, generate it using prior parameters
                    H0,HP=self.sample_hidden(visible_type=self.visible_type, visible_type_fixed_param=self.visible_type_fixed_param, hidden_type=self.hidden_type, hidden_type_fixed_param=self.hidden_type_fixed_param, NS=self.NS, X=None, H=None, a=None, b=self.b, W=None, beta_t=1, even_or_odd_first="even", rng=self.rng)
                else: # assign the given H0
                    HP=[]
                self.chainH=H0
                self.chainHM=HP
            else: # random initialize
                self.chainH=[None]*self.NK
                self.chainHM=[None]*self.NK # probabilities                    
                for nk in range(self.NK):
                    HP=self.rng.random_sample(size=(self.K[nk],NS))
                    H0=cl.Bernoulli_sampling(P=HP,rng=self.rng)
                    self.chainH[nk]=H0
                    self.chainHM[nk]=HP
            self.chain_length=0
            #return self.chainX,self.chainH,self.chainXP,self.chainHP,self.chain_length
        
        # start PCD-sampling
        for c in range(pcdk): # for each step
            # sample visible
            if (c==pcdk-1) or (not clamp_visible):
                chainX,chainXM,chainXP=self.sample_visible(visible_type=self.visible_type, a=self.a, W=self.W[0], H=self.chainH[0], visible_type_fixed_param=self.visible_type_fixed_param, tie_W_for_pretraining_DBM_top=False, NS=None, rng=self.rng )
                self.chainX=chainX
                self.chainXM=chainXM
                self.chainXP=chainXP
                
            # sample hidden
            chainH,chainHM=self.sample_hidden(visible_type=self.visible_type, visible_type_fixed_param=self.visible_type_fixed_param, hidden_type=self.hidden_type, hidden_type_fixed_param=self.hidden_type_fixed_param, X=self.chainX, H=self.chainH, a=self.a, b=self.b, W=self.W, beta_t=1, even_or_odd_first="even", rng=self.rng)
            self.chainH=chainH
            self.chainHM=chainHM
            
            # increase chain length            
            self.chain_length=self.chain_length+1
            
        return self.chainX,self.chainH,self.chainXM,self.chainXP,self.chainHM,self.chain_length


    def sample_xh_given_x(self, X, method="mean_field", num_iter=20, init_chain_time=100):
        """
        sample X and H given X samples.
        INPUTS:
        X: numpy array; each column is an observed sample.
        method: string, method to sample/infer H; can be one from {"mean_field","Gibbs_sampling"}.
        """
        if self.visible_type=="Multinoulli": # convert to binary, may cause problems when there is only sample in X
            X_binary=[None]*self.M
            for m in range(self.M):
                Z,_=cl.membership_vector_to_indicator_matrix(X[m,:],z_unique=list(range(self.Ms[m])))
                X_binary[m]=Z.transpose()
                X=X_binary
                
        if method=="mean_field":
            X,XM,XP,H,HP=self.mean_field_approximate_inference(Xbatch=X,NMF=num_iter,rand_init_H=False)
        elif method=="Gibbs_sampling":
            _,_,_,_,_,_=self.pcd_sampling(pcdk=init_chain_time*num_iter, NS=20, X0=X, clamp_visible=True, persistent=True, init_sampling=True, rand_init_X=False, rand_init_H=False)
            X,H,XM,XP,HP,chain_length=self.pcd_sampling(pcdk=num_iter, clamp_visible=True, init_sampling=False)
        return X,H,XM,XP,HP
    

    def compute_gradient(self,Xbatch,Hbatch,XS,HS):
        """
        Compute gradient.
        """
        
        if self.visible_type=="Bernoulli" or self.visible_type=="Poisson" or self.visible_type=="NegativeBinomial" or self.visible_type=="Multinomial" or self.visible_type=="Gaussian_FixPrecision2":
            # gradient of a: data_dep - data_indep
            data_dep=-numpy.mean(Xbatch,axis=1)
            data_indep=-numpy.mean(XS,axis=1)
            grad_a=data_dep - data_indep
            grad_a.shape=(self.M,1)

            # gradient of b
            grad_b=[]
            for nk in range(self.NK):
                data_dep=-numpy.mean(Hbatch[nk],axis=1)
                data_indep=-numpy.mean(HS[nk],axis=1)
                grad_b_nk=data_dep - data_indep
                grad_b_nk.shape=(self.K[nk],1)
                grad_b.append(grad_b_nk)

            # gradient of W
            grad_W=[]
            for nk in range(self.NK):
                if nk==0:
                    data_dep=-numpy.dot(Xbatch,Hbatch[nk].transpose())/self.batch_size
                    data_indep=-numpy.dot(XS,HS[nk].transpose())/self.NS
                else:
                    data_dep=-numpy.dot(Hbatch[nk-1],Hbatch[nk].transpose())/self.batch_size
                    data_indep=-numpy.dot(HS[nk-1],HS[nk].transpose())/self.NS
                grad_W_nk=data_dep - data_indep
                grad_W.append(grad_W_nk) # gradient of the negtive log-likelihood

            self.grad_a=grad_a
            self.grad_W=grad_W
            self.grad_b=grad_b
            
        elif self.visible_type=="Gaussian_FixPrecision1":
            # gradient of a: data_dep - data_indep
            data_dep=-numpy.mean(self.visible_type_fixed_param*Xbatch,axis=1)
            data_indep=-numpy.mean(self.visible_type_fixed_param*XS,axis=1)
            grad_a=data_dep - data_indep
            grad_a.shape=(self.M,1)

            # gradient of b
            grad_b=[]
            for nk in range(self.NK):
                data_dep=-numpy.mean(Hbatch[nk],axis=1)
                data_indep=-numpy.mean(HS[nk],axis=1)
                grad_b_nk=data_dep - data_indep
                grad_b_nk.shape=(self.K[nk],1)
                grad_b.append(grad_b_nk)

            # gradient of W
            grad_W=[]
            for nk in range(self.NK):
                if nk==0:
                    data_dep=-numpy.dot(self.visible_type_fixed_param*Xbatch,Hbatch[nk].transpose())/self.batch_size
                    data_indep=-numpy.dot(self.visible_type_fixed_param*XS,HS[nk].transpose())/self.NS
                else:
                    data_dep=-numpy.dot(Hbatch[nk-1],Hbatch[nk].transpose())/self.batch_size
                    data_indep=-numpy.dot(HS[nk-1],HS[nk].transpose())/self.NS
                grad_W_nk=data_dep - data_indep
                grad_W.append(grad_W_nk) # gradient of the negtive log-likelihood

            self.grad_a=grad_a
            self.grad_W=grad_W
            self.grad_b=grad_b
            
        elif self.visible_type=="Multinoulli":
            # gradient of a: data_dep - data_indep
            grad_a=[None]*self.M
            for m in range(self.M):
                data_dep=-numpy.mean(Xbatch[m],axis=1)
                data_indep=-numpy.mean(XS[m],axis=1)
                grad_a[m]=data_dep - data_indep
                grad_a[m].shape=(self.Ms[m],1)

            # gradient of b
            grad_b=[]
            for nk in range(self.NK):
                data_dep=-numpy.mean(Hbatch[nk],axis=1)
                data_indep=-numpy.mean(HS[nk],axis=1)
                grad_b_nk=data_dep - data_indep
                grad_b_nk.shape=(self.K[nk],1)
                grad_b.append(grad_b_nk)

            # gradient of W
            grad_W=[]
            for nk in range(self.NK):
                if nk==0:
                    grad_W_input=[None]*self.M
                    for m in range(self.M):
                        data_dep=-numpy.dot(Xbatch[m],Hbatch[nk].transpose())/self.batch_size
                        data_indep=-numpy.dot(XS[m],HS[nk].transpose())/self.NS
                        grad_W_input[m]=data_dep - data_indep
                    grad_W.append(grad_W_input) 
                else:
                    data_dep=-numpy.dot(Hbatch[nk-1],Hbatch[nk].transpose())/self.batch_size
                    data_indep=-numpy.dot(HS[nk-1],HS[nk].transpose())/self.NS
                    grad_W_nk=data_dep - data_indep
                    grad_W.append(grad_W_nk) # gradient of the negtive log-likelihood

            self.grad_a=grad_a
            self.grad_W=grad_W
            self.grad_b=grad_b

        elif self.visible_type=="Gaussian":
            # gradient of a: data_dep - data_indep
            data_dep_x=-numpy.mean(Xbatch,axis=1)
            data_indep_x=-numpy.mean(XS,axis=1)
            #grad_a=numpy.diag(self.beta).dot(data_dep - data_indep)
            grad_a1=data_dep_x - data_indep_x
            grad_a1.shape=(self.M,1)
            
            # gradient of beta
            data_dep=-numpy.mean(Xbatch**2,axis=1)
            data_indep=-numpy.mean(XS**2,axis=1)
            grad_a2=data_dep - data_indep
            grad_a2.shape=(self.M,1)

            # gradient of b
            grad_b=[]
            for nk in range(self.NK):
                data_dep=-numpy.mean(Hbatch[nk],axis=1)
                data_indep=-numpy.mean(HS[nk],axis=1)
                grad_b_nk=data_dep - data_indep
                grad_b_nk.shape=(self.K[nk],1)
                grad_b.append(grad_b_nk)

            # gradient of W
            grad_W=[]
            for nk in range(self.NK):
                if nk==0:
                    data_dep=-numpy.dot(Xbatch,Hbatch[nk].transpose())/self.batch_size
                    data_indep=-numpy.dot(XS,HS[nk].transpose())/self.NS
                else:
                    data_dep=-numpy.dot(Hbatch[nk-1],Hbatch[nk].transpose())/self.batch_size
                    data_indep=-numpy.dot(HS[nk-1],HS[nk].transpose())/self.NS
                grad_W_nk=data_dep - data_indep
                grad_W.append(grad_W_nk) # gradient of the negtive log-likelihood

            self.grad_a=[grad_a1, grad_a2]
            self.grad_W=grad_W
            self.grad_b=grad_b
            
        elif self.visible_type=="Gaussian_Hinton":
            # gradient of a: data_dep - data_indep
            data_dep_x=-numpy.mean(Xbatch,axis=1)
            data_indep_x=-numpy.mean(XS,axis=1)
            #grad_a=numpy.diag(self.beta).dot(data_dep - data_indep)
            grad_a1=self.a[1] * (data_dep_x - data_indep_x)
            grad_a1.shape=(self.M,1)
            
            # gradient of beta
            #data_dep_beta=(data_dep_x - self.a)**2 - 1/numpy.sqrt(self.beta)*data_dep_x*self.W[0].dot(data_dep_h[0]) 
            #data_indep_beta=(data_indep_x - self.a)**2 - 1/numpy.sqrt(self.beta)*data_indep_x*self.W[0].dot(data_indep_h[0])
            data_dep_a2=numpy.mean((Xbatch-self.a[0])**2,axis=1) - numpy.mean(1/self.a[1]*Xbatch*self.W[0].dot(Hbatch[0]),axis=1)
            data_indep_a2=numpy.mean((XS-self.a[0])**2,axis=1) - numpy.mean(1/self.a[1]*XS*self.W[0].dot(HS[0]),axis=1)
            grad_a2=data_dep_a2 - data_indep_a2
            grad_a2.shape=(self.M,1)

            # gradient of b
            grad_b=[]
            data_dep_h=[None]*self.NK
            data_indep_h=[None]*self.NK
            for nk in range(self.NK):
                data_dep_h[nk]=-numpy.mean(Hbatch[nk],axis=1)
                data_indep_h[nk]=-numpy.mean(HS[nk],axis=1)
                grad_b_nk=data_dep_h[nk] - data_indep_h[nk]
                grad_b_nk.shape=(self.K[nk],1)
                grad_b.append(grad_b_nk)

            # gradient of W
            grad_W=[]
            for nk in range(self.NK):
                if nk==0:
                    data_dep_Wnk=-numpy.dot(numpy.sqrt(self.a[1])*Xbatch,Hbatch[nk].transpose())/self.batch_size
                    data_indep_Wnk=-numpy.dot(numpy.sqrt(self.a[1])*XS,HS[nk].transpose())/self.NS
                else:
                    data_dep_Wnk=-numpy.dot(Hbatch[nk-1],Hbatch[nk].transpose())/self.batch_size
                    data_indep_Wnk=-numpy.dot(HS[nk-1],HS[nk].transpose())/self.NS
                grad_W_nk=data_dep_Wnk - data_indep_Wnk
                grad_W.append(grad_W_nk) # gradient of the negtive log-likelihood

            self.grad_a=[grad_a1,grad_a2]
            self.grad_W=grad_W
            self.grad_b=grad_b


    def update_param(self):
        """
        Update parameters.
        """
        
        #tol=1e-8
        tol_negbin_max=-1e-8
        tol_negbin_min=-100
        tol_poisson_max=self.tol_poisson_max#16 #numpy.log(255)
        #tol_gamma_min=1e-3
        #tol_gamma_max=1e3
        
        if self.if_fix_vis_bias:
            fix_a_log_ind=self.fix_a_log_ind
            not_fix_a_log_ind=numpy.logical_not(fix_a_log_ind)
            not_fix_a_log_ind=numpy.array(not_fix_a_log_ind,dtype=int)
            not_fix_a_log_ind.shape=(len(not_fix_a_log_ind),1)
        if self.visible_type=="Bernoulli" or self.visible_type=="Multinomial" or self.visible_type=="Gaussian_FixPrecision1" or self.visible_type=="Gaussian_FixPrecision2":
            if not self.if_fix_vis_bias:
                self.a=self.a - self.learn_rate_a * self.grad_a
            else:
                self.a=self.a - self.learn_rate_a * (not_fix_a_log_ind * self.grad_a)
            for nk in range(self.NK):
                self.W[nk]=self.W[nk] - self.learn_rate_W[nk] * self.grad_W[nk]
                self.b[nk]=self.b[nk] - self.learn_rate_b[nk] * self.grad_b[nk]
                
        elif self.visible_type=="Poisson":
            if not self.if_fix_vis_bias:
                self.a=self.a - self.learn_rate_a * self.grad_a
            else:
                self.a=self.a - self.learn_rate_a * (not_fix_a_log_ind * self.grad_a)
            for nk in range(self.NK):
                self.W[nk]=self.W[nk] - self.learn_rate_W[nk] * self.grad_W[nk]
                self.b[nk]=self.b[nk] - self.learn_rate_b[nk] * self.grad_b[nk]
            # set boundary for a
            self.a[self.a>tol_poisson_max]=tol_poisson_max
                
        elif self.visible_type=="NegativeBinomial":
            if not self.if_fix_vis_bias:
                self.a=self.a - self.learn_rate_a * self.grad_a
            else:
                self.a=self.a - self.learn_rate_a * (not_fix_a_log_ind * self.grad_a)
                
            for nk in range(self.NK):
                self.W[nk]=self.W[nk] - self.learn_rate_W[nk] * self.grad_W[nk]
                self.b[nk]=self.b[nk] - self.learn_rate_b[nk] * self.grad_b[nk]
                
            # a not too small, not positive,s [-100,0)
            self.a[self.a>=0]=tol_negbin_max # project a to negative
            self.a[self.a<tol_negbin_min]=tol_negbin_min
            self.W[0][self.W[0]>0]=0 # project W[0] to negative
 
        elif self.visible_type=="Multinoulli":
            for nk in range(1,self.NK):
                if nk==0:
                    # the first layer/RBM
                    for m in range(self.M):
                        if not self.if_fix_vis_bias:
                            self.a[m]=self.a[m] - self.learn_rate_a * self.grad_a[m]
                        self.W[nk][m]=self.W[nk][m] - self.learn_rate_W[nk] * self.grad_W[nk][m]
                    self.b[nk]=self.b[nk] - self.learn_rate_b * self.grad_b[nk]
                else:
                    # the second and upper layers, if any
                    self.W[nk]=self.W[nk] - self.learn_rate_W[nk] * self.grad_W[nk]
                    self.b[nk]=self.b[nk] - self.learn_rate_b[nk] * self.grad_b[nk]
                    
        elif self.visible_type=="Gaussian" or self.visible_type=="Gaussian_Hinton" or self.visible_type=="Gamma":
            if not self.if_fix_vis_bias:
                self.a[0]=self.a[0] - self.learn_rate_a[0] * self.grad_a[0]
                self.a[1]=self.a[1] - self.learn_rate_a[1] * self.grad_a[1]
            else: # fix some of the vis bias
                self.a[0]=self.a[0] - self.learn_rate_a[0] * (not_fix_a_log_ind * self.grad_a[0])
                self.a[1]=self.a[1] - self.learn_rate_a[1] * (not_fix_a_log_ind * self.grad_a[1])
            for nk in range(self.NK):
                self.W[nk]=self.W[nk] - self.learn_rate_W[nk] * self.grad_W[nk]
                self.b[nk]=self.b[nk] - self.learn_rate_b[nk] * self.grad_b[nk]


    def update_rbms(self):
        """ 
        Update the parameters of separate RBMs.
        """
        
        for nk in range(self.NK):
            if nk==0: # first RBM
                a=self.a
                b=self.b[nk]
                W=self.W[nk]
            elif nk==self.NK-1: # last RBM
                a=self.b[nk-1]
                b=self.b[nk]
                W=self.W[nk]
            else: # RBMs in the middle
                a=self.b[nk-1]
                b=self.b[nk]
                W=2*self.W[nk]
            self.rbms[nk].set_param(a,b,W)


    def get_param(self):
        return self.a,self.b,self.W


    def set_param(self, a=None, b=None, W=None, update_rbms=True):
        """
        param is a dict type.
        """
        if a is not None:
            self.a=a
        if b is not None:
            self.b=b
        if W is not None:
            self.W=W
        if update_rbms:
            self.update_rbms()     
        
        
    def make_dir_save(self,parent_dir_save, prefix, learn_rate_a, learn_rate_b, learn_rate_W, maxiter=None, normalization_method="None"):
        
        print("start making dir...")
        # different layers can have different learning rates         
        if numpy.isscalar(learn_rate_b):
            learn_rate_b=[learn_rate_b]*self.NK
        if numpy.isscalar(learn_rate_W):
            learn_rate_W=[learn_rate_W]*self.NK 
            
            
#        if self.visible_type=="Gaussian" or self.visible_type=="Gamma": 
#            foldername=prefix + "_X"+self.visible_type+":" + str(self.M) + "_H" + "_".join(numpy.array(self.hidden_type)) + ":" + "_".join(numpy.array(self.K,dtype=str)) + "_learnrateabW:" + str(learn_rate_a[0]) + "_" + str(learn_rate_a[1]) + "_" + "_".join(numpy.array(learn_rate_b,dtype=str)) + "_" + "_".join(numpy.array(learn_rate_W,dtype=str)) + "_visfix:" + str(self.visible_type_fixed_param) + "_hidfix:" + "_".join(numpy.array(self.hidden_type_fixed_param,dtype=str)) + "_iter:" + str(maxiter) + "_norm:" + normalization_method + time.strftime("_%Y%b%d%X") + "/"
#        elif self.visible_type=="Multinoulli":
#            foldername=prefix + "_X"+self.visible_type+":" + str(len(self.M)) + "_H" + "_".join(numpy.array(self.hidden_type)) + ":" + "_".join(numpy.array(self.K,dtype=str)) + "_learnrateabW:" + str(learn_rate_a) + "_" + "_".join(numpy.array(learn_rate_b,dtype=str)) + "_" + "_".join(numpy.array(learn_rate_W,dtype=str)) + "_visfix:" + str(self.visible_type_fixed_param) + "_hidfix:" + "_".join(numpy.array(self.hidden_type_fixed_param,dtype=str)) + "_iter:" + str(maxiter) + "_norm:" + normalization_method + time.strftime("_%Y%b%d%X") + "/"
#        else:
#            foldername=prefix + "_X"+self.visible_type+":" + str(self.M) + "_H" + "_".join(numpy.array(self.hidden_type)) + ":" + "_".join(numpy.array(self.K,dtype=str)) + "_learnrateabW:" + str(learn_rate_a) + "_" + "_".join(numpy.array(learn_rate_b,dtype=str)) + "_" + "_".join(numpy.array(learn_rate_W,dtype=str)) + "_visfix:" + str(self.visible_type_fixed_param[0]) + "_hidfix:" + "_".join(numpy.array(self.hidden_type_fixed_param,dtype=str)) + "_iter:" + str(maxiter) + "_norm:" + normalization_method + time.strftime("_%Y%b%d%X") + "/"
            
        if self.visible_type=="Gaussian" or self.visible_type=="Gamma": 
            foldername=prefix + "_X"+self.visible_type+":" + str(self.M) + "_H" + "_".join(numpy.array(self.hidden_type)) + ":" + "_".join(numpy.array(self.K,dtype=str)) + "_learnrateabW:" + str(learn_rate_a[0]) + "_" + str(learn_rate_a[1]) + "_" + "_".join(numpy.array(learn_rate_b,dtype=str)) + "_" + "_".join(numpy.array(learn_rate_W,dtype=str)) + "_iter:" + str(maxiter) + "_norm:" + normalization_method + time.strftime("_%Y%b%d%X") + "/"
        elif self.visible_type=="Multinoulli":
            foldername=prefix + "_X"+self.visible_type+":" + str(len(self.M)) + "_H" + "_".join(numpy.array(self.hidden_type)) + ":" + "_".join(numpy.array(self.K,dtype=str)) + "_learnrateabW:" + str(learn_rate_a) + "_" + "_".join(numpy.array(learn_rate_b,dtype=str)) + "_" + "_".join(numpy.array(learn_rate_W,dtype=str)) + "_".join(numpy.array(self.hidden_type_fixed_param,dtype=str)) + "_iter:" + str(maxiter) + "_norm:" + normalization_method + time.strftime("_%Y%b%d%X") + "/"
        else:
            foldername=prefix + "_X"+self.visible_type+":" + str(self.M) + "_H" + "_".join(numpy.array(self.hidden_type)) + ":" + "_".join(numpy.array(self.K,dtype=str)) + "_learnrateabW:" + str(learn_rate_a) + "_" + "_".join(numpy.array(learn_rate_b,dtype=str)) + "_" + "_".join(numpy.array(learn_rate_W,dtype=str)) + "_iter:" + str(maxiter) + "_norm:" + normalization_method + time.strftime("_%Y%b%d%X") + "/"
        
        dir_save=parent_dir_save+foldername
        self.dir_save=dir_save
            
        try:
            os.makedirs(dir_save)
        except OSError:
            #self.dir_save=parent_dir_save
            pass
        print("The results will be saved in " + self.dir_save)
        return self.dir_save


    def save_sampling(self, XM, ifsort=True, dir_save="./", prefix="DBM"):
        """
        Save the sampling results for bag of word data.
        """
        if ifsort:
            num_features=XM.shape[0]
            num_samples=XM.shape[1]
            XM_sorted=numpy.zeros(shape=(num_features,num_samples),dtype=float)
            features_sorted=numpy.zeros(shape=(num_features,num_samples),dtype=object)
            for n in range(num_samples):
                x=XM[:,n]
                ind=numpy.argsort(x,kind="mergesort")
                ind=ind[::-1]
                XM_sorted[:,n]=x[ind]
                features_sorted[:,n]=self.features[ind]
                
            filename=dir_save + prefix + "_sampled_XM_sorted.txt"
            numpy.savetxt(filename,XM_sorted, fmt="%.2f", delimiter="\t")
            filename=dir_save + prefix + "_sampled_features_sorted.txt"
            numpy.savetxt(filename,features_sorted, fmt="%s", delimiter="\t")
        else:
            filename=dir_save + prefix + "_sampled_XM.txt"
            numpy.savetxt(filename,XM, fmt="%.2f", delimiter="\t")
            filename=dir_save + prefix + "_features.txt"
            numpy.savetxt(filename,self.features, fmt="%s", delimiter="\t")
