
#from __future__ import division
import numpy
import math
import restricted_boltzmann_machine
import classification as cl
import copy
import os
import time

class helmholtz_machine:
    def __init__(self, features=None, M=None, K=None, visible_type="Bernoulli", visible_type_fixed_param=1, hidden_type="Bernoulli", hidden_type_fixed_param=0, if_fix_vis_bias=False, a=None, fix_a_log_ind=None, tol_poisson_max=8, rng=numpy.random.RandomState(100)):
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
        self.a=[] # generative
        self.b=[] # generative
        self.W=[] # generative
        self.br=[] # recognition
        self.Wr=[] # recognition
        self.rbms=[] 
        self.rng=rng
        
        self.tol_poisson_max=tol_poisson_max
        
        if self.visible_type=="Bernoulli":
            #self.a=self.rng.normal(loc=0, scale=0.01, size=(self.M,1)) # M X 1
            #self.a=self.rng.uniform(low=-0.01, high=0.01, size=(self.M,1)) # M X 1
            self.a=numpy.zeros(shape=(self.M,1))
            for nk in range(self.NK):
                if nk==0:
                    nrow_W_nk=self.M
                else:
                    nrow_W_nk=self.K[nk-1]
                ncol_W_nk=self.K[nk]
                W_nk=self.rng.normal(loc=0, scale=0.0001, size=(nrow_W_nk,ncol_W_nk))
                self.W.append( W_nk ) # M by K[n], initialize weight matrices
                b_nk=numpy.zeros(shape=(ncol_W_nk,1))
                self.b.append( b_nk )
                #self.W.append( numpy.zeros(shape=(nrow_W_nk,ncol_W_nk), dtype=float) )
                #self.b.append( self.rng.normal(loc=0, scale=0.01, size=(ncol_W_nk,1)) ) # K[n] X 1
                
                Wr_nk=numpy.copy(W_nk.transpose())
                self.Wr.append(Wr_nk)
                br_nk=numpy.copy(b_nk)
                self.br.append( br_nk )
                
        elif self.visible_type=="Gaussian":
            self.a=[None]*2
            self.a[0]=self.rng.normal(loc=0, scale=0.001, size=(self.M,1)) # M X 1
            self.a[1]=-5*numpy.ones(shape=(self.M,1),dtype=float)  # M X 1, -precision/2, beta>0f.M,1)) # M X 1
            for nk in range(self.NK):
                if nk==0:
                        nrow_W_nk=self.M
                else:
                    nrow_W_nk=self.K[nk-1]
                ncol_W_nk=self.K[nk]
                W_nk=self.rng.normal(loc=0, scale=0.001, size=(nrow_W_nk,ncol_W_nk))
                self.W.append( W_nk ) # M by K[n], initialize weight matrices
                b_nk=numpy.zeros(shape=(ncol_W_nk,1))
                self.b.append( b_nk ) # K[n] X 1
                
                Wr_nk=numpy.copy(W_nk.transpose())
                self.Wr.append(Wr_nk)
                br_nk=numpy.copy(b_nk)
                self.br.append( br_nk )
                
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
                W_nk=self.rng.normal(loc=0, scale=0.001, size=(nrow_W_nk,ncol_W_nk))
                self.W.append( W_nk ) # M by K[n], initialize weight matrices
                b_nk=numpy.zeros(shape=(ncol_W_nk,1))
                self.b.append( b_nk ) # K[n] X 1
                
                Wr_nk=numpy.copy(W_nk.transpose())
                self.Wr.append(Wr_nk)
                br_nk=numpy.copy(b_nk)
                self.br.append( br_nk )
                
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
                W_nk=numpy.zeros(shape=(nrow_W_nk,ncol_W_nk), dtype=float)
                self.W.append( W_nk ) # M by K[n], initialize weight matrices
                #self.b.append( self.rng.normal(loc=0, scale=0.01, size=(ncol_W_nk,1)) ) # K[n] X 1
                b_nk=numpy.zeros(shape=(ncol_W_nk,1))
                self.b.append( b_nk )
                
                Wr_nk=numpy.copy(W_nk.transpose())
                self.Wr.append(Wr_nk)
                br_nk=numpy.copy(b_nk)
                self.br.append( br_nk )
                
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
                W_nk=numpy.zeros(shape=(nrow_W_nk,ncol_W_nk), dtype=float)
                self.W.append( W_nk ) # M by K[n], initialize weight matrices
                #self.b.append( self.rng.normal(loc=0, scale=0.01, size=(ncol_W_nk,1)) ) # K[n] X 1
                b_nk=numpy.zeros(shape=(ncol_W_nk,1))
                self.b.append( b_nk )
                
                Wr_nk=numpy.copy(W_nk.transpose())
                self.Wr.append(Wr_nk)
                br_nk=numpy.copy(b_nk)
                self.br.append( br_nk )
                
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
                W_nk=numpy.zeros(shape=(nrow_W_nk,ncol_W_nk), dtype=float)
                self.W.append( W_nk )
                #self.b.append( self.rng.normal(loc=0, scale=0.01, size=(ncol_W_nk,1)) ) # K[n] X 1
                b_nk=numpy.zeros(shape=(ncol_W_nk,1))
                self.b.append( b_nk )
                
                Wr_nk=numpy.copy(W_nk.transpose())
                self.Wr.append(Wr_nk)
                br_nk=numpy.copy(b_nk)
                self.br.append( br_nk )
                
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
                    W_nk=numpy.abs(self.rng.normal(loc=0, scale=0.001, size=(nrow_W_nk,ncol_W_nk) ) )
                else:
                    W_nk=numpy.zeros(shape=(nrow_W_nk,ncol_W_nk), dtype=float)
                self.W.append( W_nk )
                b_nk=numpy.zeros(shape=(ncol_W_nk,1))
                self.b.append( b_nk )
                
                Wr_nk=numpy.copy(W_nk.transpose())
                self.Wr.append(Wr_nk)
                br_nk=numpy.copy(b_nk)
                self.br.append( br_nk )
            
            if self.visible_type_fixed_param is None:
                self.visible_type_fixed_param=100*numpy.ones(shape=(self.M,1),dtype=float)
            if numpy.isscalar(self.visible_type_fixed_param):
                self.visible_type_fixed_param=self.visible_type_fixed_param*numpy.ones(shape=(self.M,1),dtype=float)
                
        elif self.visible_type=="Multinomial":
            self.a=numpy.zeros(shape=(self.M,1),dtype=float)
            for nk in range(self.NK):
                if nk==0:
                    nrow_W_nk=self.M
                else:
                    nrow_W_nk=self.K[nk-1]
                ncol_W_nk=self.K[nk]
                W_nk=numpy.zeros(shape=(nrow_W_nk,ncol_W_nk), dtype=float)
                self.W.append( W_nk )
                #self.b.append( self.rng.normal(loc=0, scale=0.01, size=(ncol_W_nk,1)) ) # K[n] X 1
                b_nk=numpy.zeros(shape=(ncol_W_nk,1))
                self.b.append( b_nk )
                
                Wr_nk=numpy.copy(W_nk.transpose())
                self.Wr.append(Wr_nk)
                br_nk=numpy.copy(b_nk)
                self.br.append( br_nk )
                
        elif self.visible_type=="Multinoulli":
            self.Ms=visible_type_fixed_param
            self.a=math.log(1/self.M)*numpy.ones(shape=(self.M,1))
            self.W=[]
            self.b=[]
            for nk in range(self.NK):
                if nk==0:
                    W_input=[None]*self.M
                    Wr_input=[None]*self.M
                    ncol_W_nk=self.K[nk]
                    for m in range(self.M):
                        W_input[m]=self.rng.normal(loc=0, scale=0.001, size=(self.Ms[m],ncol_W_nk))
                        Wr_input[m]=numpy.copy(W_input[m])
                    self.W.append(W_input)
                    self.W.append(Wr_input)
                else:
                    nrow_W_nk=self.K[nk-1]
                    ncol_W_nk=self.K[nk]
                    W_nk=numpy.zeros(shape=(nrow_W_nk,ncol_W_nk), dtype=float)
                    self.W.append( W_nk )
                
                    Wr_nk=numpy.copy(W_nk.transpose())
                    self.Wr.append(Wr_nk)
                    
                b_nk=numpy.zeros(shape=(ncol_W_nk,1))
                self.b.append( b_nk )
                br_nk=numpy.copy(b_nk)
                self.br.append( br_nk )

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
                W_nk=numpy.zeros(shape=(nrow_W_nk,ncol_W_nk), dtype=float)
                self.W.append( W_nk )
                #self.b.append( self.rng.normal(loc=0, scale=0.01, size=(ncol_W_nk,1)) ) # K[n] X 1
                b_nk=numpy.zeros(shape=(ncol_W_nk,1))
                self.b.append( b_nk )
                
                Wr_nk=numpy.copy(W_nk.transpose())
                self.Wr.append(Wr_nk)
                br_nk=numpy.copy(b_nk)
                self.br.append( br_nk )
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
        Pretraining HM using RBMs.
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
        self.batch_size=batch_size
        visible_type=self.visible_type
        #self.rbms=[] # define it in initialization
        self.H_pretrain=[]
        print("Start pretraining DBM...")
        for nk in range(self.NK):
            print("the {0}-th hidden layer...".format(nk+1))
            if nk==0: # bottom RBM
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
                #a=self.b[nk-1] # a is already updated below
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
            a_nk,b_nk,W_nk=rbm_model.get_param()
            if self.visible_type=="Multinoulli" and nk==0:
                Wr_nk=[copy.deepcopy(w.transpose()) for w in W_nk]
            else:
                Wr_nk=numpy.copy(W_nk.transpose()) # or copy.deepcopy(W_nk.transpose())
            if nk==0: # bottom RBM                    
                self.a=a_nk
                self.W[nk]=W_nk
                self.b[nk]=b_nk
                self.Wr[nk]=Wr_nk
                self.br[nk]=copy.deepcopy(b_nk)
            else: # middle or top RBMs
                self.W[nk]=W_nk
                self.b[nk]=b_nk
                self.Wr[nk]=Wr_nk
                self.br[nk]=copy.deepcopy(b_nk)

            #rbm_X,_=rbm_model.sample_h_given_x(rbm_X) # the output of this layer is used as input of the next layer
            _,rbm_X=rbm_model.sample_h_given_x(rbm_X) # Hinton suggested to use probabilities
            a=b_nk # the bias of the nk-th hidden layer is used as the bias of visible notes of the nk+1-th layer 

            # save the trained rbms for initialize mean-filed approximation and Gibbs sampling.
            self.rbms.append(rbm_model)
            self.H_pretrain.append(rbm_X) # H of each RBM, for the purpose of (1) initializing mean-field approximation inference, (2) Gibbs sampling, and (3) building multi-modal DBM.

        print("Finished pretraining of HM!")
        end_time = time.clock()
        self.pretrain_time=end_time-start_time
        return self.pretrain_time
        print("It took {0} seconds.".format(self.pretrain_time))


    def train(self,X=None, X_validate=None, batch_size=10, maxiter=100, learn_rate_a=0.01, learn_rate_b=0.01, learn_rate_W=0.01, change_rate=0.8, adjust_change_rate_at=None, adjust_coef=1.02, change_every_many_iters=10, init_chain_time=100, train_subset_size_for_compute_error=100, valid_subset_size_for_compute_error=100, track_reconstruct_error=True, track_free_energy=True, if_plot_error_free_energy=False, dir_save="./", prefix="HM", figwidth=5, figheight=3):
        """
        Wake-sleep algorithm to train HM.
        Different layers have different learning rate.
        """
        start_time=time.clock()
        print("Start training HM...")
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

        for i in range(self.maxiter):
            
            if adjust_change_rate_at is not None:
                if i==adjust_change_rate_at[0]:
                    change_rate=change_rate*adjust_coef # increast change_rate
                    change_rate=1.0 if change_rate>1.0 else change_rate # make sure not greater than 1
                    if len(adjust_change_rate_at)>1:
                        adjust_change_rate_at=adjust_change_rate_at[1:] # delete the first element
                    else:
                        adjust_change_rate_at=None
                        
            # change learning rates
            self.learn_rate_a=self.change_learning_rate(current_learn_rate=self.learn_rate_a, change_rate=change_rate, current_iter=i, change_every_many_iters=change_every_many_iters)
            self.learn_rate_b=self.change_learning_rate(current_learn_rate=self.learn_rate_b, change_rate=change_rate, current_iter=i, change_every_many_iters=change_every_many_iters)
            self.learn_rate_W=self.change_learning_rate(current_learn_rate=self.learn_rate_W, change_rate=change_rate, current_iter=i, change_every_many_iters=change_every_many_iters)
            #print "starting the {0}-th iteration, the learning rate of a, b, W: {1}, {2}, {3}".format(i,self.learn_rate_a,self.learn_rate_b,self.learn_rate_W)
            # get mini-batch
            
            ## wake phase
            Xbatch=self.sample_minibatch(self.batch_size)
            XbatchMg,HbatchMg,Hbatchr,HbatchMr,a_hat_gen,b_hat_gen=self.sample_xh_wake(Xbatch)            
            self.compute_gradient_wake(Xbatch, XbatchMg, Hbatchr, HbatchMg)
            self.update_param_wake()
            
            # update the parameters for RBMs
            self.update_rbms()            
            
            ## sleep phase Xg,XMg,Hg,HMr
            #Xfantacy,_,Hfantacy,HfantacyMr=self.sample_xh_sleep(self.batch_size) # this does not generate good fantacies
            Xfantacy,_,Hfantacy,HfantacyMr=self.sample_xh_sleep(self.batch_size, Hg=Hbatchr) # I want to try this
            
            self.compute_gradient_sleep(Xfantacy,Hfantacy,HfantacyMr)
            self.update_param_sleep()
            
            # compute reconstruction error of the training samples
            # sample some training samples, rather than use all training samples which is time-consuming
            if track_reconstruct_error:
                rec_error_train,_,_,_=self.compute_reconstruction_error(X0=Xbatch, X0RM=XbatchMg )
             
             
            #we can monitor the lower bound for each iteration    
                
            if track_free_energy:
                mfe_train,_=self.compute_free_energy(X=Xbatch, HMr=HbatchMr, a_hat_gen=a_hat_gen, b_hat_gen=b_hat_gen)
            self.rec_errors_train.append(rec_error_train)
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
                        rec_error_valid,HMr_valid,a_hat_gen_valid,b_hat_gen_valid=self.compute_reconstruction_error(X0=X_validate_subset, X0RM=None)
                    if track_free_energy:
                        mfe_validate,_=self.compute_free_energy(X=X_validate_subset, HMr=HMr_valid, a_hat_gen=a_hat_gen_valid, b_hat_gen=b_hat_gen_valid)
                else:
                    if track_reconstruct_error:                  
                        rec_error_valid,HMr_valid,a_hat_gen_valid,b_hat_gen_valid=self.compute_reconstruction_error(X0=self.X_validate, X0RM=None)
                    if track_free_energy:
                        mfe_validate,_=self.compute_free_energy(X=self.X_validate, HMr=HMr_valid, a_hat_gen=a_hat_gen_valid, b_hat_gen=b_hat_gen_valid)
                self.rec_errors_valid.append(rec_error_valid)
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

        print("The (fine-tuning) training of HM is finished!")
        end_time = time.clock()
        self.train_time=end_time-start_time
        return self.train_time
        print("It took {0} seconds.".format(self.train_time))
    
    
    def sample_xh_wake(self, X, compute_HMg=True):
        """
        Use the recognition parameters to sample hidden states.
        """
        # sample h
        Hr=[None]*self.NK # sampled by recognition parameters
        HMr=[None]*self.NK # the mean of H computed by recognition parameters, to compute the free energy
        for nk in range(self.NK):
            if nk==0:
                if self.visible_type=="Multinoulli":
                    b_hat_nk=self.br[nk]
                    for m in range(self.M):
                        b_hat_nk = b_hat_nk + numpy.dot( self.Wr[m][nk], X[m] )
                else:
                    b_hat_nk=self.br[nk] + numpy.dot( self.Wr[nk], X )
            else:
                b_hat_nk=self.br[nk] + numpy.dot( self.Wr[nk], Hr[nk-1] )
                
            Hr[nk],HMr[nk]=self.sample_h_given_b_hat(b_hat=b_hat_nk, hidden_type=self.hidden_type[nk], hidden_type_fixed_param=self.hidden_type_fixed_param[nk], hidden_value_or_meanfield="value")
            
        # compute mean of x,h using generative parameters
        HMg=[None]*self.NK # sampled by generative parameters
        if compute_HMg:
            a_hat_gen,b_hat_gen=self.compute_posterior_bias_use_generative_param(Hr, a_hat_only=False)
            for nk in range(self.NK):
                _,HMg[nk]=self.sample_h_given_b_hat(b_hat=b_hat_gen[nk], hidden_type=self.hidden_type[nk], hidden_type_fixed_param=self.hidden_type_fixed_param[nk], hidden_value_or_meanfield="value")
        else:
            a_hat_gen,_=self.compute_posterior_bias_use_generative_param(Hr, a_hat_only=True)
            b_hat_gen = None
        _,XMg,_=self.sample_visible(visible_type=self.visible_type, a=a_hat_gen, W=None, H=None, visible_type_fixed_param=self.visible_type_fixed_param, rng=self.rng)
        
        return XMg,HMg,Hr,HMr,a_hat_gen,b_hat_gen


    def compute_posterior_bias_use_recognition_param(self, X, H):
        b_hat=[None]*self.NK
        for nk in range(self.NK):
            if nk==0:
                if self.visible_type=="Multinoulli":
                    b_hat0=self.br[nk]
                    for m in range(self.M):
                        b_hat0 = b_hat0 + numpy.dot( self.Wr[nk][m], X[m] )
                    b_hat[nk]=b_hat0
                else:
                    b_hat[nk]=self.br[nk] + numpy.dot( self.Wr[nk], X )
            else:
                b_hat[nk]=self.br[nk] + numpy.dot( self.Wr[nk], H[nk-1] )
                
        return b_hat

        
    def compute_posterior_bias_use_generative_param(self, H, a_hat_only=False):
        
        # a_hat
        if self.visible_type=="Multinoulli":
            a_hat=[None]*self.M
            for m in range(self.M):
                a_hat[m]= self.a[m] + numpy.dot(self.W[0][m],H[0])
        elif self.visible_type=="Gaussian":
            a_hat=[None]*2
            a1=self.a[0]
            a2=self.a[1]
            a_hat[0]=a1 + numpy.dot(self.W[0],H[0]) 
            a_hat[1]=a2
        elif self.visible_type=="Gaussian_Hinton":
            a1=self.a[0]
            a2=self.a[1]
            a_hat[0]=a1 + 1/a2*numpy.dot(self.W[0],H[0]) 
            a_hat[1]=a2
        else:
            a_hat=self.a + numpy.dot(self.W[0],H[0])

        # b_hat
        b_hat=[None]*self.NK
        if a_hat_only:
            return a_hat,b_hat
            
        for nk in range(self.NK):
            if nk==self.NK-1:
                b_hat[nk]=self.b[nk]
            else:
                b_hat[nk]=self.b[nk] + numpy.dot( self.W[nk+1], H[nk+1] )
        return a_hat,b_hat
        
        
    def compute_gradient_wake(self,Xbatch, XbatchMg, Hbatchr, HbatchMg):
        """
        Compute gradient in the wake phase to update the generative parameters.
        """
        if self.visible_type=="Bernoulli" or self.visible_type=="Poisson" or self.visible_type=="NegativeBinomial" or self.visible_type=="Multinomial" or self.visible_type=="Gaussian_FixPrecision2":
            grad_a=-numpy.mean(Xbatch-XbatchMg,axis=1)
            grad_a.shape=(self.M,1)
        elif self.visible_type=="Gaussian":
            grad_a1=-numpy.mean(Xbatch-XbatchMg,axis=1)
            XbatchMg2=XbatchMg**2 - 1/(2*self.a[2])
            grad_a2=-numpy.mean(Xbatch**2-XbatchMg2,axis=1)
            grad_a1.shape=(self.M,1)
            grad_a2.shape=(self.M,1)
            grad_a=[grad_a1,grad_a2]
        elif self.visible_type=="Gaussian_FixPrecision1":
            grad_a=-numpy.mean(self.visible_type_fixed_param*(Xbatch-XbatchMg),axis=1)
            grad_a.shape=(self.M,1)
        elif self.visible_type=="Multinoulli":
            grad_a=[None]*self.M
            for m in range(self.M):
                grad_am=-numpy.mean(Xbatch[m]-XbatchMg[m])
                grad_am.shape=(self.Ms[m],1)
                grad_a[m]=grad_am
        
        grad_b=[None]*self.NK
        for nk in range(self.NK):
            grad_bnk=-numpy.mean(Hbatchr[nk] - HbatchMg[nk], axis=1)
            grad_bnk.shape=(self.K[nk],1)
            grad_b[nk]=grad_bnk
            
        grad_W=[None]*self.NK
        for nk in range(self.NK):
            if nk==0:
                if self.visible_type=="Multinoulli":
                    grad_W0=[None]*self.M
                    for m in range(self.M):
                        grad_W0[m]=-numpy.dot(Xbatch[m]-XbatchMg[m],Hbatchr[nk].transpose())/self.batch_size
                    grad_W[nk]=grad_W0
                elif self.visible_type=="Gaussian_FixPrecision1":
                    grad_W[nk]=-numpy.dot(self.visible_type_fixed_param*(Xbatch-XbatchMg),Hbatchr[nk].transpose())/self.batch_size
                else: # not Multinoulli, Gaussian_FixPrecision1 distributions for visible types
                    grad_W[nk]=-numpy.dot(Xbatch-XbatchMg,Hbatchr[nk].transpose())/self.batch_size
            else: # not first hidden layer
                grad_W[nk]=-numpy.dot(Hbatchr[nk-1]-HbatchMg[nk-1],Hbatchr[nk].transpose())/self.batch_size
                
            self.grad_a=grad_a
            self.grad_b=grad_b
            self.grad_W=grad_W
    
    
    def update_param_wake(self):
        """
        Update the generative parameters.
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


    def sample_xh_sleep(self, NS=100, Hg=None, value_or_mean="value", compute_HMr=True):
        """
        Use the generative parameters to sample hidden states and visible states.
        Hg: None or a list of length of NK. If Hg is a list, Hg[-1] is a matrix, the rest are None's. This is used in MDBN.
        """
        if Hg is None:
            Hg=[None]*self.NK
            last=self.NK-1
        else:
            last=self.NK-2 # in this case, NS is not used.
            
        for nk in range(last,-1,-1):
            if nk==self.NK-1:
                b_hat_nk= numpy.repeat( self.b[nk], NS, axis=1 )
            else:
                b_hat_nk=self.b[nk] + numpy.dot(self.W[nk+1],Hg[nk+1])
                
            Hg_nk,HMg_nk=self.sample_h_given_b_hat(b_hat=b_hat_nk, hidden_type=self.hidden_type[nk], hidden_type_fixed_param=self.hidden_type_fixed_param[nk], hidden_value_or_meanfield="value")
            if value_or_mean=="value":
                Hg[nk]=Hg_nk
            if value_or_mean=="mean":
                Hg[nk]=HMg_nk
        a_hat,_=self.compute_posterior_bias_use_generative_param(Hg, a_hat_only=True)
        Xg,XMg,_=self.sample_visible(visible_type=self.visible_type, a=a_hat, W=None, H=None, visible_type_fixed_param=self.visible_type_fixed_param, rng=self.rng)
        if value_or_mean=="mean":
            Xg=XMg
            
        # compute expected h using recognition parameters
        HMr=[None]*self.NK
        if compute_HMr:
            b_hat=self.compute_posterior_bias_use_recognition_param(Xg, Hg)
            for nk in range(self.NK):
                _,HMr[nk]=self.sample_h_given_b_hat(b_hat=b_hat[nk], hidden_type=self.hidden_type[nk], hidden_type_fixed_param=self.hidden_type_fixed_param[nk], hidden_value_or_meanfield="value")
            
        return Xg,XMg,Hg,HMr


    def generate_x(self, NS=100, num_iter=1000, init=True):
        """
        Generate samples from the learned exp-HM.
        """
        if init:
            self.Xg,self.XMg,self.Hg,_=self.sample_xh_sleep(NS, compute_HMr=False)
            
        for i in range(num_iter):
            _,_,self.Hr,self.HMr,_,_=self.sample_xh_wake(self.Xg, compute_HMg=False)
            self.Xg,self.XMg,self.Hg,_=self.sample_xh_sleep(Hg=self.Hr, compute_HMr=False)
        return self.Xg,self.XMg
        

    def compute_gradient_sleep(self, Xfantacy, Hfantacy, HfantacyMr):
        """
        Compute gradient in the sleep phase to update the recognition parameters.
        """
        grad_br=[None]*self.NK
        for nk in range(self.NK):
            grad_brnk=-numpy.mean(Hfantacy[nk] - HfantacyMr[nk], axis=1)
            grad_brnk.shape=(self.K[nk],1)
            grad_br[nk]=grad_brnk
            
        grad_Wr=[None]*self.NK
        for nk in range(self.NK):
            if nk==0:
                if self.visible_type=="Multinoulli":
                    grad_Wr0=[None]*self.M
                    for m in range(self.M):
                        grad_Wr0[m]=-numpy.dot(Hfantacy[nk]-HfantacyMr[nk],Xfantacy[m].transpose())/self.batch_size
                    grad_Wr[nk]=grad_Wr0
                elif self.visible_type=="Gaussian_FixPrecision1":
                    grad_Wr[nk]=-numpy.dot(Hfantacy[nk] - HfantacyMr[nk], self.visible_type_fixed_param*Xfantacy.transpose())/self.batch_size
                else: # not Multinoulli, Gaussian_FixPrecision1 distributions for visible types
                    grad_Wr[nk]=-numpy.dot(Hfantacy[nk]-HfantacyMr[nk],Xfantacy.transpose())/self.batch_size
            else: # not first hidden layer
                grad_Wr[nk]=-numpy.dot(Hfantacy[nk]-HfantacyMr[nk],Hfantacy[nk-1].transpose())/self.batch_size
            
        self.grad_br=grad_br
        self.grad_Wr=grad_Wr
    
    
    def update_param_sleep(self):
        """
        Update the recognition parameters.
        """
#        if self.visible_type=="Bernoulli" or self.visible_type=="Poisson" or self.visible_type=="NegativeBinomial" or self.visible_type=="Multinomial" or or self.visible_type=="Gaussian" or self.visible_type=="Gaussian_FixPrecision1" or self.visible_type=="Gaussian_FixPrecision2" or self.visible_type=="Gaussian_Hinton":
#            for nk in range(self.NK):
#                self.Wr[nk]=self.Wr[nk] - self.learn_rate_W[nk] * self.grad_Wr[nk]
#                self.br[nk]=self.br[nk] - self.learn_rate_b[nk] * self.grad_br[nk]
# 
#        elif self.visible_type=="Multinoulli":
#            for nk in range(1,self.NK):
#                if nk==0:
#                    # the first layer/RBM
#                    for m in range(self.M):
#                        self.Wr[nk][m]=self.Wr[nk][m] - self.learn_rate_W[nk] * self.grad_Wr[nk][m]
#                    self.br[nk]=self.br[nk] - self.learn_rate_b * self.grad_br[nk]
#                else:
#                    # the second and upper layers, if any
#                    self.Wr[nk]=self.Wr[nk] - self.learn_rate_W[nk] * self.grad_Wr[nk]
#                    self.br[nk]=self.br[nk] - self.learn_rate_b[nk] * self.grad_br[nk]

        if self.visible_type=="Multinoulli":
            for nk in range(1,self.NK):
                if nk==0:
                    # the first layer/RBM
                    for m in range(self.M):
                        self.Wr[nk][m]=self.Wr[nk][m] - self.learn_rate_W[nk] * self.grad_Wr[nk][m]
                    self.br[nk]=self.br[nk] - self.learn_rate_b * self.grad_br[nk]
                else:
                    # the second and upper layers, if any
                    self.Wr[nk]=self.Wr[nk] - self.learn_rate_W[nk] * self.grad_Wr[nk]
                    self.br[nk]=self.br[nk] - self.learn_rate_b[nk] * self.grad_br[nk]
        else:
            for nk in range(self.NK):
                self.Wr[nk]=self.Wr[nk] - self.learn_rate_W[nk] * self.grad_Wr[nk]
                self.br[nk]=self.br[nk] - self.learn_rate_b[nk] * self.grad_br[nk]


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


#    def plot_error_free_energy(self, dir_save="./", prefix="HM", mean_over=5, figwidth=5, figheight=3):
#        import matplotlib as mpl
#        mpl.use("pdf")
#        import matplotlib.pyplot as plt
#        
#        if len(self.rec_errors_train)>0:
#            num_iters=len(self.rec_errors_train)
#            if mean_over>0:
#                self.rec_errors_train=self.smooth(self.rec_errors_train, mean_over=mean_over)
#            else:
#                self.rec_errors_train=numpy.array(self.rec_errors_train)
#
#        if len(self.rec_errors_valid)>0:
#            num_iters=len(self.rec_errors_valid)
#            if mean_over>0:
#                self.rec_errors_valid=self.smooth(self.rec_errors_valid, mean_over=mean_over)
#            else:
#                self.rec_errors_valid=numpy.array(self.rec_errors_valid)
#
#        if len(self.mfes_train)>0:
#            num_iters=len(self.mfes_train)
#            if mean_over>0:
#                self.mfes_train=self.smooth(self.mfes_train, mean_over=mean_over)
#            else:
#                self.mfes_train=numpy.array(self.mfes_train)
#
#        if len(self.mfes_valid)>0:
#            num_iters=len(self.mfes_valid)
#            if mean_over>0:
#                self.mfes_valid=self.smooth(self.mfes_valid, mean_over=mean_over)
#            else:
#                self.mfes_valid=numpy.array(self.mfes_valid)
#
#        iters=numpy.array(range(num_iters),dtype=int)
#        
#        # ignore the first five results as they are not stable
#        iters=iters[5:]
#        if len(self.rec_errors_train)>0:
#            self.rec_errors_train=self.rec_errors_train[5:]
#        if len(self.rec_errors_valid)>0:
#            self.rec_errors_valid=self.rec_errors_valid[5:]
#        if len(self.mfes_train)>0:
#            self.mfes_train=self.mfes_train[5:]
#        if len(self.mfes_valid)>0:
#            self.mfes_valid=self.mfes_valid[5:]
#        
#        #plt.ion()
#        fig=plt.figure(num=1,figsize=(figwidth,figheight))
#        ax=fig.add_subplot(1,1,1)
#        if len(self.mfes_train)>0:
#            ax.plot(iters,self.mfes_train,linestyle="-", color="blue", linewidth=0.5, label="FE:Train")
#        if len(self.mfes_valid)>0:
#            ax.plot(iters,self.mfes_valid,linestyle=":",color="blueviolet",linewidth=0.5, label="FE:Test")
#        ax.set_xlabel("Iteration",fontsize=8)
#        ax.set_ylabel("Free Energy (FE)",color="blue",fontsize=8)
#        for tl in ax.get_yticklabels():
#            tl.set_color("b")
#        plt.setp(ax.get_yticklabels(), fontsize=8)
#        plt.setp(ax.get_xticklabels(), fontsize=8)
#            
#        #ax.legend(loc="lower left",fontsize=8)
#
#        ax2=ax.twinx()
#        if len(self.rec_errors_train)>0:
#            ax2.plot(iters,self.rec_errors_train,linestyle="-",color="red",linewidth=0.5, label="RCE:Train")
#        if len(self.rec_errors_valid)>0:
#            ax2.plot(iters,self.rec_errors_valid,linestyle=":",color="darkgoldenrod",linewidth=0.5, label="RCE:Test")
#        ax2.set_ylabel("Reconstruction Error (RCE)", color="red",fontsize=8)
#        for tl in ax2.get_yticklabels():
#            tl.set_color("r")
#        plt.setp(ax2.get_yticklabels(), fontsize=8)
#        plt.setp(ax2.get_xticklabels(), fontsize=8)
#        # legend
#        ax.legend(loc="lower left",fontsize=8)
#        ax2.legend(loc="upper right",fontsize=8)
#        filename=dir_save+prefix+"_error_free_energy.pdf"
#        plt.tight_layout()
#        fig.savefig(filename,bbox_inches='tight',format="pdf",dpi=600)
#        plt.close(fig)
#        #plt.close("all")
        

    def plot_error_free_energy(self, dir_save="./", prefix="RBM", mean_over=5, figwidth=5, figheight=3):
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
        ax1=fig.add_subplot(1,1,1)
        if len(self.rec_errors_train)>0:
            ax1.plot(iters,self.rec_errors_train,linestyle="-",color="red",linewidth=0.5, label="RCE:Train")
        if len(self.rec_errors_valid)>0:
            ax1.plot(iters,self.rec_errors_valid,linestyle=":",color="darkgoldenrod",linewidth=0.5, label="RCE:Test")
        ax1.set_xlabel("Iteration",fontsize=8)
        ax1.set_ylabel("Reconstruction Error (RCE)", color="red",fontsize=8)
        for tl in ax1.get_yticklabels():
            tl.set_color("r")
        plt.setp(ax1.get_yticklabels(), fontsize=8)
        plt.setp(ax1.get_xticklabels(), fontsize=8)
        ax1.legend(loc="upper right",fontsize=8)
        prefix=prefix+"_error"
            
        #ax.legend(loc="lower left",fontsize=8)
        if len(self.mfes_train)>0 or len(self.mfes_valid)>0:
            ax2=ax1.twinx()
            if len(self.mfes_train)>0:
                ax2.plot(iters,self.mfes_train,linestyle="-", color="blue", linewidth=0.5, label="FE:Train")
            if len(self.mfes_valid)>0:
                ax2.plot(iters,self.mfes_valid,linestyle=":",color="blueviolet",linewidth=0.5, label="FE:Test")
            ax2.set_xlabel("Iteration",fontsize=8)
            ax2.set_ylabel("Free Energy (FE)",color="blue",fontsize=8)
            for tl in ax2.get_yticklabels():
                tl.set_color("b")
            plt.setp(ax2.get_yticklabels(), fontsize=8)
            plt.setp(ax2.get_xticklabels(), fontsize=8)
            ax1.legend(loc="upper left",fontsize=8)
            ax2.legend(loc="upper right",fontsize=8)
            prefix=prefix+"_free_energy"
        
        filename=dir_save+prefix+".pdf"
        plt.tight_layout()
        fig.savefig(filename,bbox_inches='tight',format="pdf",dpi=600)
        plt.close(fig)
        #plt.close("all")
        
        
    def compute_reconstruction_error(self, X0, X0RM=None):
        """
        Compute the difference between the real sample X0 and the recoverd sample X0RM by mean-field.
        """
        if X0RM is None:
            X0RM,_,_,HMr,a_hat_gen,b_hat_gen=self.sample_xh_wake(X0, compute_HMg=False) # HMr,a_hat_gen,b_hat_gen may be used to compute the free energy as well
        else:
            HMr=None
            a_hat_gen=None
            b_hat_gen=None
        if self.visible_type=="Multinoulli":
            self.rec_error=0
            for m in range(self.M):
                self.rec_error= self.rec_error + self.rec_error+numpy.mean(numpy.abs(X0RM[m]-X0[m]))
        else:
            self.rec_error=numpy.mean(numpy.abs(X0RM-X0))
        return self.rec_error,HMr,a_hat_gen,b_hat_gen


    def compute_free_energy(self,X=None, HMr=None, a_hat_gen=None, b_hat_gen=None, in_mdbn=False): 
        """
        Compute "free" energy - E_q[log p(x,h)] - H(q). 
        """
        if X is None:
            X=self.X
        if HMr is None:
            _,_,_,HMr,a_hat_gen,b_hat_gen=self.sample_xh_wake(X,compute_HM=False)
            
        # compute E_q[log p(x,h)]
        mean_logpxy,_=self.compute_Eq_log_pxh(X, HMr, a_hat_gen, b_hat_gen, in_mdbn=in_mdbn)
        
        # compute entropy
        mean_entropy,_=self.compute_entropy(HMr, in_mdbn=in_mdbn)
        
        fes= - mean_logpxy - mean_entropy
        
        mfe=numpy.mean(fes) # average over N samples
        return mfe,fes
        

    def compute_Eq_log_pxh(self, X, HMr, a_hat_gen, b_hat_gen, in_mdbn=False):
        """
        Compute E [ log p(z) ] = E[ log h(z) + theta^T s(z) - A(theta) ] = E_q [ zeta(z,theta) - A(theta) ] where is theta is posterior generative parameter. 
        """
        
        rbm=restricted_boltzmann_machine.restricted_boltzmann_machine(M=100, K=100) # create a rbm for call its zeta function, the initial parameters of this RBM does not matter
        z=rbm.zeta(a_hat_gen, X, fixed_param=self.visible_type_fixed_param, distribution=self.visible_type)
        logPar=rbm.A(a_hat_gen, fixed_param=self.visible_type_fixed_param, distribution=self.visible_type)

        if in_mdbn:
            last=self.NK-1 # do not consider the last hidden layer, as it will be considered in the joint MDBM
        else:
            last=self.NK
            
        for nk in range(last):
            z = z + rbm.zeta(b_hat_gen[nk], HMr[nk], fixed_param=self.hidden_type_fixed_param[nk], distribution=self.hidden_type[nk])
            logPar = logPar + rbm.A(b_hat_gen[nk], fixed_param=self.hidden_type_fixed_param[nk], distribution=self.hidden_type[nk])
    
        logpxh = z - logPar # for all samples
        
        mean_logpxh=numpy.mean(logpxh)
        
        return mean_logpxh,logpxh
        
        
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
        

    def compute_entropy(self, HP, in_mdbn=False):
        """
        Compute the entropy of approximate distribution q(h).
        Only work for Bernoulli and Multinoulli distributions.
        HP: each column of HP[l] is a sample.
        """
        print("I am computing entropy...")
        entropies=0
        num_samples=HP[0].shape[1]
        #print "There are {} samples".format(num_samples)

        if in_mdbn:
            last=self.NK-1 # do not consider the last hidden layer, as it will be considered in the joint MDBM
        else:
            last=self.NK
            
        for n in range(num_samples):
            entropies_n=0
            #print "there are {} hidden layers".format(self.NK)
            for l in range(last):
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
        

    def sample_h_given_b_hat(self, b_hat=None, hidden_type="Bernoulli", hidden_type_fixed_param=0, hidden_value_or_meanfield="value"):
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


    def update_rbms(self):
        """ 
        Update the parameters of separate RBMs.
        """
        
        for nk in range(self.NK):
            if nk==0: # first RBM
                a=self.a
                b=self.b[nk]
                W=self.W[nk]
            else:
                a=self.b[nk-1]
                b=self.b[nk]
                W=self.W[nk]
                
            self.rbms[nk].set_param(a,b,W)


    def get_param(self):
        return self.a,self.b,self.W,self.br,self.Wr


    def set_param(self, a=None, b=None, W=None, br=None, Wr=None, update_rbms=True):
        """
        param is a dict type.
        """
        if a is not None:
            self.a=a
        if b is not None:
            self.b=b
        if W is not None:
            self.W=W
        if br is not None:
            self.br=br
        if Wr is not None:
            self.Wr=Wr
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


    def save_sampling(self, XM, ifsort=True, dir_save="./", prefix="HM"):
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
