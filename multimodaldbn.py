
#from __future__ import division
import deep_boltzmann_machine
import multimodaldbm
import helmholtz_machine
import time
import numpy
#import math
import classification as cl
#import copy
import os

class multimodaldbn:
    def __init__(self, num_views=2, features=[None,None], visible_types=["Bernoulli","Bernoulli"], visible_type_fixed_param=[0,0], M=[1000,1000],K_view=[[1000,1000],[1000,1000]],K_joint=[2000,2000], fix_a_view=None, tol_poisson_max=8, rng=numpy.random.RandomState(100)):
        """
        When visible_types[v] is "Multinoulli", visible_type_fixed_param[v] is vector/list of dimensions of the multinoulli variables.
        fix_a_view is a list of bool types.
        """
        self.num_views=num_views
        self.features=features
        self.visible_types=visible_types # list, the type of visible variables for each view
        self.visible_type_fixed_param=visible_type_fixed_param # list
        self.M=M
        self.K_view=K_view
        self.K_joint=K_joint
        self.NK_view=[0]*self.num_views # number of layers for each view, program should allow zero hidden layer for class component
        self.NK_joint=len(self.K_joint)
        self.fix_a_view=fix_a_view
        if self.fix_a_view is None:
            self.fix_a_view=[False]*self.num_views
        self.rng=rng
        self.tol_poisson_max=tol_poisson_max        
        
        # initialize parameters
        self.a_view=[None]*self.num_views
        self.b_view=[None]*self.num_views
        self.W_view=[None]*self.num_views
        self.br_view=[None]*self.num_views
        self.Wr_view=[None]*self.num_views        
        self.a_joint=None
        self.b_joint=[None]
        self.W_joint=[None]
        self.a_view2joint=[None]*self.num_views
        self.W_view2joint=[None]*self.num_views # list of matrice, should be updated once W.joint gets updated; when the model is a multimodal RBM, it also hold parts of the weight matrix. 
        self.hm_view=[]
        self.mdbm_joint=None

        # initialize parameter of each view
        self.M_joint_component=[0]*self.num_views # numbers of input variables of the joint component which is a MDBM with no view-specific hidden layers
        self.visible_types_joint_component=["Bernoulli"]*self.num_views
        self.visible_type_fixed_param_joint_component=[0]*self.num_views
        fix_a_view_joint_component=[False]*self.num_views
        for v in range(self.num_views):
            # no hidden layer
            if self.K_view[v]==None:
                self.NK_view[v]=0 # there is 0 hidden layers for the v-th view
                # the number of input variables for the joint component
                dbm_view_no_hidden=deep_boltzmann_machine.deep_boltzmann_machine(M=self.M[v], K=[self.K_joint[0]], visible_type=self.visible_types[v], visible_type_fixed_param=self.visible_type_fixed_param[v], if_fix_vis_bias=self.fix_a_view[v], a=None, fix_a_log_ind=None, tol_poisson_max=self.tol_poisson_max, rng=rng)
                a,b,W=dbm_view_no_hidden.get_param()
                self.a_view[v]=a
                self.W_view[v]=W
                self.b_view[v]=b
                self.visible_type_fixed_param[v]=dbm_view_no_hidden.visible_type_fixed_param #maybe vector
                #self.a_view[v]=math.log(1/self.M[v])*numpy.ones(shape=(self.M[v],1)) # random initialize a (W and b do not exist)
                self.M_joint_component[v]=self.M[v] # add the # of input of the trival v-th view
                self.visible_types_joint_component[v]=self.visible_types[v]
                self.visible_type_fixed_param_joint_component[v]=self.visible_type_fixed_param[v]
                fix_a_view_joint_component[v]=self.fix_a_view[v] # in trivial case, usually not fix it
                self.hm_view.append(dbm_view_no_hidden)
                
            else: # self.K_view[v]!=None
                # there are at least one hidden layers in the v-th view
                self.NK_view[v]=len(self.K_view[v]) # real number of layers for each view
                hm_model=helmholtz_machine.helmholtz_machine(M=self.M[v], K=self.K_view[v], visible_type=self.visible_types[v], visible_type_fixed_param=self.visible_type_fixed_param[v], if_fix_vis_bias=self.fix_a_view[v], a=None, fix_a_log_ind=None, tol_poisson_max=self.tol_poisson_max, rng=rng) # use DBM to initialize parameters, DBM uses stack of RBMs. :
                a,b,W,br,Wr=hm_model.get_param() # get initialized parameters
                self.a_view[v]=a # when "Multinoulli", a, that is self.a_view[v] is a list.
                self.W_view[v]=W # when "Multinoulli", W[0], that is self.W_view[v][0], is a list.
                self.b_view[v]=b
                self.br_view[v]=br
                self.Wr_view[v]=Wr
                self.visible_type_fixed_param[v]=hm_model.visible_type_fixed_param #maybe vector
                # the number of input variables for the joint component
                self.M_joint_component[v]=self.K_view[v][-1]
                self.visible_types_joint_component[v]="Bernoulli"
                self.visible_type_fixed_param_joint_component[v]=0
                fix_a_view_joint_component[v]=True # non-trivial case, fix a when updating parameters of the joint component
                self.hm_view.append(hm_model)
        
        # initialize parameter of the joint component, the joint component is a multi-modal DBM who has no view-specific hidden layers
        # I algin a_joint to the view-specific components initialized above
        # I algin b_view[v] and W_view[v] to the joint MDBM's parameter
        mdbm_model=multimodaldbm.multimodaldbm(num_views=self.num_views, features=self.features, visible_types=self.visible_types_joint_component, visible_type_fixed_param=self.visible_type_fixed_param_joint_component, M=self.M_joint_component, K_view=[None]*self.num_views, K_joint=self.K_joint, fix_a_view=fix_a_view_joint_component, tol_poisson_max=self.tol_poisson_max, rng=self.rng)
        a_view_jc,b_view_jc,W_view_jc,a_joint_jc,b_joint_jc,W_joint_jc=mdbm_model.get_param()
        self.get_a_joint() # aligned to the view-specific components initialized above
        mdbm_model.get_a_joint()
        mdbm_model.a_joint=self.a_joint # for consistency, update the joint mdbm's a_joint
        self.b_joint=b_joint_jc
        self.W_joint=W_joint_jc
        self.mdbm_joint=mdbm_model
        #self.get_a_view2joint() # update self.a_view2joint
        self.get_W_view2joint() # update self.W_view2joint
        
        # for trivial views, make their b and W consistent with the joint MDBM
        for v in range(self.num_views):
            if self.K_view[v]==None:
                self.hm_view[v].set_param(a=None,b=[self.b_joint[0]],W=[self.W_view2joint[v]], update_rbms=False)
                self.W_view[v]=self.W_view2joint[v]
                self.b_view[v]=[self.b_joint[0]]
                

    def get_a_joint(self):
        """
        # Define self.fix_a_joint_log_ind, and update self.a_joint (vector) and self.a_view2joint (list)
        """
        self.a_view2joint=[None]*self.num_views
        for v in range(self.num_views):
            if v==0:
                if self.K_view[v]==None:
                    self.a_joint=self.a_view[v] # no hidden layers, WHAT IF GAUSSIAN WHICH HAS a1 and a2?
                    self.a_view2joint[v]=self.a_view[v]
                    self.fix_a_joint_log_ind=numpy.array([self.fix_a_view[v]]*self.M[v],dtype=bool)
                else:
                    self.a_joint=self.b_view[v][-1] # at least one hidden layer
                    self.a_view2joint[v]=self.b_view[v][-1]
                    self.fix_a_joint_log_ind=numpy.array([True]*self.K_view[v][0],dtype=bool)
            else:
                if self.K_view[v]==None:
                    self.a_joint=numpy.vstack((self.a_joint,self.a_view[v]))
                    self.a_view2joint[v]=self.a_view[v]
                    self.fix_a_joint_log_ind=numpy.concatenate((self.fix_a_joint_log_ind, numpy.array([self.fix_a_view[v]]*self.M[v],dtype=bool) ))
                else:
                    self.a_joint=numpy.vstack((self.a_joint,self.b_view[v][-1]))
                    self.a_view2joint[v]=self.b_view[v][-1]
                    self.fix_a_joint_log_ind=numpy.concatenate((self.fix_a_joint_log_ind, numpy.array([True]*self.K_view[v][0],dtype=bool) ))


    def get_grad_a_joint(self,grad_a_view2joint):
        for v in range(self.num_views):
            if v==0:
                grad_a_joint=grad_a_view2joint[v]
            else:
                grad_a_joint=numpy.vstack((grad_a_joint,grad_a_view2joint[v]))
        return grad_a_joint


    def get_X_joint(self, X, Hr_view):
        """
        Get psuedo input of the joint MDBM.
        X_joint is a list of length V.
        """
        X_joint=[None]*self.num_views
        for v in range(self.num_views):
            if self.NK_view[v]==0:
                X_joint[v]=X[v]
            else:
                if Hr_view[v] is not None:
                    X_joint[v]=Hr_view[v][-1]
                else: # Hr_view[v] is None, means data for this view is missing 
                    X_joint[v]=None # the joint component for this view must be None then
        return X_joint


    def get_W_view2joint(self):
        self.W_view2joint=[]
        first=0
        for v in range(self.num_views):
            if self.K_view[v] is not None:
                last=first+self.K_view[v][-1]
            else:
                last=first+self.M[v]
            self.W_view2joint.append(self.W_joint[0][first:last,:])
            first=last
            # if no hidden layer, update W_view as well
            if self.K_view[v] is None:
                self.W_view[v]=self.W_view2joint[v]
            
    
    def get_a_view2joint(self):
        self.a_view2joint=[]
        first=0
        for v in range(self.num_views):
            if self.K_view[v] is not None:
                last=first+self.K_view[v][-1]
            else:
                last=first+self.M[v]
            self.a_view2joint.append(self.a_joint[first:last,:])
            first=last
            # if no hidden layer, update a_view as well
            if self.K_view[v] is None:
                self.a_view[v]=self.a_view2joint[v]                


    def reinit_a(self, X=None, v=0):
        # X is the whole multi-view data
        if self.visible_types[v]=="Bernoulli":
            mean=numpy.mean(X[v], axis=1)
            mean.shape=(mean.size,1)
            var=numpy.var(X[v], axis=1)
            var.shape=(var.size,1)
            self.a_view[v]=mean
        elif self.visible_types[v]=="Gaussian":
            mean=numpy.mean(X[v], axis=1)
            mean.shape=(mean.size,1)
            var=10*numpy.var(X[v], axis=1)
            var.shape=(var.size,1)
            precision=1/var
            self.a_view[v][0]=mean*precision
            self.a_view[v][1]=-0.5*precision
        elif self.visible_types[v]=="Poisson":
            mean=numpy.mean(X[v], axis=1)
            mean.shape=(mean.size,1)
            #var=numpy.var(self.X, axis=1)                                                                                                                                    
            #var.shape=(var.size,1)                                                                                                                                           
            self.a_view[v]=numpy.log(mean)
        elif self.visible_types[v]=="NegativeBinomial":
            mean=numpy.mean(X[v], axis=1)
            mean.shape=(mean.size,1)
            #var=numpy.var(self.X, axis=1)                                                                                                                                    
            #var.shape=(var.size,1)                                                                                                                                           
            self.a=numpy.log(mean/(self.visible_type_fixed_param[v]+mean))
        elif self.visible_types[v]=="Multinoulli":
            for m in range(self.M[v]):
                mean=numpy.mean(X[v][m], axis=1)
                mean.shape=(mean.size,1)
                #var=numpy.var(self.X[m], axis=1)                                                                                                                             
                #var.shape=(var.size,1)                                                                                                                                       
                self.a_view[v][m]=numpy.log(mean/mean.sum())
        elif self.visible_types[v]=="Multinomial":
            mean=numpy.mean(X[v], axis=1)
            mean.shape=(mean.size,1)
            #var=numpy.var(self.X, axis=1)                                                                                                                                    
            #var.shape=(var.size,1)                                                                                                                                           
            #self.a_view[v]=numpy.log(mean/mean.sum())
            self.a_view[v]=mean/mean.sum()
        elif self.visible_types[v]=="Gamma":
            mean=numpy.mean(X[v]+1, axis=1)
            mean.shape=(mean.size,1)
            var=numpy.var(X[v]+1, axis=1)
            var.shape=(var.size,1)
            self.a_view[v][0]=mean**2/var - 1                                                                                                                                        
            self.a_view[v][1]=-mean/var
        
        # update the corresponding HM
        self.hm_view[v].set_param(a=self.a_view[v], b=None, W=None, update_rbms=False)
        self.a_view2joint[v]=self.a_view[v]

    
    def pretrain(self, X=None, just_pretrain_HMDBM=True, batch_size=10, NMF=10, pcdk=20, NS=10, maxiter=20, learn_rate_a=0.001, learn_rate_b=0.001, learn_rate_W=0.001, change_rate=0.5, adjust_change_rate_at=None, adjust_coef=1.02, change_every_many_iters=20, init_chain_time=100, train_subset_size_for_compute_error=100, valid_subset_size_for_compute_error=100, track_reconstruct_error=True, track_free_energy=True, reinit_a_use_data_stat=False, if_plot_error_free_energy=False, dir_save="./", prefix="pretrain_HM", figwidth=5, figheight=3):
        """
        Pretrain the multi-modal DBN using HMs. 
        learn_rate_a, b, W: respectively a list of length V+1. The last element is for the joint layer
        reinit_a_use_data_stat is a list of bool.
        """
        print("Start pretraining multimodal HM ...")
        start_time = time.clock()
        
        if numpy.isscalar(learn_rate_a):
            learn_rate_a=[learn_rate_a]*(self.num_views+1)
        if numpy.isscalar(learn_rate_b):
            learn_rate_b=[learn_rate_b]*(self.num_views+1)
        if numpy.isscalar(learn_rate_W):
            learn_rate_W=[learn_rate_W]*(self.num_views+1)
            
        self.X=X
        self.batch_size=batch_size
        self.NS=NS
        self.NMF=NMF
        self.pcdk=pcdk
        self.missing_view_train=[True]*self.num_views
        for v in range(self.num_views):
            if self.X[v] is not None:
                self.missing_view_train[v]=False
        
        input_joint=[None]*self.num_views
        for v in range(self.num_views):
            if self.X[v] is not None and self.visible_types[v]=="Multinoulli": # convert to binary
                self.X[v]=[None]*self.M[v]
                self.X_validate[v]=[None]*self.M[v]
                for m in range(self.M[v]):
                    Z,_=cl.membership_vector_to_indicator_matrix(X[v][m,:], z_unique=list(range(self.visible_type_fixed_param[v][m])))
                    self.X[v][m]=Z.transpose()
                self.N=self.X[v][0].shape[1]
            elif self.X[v] is not None and self.visible_types[v]!="Multinoulli": # not multinoulli variables
                self.N=self.X[v].shape[1] # number of training samples        
        
#        #self.hm_view=[] # list of view-specific HMS, already assigned in the __init__ function.
#        # pretrain each view-specific HM
#        for v in range(self.num_views):
#
            # pretrain each view-specific HM
            # if there is no hidden layers, just ignore/skip this view
            if self.K_view[v]==None:
                if reinit_a_use_data_stat[v]:
                    self.reinit_a(self.X,v)
                input_joint[v]=self.X[v]
                continue
            
            # there is at least one hidden layer
            hm_model=self.hm_view[v]
            if self.fix_a_view[v]:
                hm_model.fix_vis_bias(a=None,fix_a_log_ind=None)
            hm_model.pretrain(X=self.X[v], batch_size=batch_size, pcdk=pcdk, NS=NS ,maxiter=maxiter, learn_rate_a=learn_rate_a[v], learn_rate_b=learn_rate_b[v], learn_rate_W=learn_rate_W[v], change_rate=change_rate, adjust_change_rate_at=adjust_change_rate_at, adjust_coef=adjust_coef, change_every_many_iters=change_every_many_iters, init_chain_time=init_chain_time, train_subset_size_for_compute_error=train_subset_size_for_compute_error, valid_subset_size_for_compute_error=valid_subset_size_for_compute_error, track_reconstruct_error=track_reconstruct_error, track_free_energy=track_free_energy, reinit_a_use_data_stat=reinit_a_use_data_stat[v], if_plot_error_free_energy=if_plot_error_free_energy, dir_save=dir_save, prefix=prefix+"_pretrain_HM_"+str(v), figwidth=figwidth, figheight=figheight)
            
            # fine-tune each view-specific HM
            if not just_pretrain_HMDBM and self.NK_view[v]>1: # deep HM
                hm_model.train(X=self.X[v], batch_size=batch_size, maxiter=maxiter, learn_rate_a=learn_rate_a[v], learn_rate_b=learn_rate_b[v], learn_rate_W=learn_rate_W[v], change_rate=change_rate, adjust_change_rate_at=adjust_change_rate_at, adjust_coef=adjust_coef, change_every_many_iters=change_every_many_iters, init_chain_time=init_chain_time, train_subset_size_for_compute_error=train_subset_size_for_compute_error, valid_subset_size_for_compute_error=valid_subset_size_for_compute_error, track_reconstruct_error=track_reconstruct_error, track_free_energy=track_free_energy, if_plot_error_free_energy=if_plot_error_free_energy, dir_save=dir_save, prefix=prefix+"_train_HM_"+str(v), figwidth=figwidth, figheight=figheight)    
                
                # I should update H_pretrain here which is used in pretraining the joint DBM
                _,_,_,HMr,_,_=hm_model.sample_xh_wake(X=self.X[v], compute_HMg=False)
                hm_model.H_pretrain=HMr
            
            #get parameter
            a,b,W,br,Wr=hm_model.get_param()
            self.a_view[v]=a
            self.b_view[v]=b
            self.W_view[v]=W
            self.br_view[v]=br
            self.Wr_view[v]=Wr
            # update the saved HMs for each view with at least one hidden layers
            self.hm_view[v]=hm_model # this statement is not necessary, as hm_model is passed by reference
            # the input for the joint MDBM
            input_joint[v]=hm_model.H_pretrain[-1]

        # pretrain the joint MDBM
        print("pretraining the joint MDBM ...")
        mdbm_model=self.mdbm_joint
        self.get_a_joint()
        mdbm_model.a_joint=self.a_joint
        mdbm_model.a_view=self.a_view2joint
        mdbm_model.a_view2joint=self.a_view2joint
        mdbm_model.pretrain(X=input_joint, just_pretrain_DBM=just_pretrain_HMDBM, batch_size=batch_size, NMF=NMF, pcdk=pcdk, NS=NS, maxiter=maxiter, learn_rate_a=learn_rate_a[self.num_views], learn_rate_b=learn_rate_b[self.num_views], learn_rate_W=learn_rate_W[self.num_views], change_rate=change_rate, adjust_change_rate_at=adjust_change_rate_at, adjust_coef=adjust_coef, change_every_many_iters=change_every_many_iters, init_chain_time=init_chain_time, train_subset_size_for_compute_error=train_subset_size_for_compute_error, valid_subset_size_for_compute_error=valid_subset_size_for_compute_error, track_reconstruct_error=track_reconstruct_error, track_free_energy=track_free_energy, reinit_a_use_data_stat=[False]*self.num_views, if_plot_error_free_energy=if_plot_error_free_energy, dir_save=dir_save, prefix="pretrain_MDBM_joint", figwidth=figwidth, figheight=figheight)
        
        a_view_jc,b_view_jc,W_view_jc,a_joint_jc,b_joint_jc,W_joint_jc=mdbm_model.get_param()
        self.a_joint=a_joint_jc
        self.b_joint=b_joint_jc
        self.W_joint=W_joint_jc
        self.get_W_view2joint() # updated W_view2joint and W_view[v] when view v is trivial
        self.get_a_view2joint() # update a_view2joint and a_view[v] when view v is trivial
        self.mdbm_joint=mdbm_model # not necessary, as passed by reference
        
        # update the view-specific DBMs with no hidden layer
        for v in range(self.num_views):
            if self.K_view[v] is None:
                self.b_view[v]=[self.b_joint[0]] # update b_view[v] if view v is trivial
                a=self.a_view2joint[v]# or self.a_view[v]
                b=[self.b_joint[0]]
                W=[self.W_view2joint[v]]
                self.hm_view[v].set_param(a=a, b=b, W=W, update_rbms=False)
        
        end_time = time.clock()
        self.pretrain_time=end_time - start_time
        print("Finished pretraining multimodal DBN.")
        return self.pretrain_time


    def train(self, X=None, X_validate=None, batch_size=10, NMF=10, pcdk=20, NS=10, maxiter=20, learn_rate_a=0.001, learn_rate_b=0.001, learn_rate_W=0.001, change_rate=0.5, adjust_change_rate_at=None, adjust_coef=1.02, change_every_many_iters=20, init_chain_time=100, track_reconstruct_error=True, valid_subset_size_for_compute_error=100, track_free_energy=False, if_plot_error_free_energy=False, dir_save="./", prefix="MDBN", figwidth=5, figheight=3 ):
        """
        In future improvement, it will be able to handle missing views.
        learn_rate_a, b, W: each is a list of length V+1, the last emelent is for the joint component.
        """
        print("Start training multimodal DBN...")
        start_time = time.clock()
        # initialization, input data reprocessing
        self.X=X
        self.X_validate=X_validate
        for v in range(self.num_views):
            if self.X[v] is not None and self.visible_types[v]=="Multinoulli": # convert to binary
                self.X[v]=[None]*self.M[v]
                self.X_validate[v]=[None]*self.M[v]
                for m in range(self.M[v]):
                    Z,_=cl.membership_vector_to_indicator_matrix(X[v][m,:], z_unique=list(range(self.visible_type_fixed_param[v][m])))
                    self.X[v][m]=Z.transpose()
                    Z,_=cl.membership_vector_to_indicator_matrix(X_validate[v][m,:], z_unique=list(range(self.visible_type_fixed_param[v][m])))
                    self.X_validate[v][m]=Z.transpose()
                self.N=self.X[v][0].shape[1]
                if self.X_validate is not None: 
                    self.N_validate=self.X_validate[v][0].shape[1] # number of validation samples
                else:
                    self.N_validate=0
            elif self.X[v] is not None and self.visible_types[v]!="Multinoulli": # not multinoulli variables
                self.N=self.X[v].shape[1] # number of training samples
                if self.X_validate is not None: 
                    self.N_validate=self.X_validate[v].shape[1] # number of validation samples
                else:
                    self.N_validate=0
        
        #if self.if_multimodal_RBM:
        if numpy.sum(self.NK_view)+self.NK_joint==1:
            end_time = time.clock()
            self.train_time= end_time - start_time
            print("No need to fine-tune, because it is a multimodal-RBM!")
            return self.train_time
            
        # actually a (multi-modal) RBM in the joint MDBM
        if self.NK_joint==1:
            NMF=1
        
        if numpy.isscalar(learn_rate_a):
            learn_rate_a=[learn_rate_a]*(self.num_views+1)
        if numpy.isscalar(learn_rate_b):
            learn_rate_b=[learn_rate_b]*(self.num_views+1)
        if numpy.isscalar(learn_rate_W):
            learn_rate_W=[learn_rate_W]*(self.num_views+1)
        self.learn_rate_a=learn_rate_a
        self.learn_rate_b=learn_rate_b
        self.learn_rate_W=learn_rate_W
        
        self.batch_size=batch_size
        self.NMF=NMF
        self.pcdk=pcdk
        self.cdk=pcdk
        self.NS=NS
        self.maxiter=maxiter
        self.change_rate=change_rate
        self.change_every_many_iters=change_every_many_iters
        
        self.rec_errors_train=[]
        self.rec_errors_valid=[]
        self.mfes_train=[]
        self.mfes_valid=[]
        
        # indicator of missing views       
        self.missing_view_train=[True]*self.num_views
        for v in range(self.num_views):
            if self.X[v] is not None:
                self.missing_view_train[v]=False
        self.mdbm_joint.missing_view_train=self.missing_view_train

        self.batch_size=batch_size
        if self.batch_size>self.N:
            self.batch_size=1
        
        # assign batch size to subnetworks
        for v in range(self.num_views):
            self.hm_view[v].batch_size=self.batch_size
        self.mdbm_joint.batch_size=self.batch_size
        
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
            self.learn_rate_a=self.change_learning_rate(current_learn_rate=self.learn_rate_a, change_rate=change_rate, current_iter=i, change_every_many_iters=self.change_every_many_iters)
            self.learn_rate_b=self.change_learning_rate(current_learn_rate=self.learn_rate_b, change_rate=change_rate, current_iter=i, change_every_many_iters=self.change_every_many_iters)
            self.learn_rate_W=self.change_learning_rate(current_learn_rate=self.learn_rate_W, change_rate=change_rate, current_iter=i, change_every_many_iters=self.change_every_many_iters)
            
            # update the learning rates for subnetworks
            for v in range(self.num_views):
                self.hm_view[v].learn_rate_a=self.learn_rate_a[v]
                self.hm_view[v].learn_rate_b=self.learn_rate_b[v]
                if numpy.isscalar(self.learn_rate_b[v]):
                    self.hm_view[v].learn_rate_b=[self.hm_view[v].learn_rate_b]*self.NK_view[v]
                self.hm_view[v].learn_rate_W=self.learn_rate_W[v]
                if numpy.isscalar(self.learn_rate_W[v]):
                    self.hm_view[v].learn_rate_W=[self.hm_view[v].learn_rate_W]*self.NK_view[v]
            
            self.mdbm_joint.learn_rate_a=self.learn_rate_a[self.num_views]
            if numpy.isscalar(self.mdbm_joint.learn_rate_a):
                self.mdbm_joint.learn_rate_a=[self.mdbm_joint.learn_rate_a]*(self.num_views+1)
            self.mdbm_joint.learn_rate_b=self.learn_rate_b[self.num_views]
            if numpy.isscalar(self.mdbm_joint.learn_rate_b):
                self.mdbm_joint.learn_rate_b=[self.mdbm_joint.learn_rate_b]*(self.num_views+1)
            self.mdbm_joint.learn_rate_W=self.learn_rate_W[self.num_views]
            if numpy.isscalar(self.mdbm_joint.learn_rate_W):
                self.mdbm_joint.learn_rate_W=[self.mdbm_joint.learn_rate_W]*(self.num_views+1)
                
            # get mini-batch
            Xbatch=self.sample_minibatch(self.batch_size)
            # wake phase
            XbatchMg_view,HbatchMg_view,Hbatchr_view,HbatchMr_view, Xbatch_joint,Hbatch_view_joint,Xbatch_joint_joint,Hbatch_joint_joint, Xfantacy_joint,XMfantacy_joint,Hfantacy_view_joint,Xfantacy_joint_joint,Hfantacy_joint_joint,a_hat_gen_view,b_hat_gen_view = self.sample_xh_wake(X=Xbatch, compute_HMg=True)
            # compute gradients of generative parameters
            self.compute_gradient_wake(Xbatch, XbatchMg_view=XbatchMg_view, HbatchMg_view=HbatchMg_view, Hbatchr_view=Hbatchr_view, Xbatch_joint=Xbatch_joint, Hbatch_view_joint=Hbatch_view_joint, Xbatch_joint_joint=Xbatch_joint_joint, Hbatch_joint_joint=Hbatch_joint_joint, Xfantacy_joint=Xfantacy_joint, Hfantacy_view_joint=Hfantacy_view_joint, Xfantacy_joint_joint=Xfantacy_joint_joint, Hfantacy_joint_joint=Hfantacy_joint_joint)
            
            ## update generative parameters, can be here as well
            self.update_param_wake()
        
            # sleep phase
            Xfantacy,_,Hfantacy,HfantacyMr=self.sample_xh_sleep(Xfantacy_joint, XMfantacy_joint)       
            # compute gradients of recognition parameters
            self.compute_gradient_sleep(Xfantacy, Hfantacy, HfantacyMr)
            
            # update generative parameters
            #self.update_param_wake()                        
            # update recognition parameters
            self.update_param_sleep()
            
            # compute reconstruct error and free energy on training subset
            if track_reconstruct_error:
                rec_errors_train=self.compute_reconstruction_error(Xbatch, XbatchMg_view)
                rec_errors_train_str="_".join(numpy.array(rec_errors_train,dtype=str))
                self.rec_errors_train.append(numpy.mean(rec_errors_train))
                
            if track_free_energy:
                mfe_train=self.compute_free_energy(X=Xbatch, HMr_view=HbatchMr_view, a_hat_gen_view=a_hat_gen_view, b_hat_gen_view=b_hat_gen_view, X_joint=Xbatch_joint, H_view_joint=Hbatch_view_joint, H_joint_joint=Hbatch_joint_joint)
                self.mfes_train.append(mfe_train)
            
            # compute reconstruct error and free energy on validation subset
            if self.X_validate is not None:
                if valid_subset_size_for_compute_error is not None:
                    valid_subset_ind=self.rng.choice(numpy.arange(self.N_validate,dtype=int),size=valid_subset_size_for_compute_error)
                    X_validate_subset=[None]*self.num_views
                    for v in range(self.num_views):
                        if self.visible_types[v]=="Multinoulli":
                            X_validate_subset[v]=[None]*self.M
                            for m in range(self.M):
                                X_validate_subset[v][m]=self.X_validate[v][m][:,valid_subset_ind]
                        else:
                            X_validate_subset[v]=self.X_validate[v][:,valid_subset_ind]
                    if track_reconstruct_error:
                        rec_errors_valid=self.compute_reconstruction_error(X0=X_validate_subset, X0RM=None)
                    if track_free_energy:
                        mfe_validate=self.compute_free_energy(X=X_validate_subset)
                else:
                    if track_reconstruct_error:                    
                        rec_errors_valid=self.compute_reconstruction_error(X0=self.X_validate, X0RM=None)
                    if track_free_energy:
                        mfe_validate=self.compute_free_energy(X=self.X_validate)
                if track_reconstruct_error:
                    self.rec_errors_valid.append(numpy.mean(rec_errors_valid))
                    rec_errors_valid_str="_".join(numpy.array(rec_errors_valid,dtype=str))
                if track_free_energy:
                    self.mfes_valid.append(mfe_validate)

            # print info
            if track_reconstruct_error and track_free_energy:
                if self.X_validate is not None:
                    free_energy_dif=mfe_train - mfe_validate
                    print("{0}-th iteration, learn_rate_W: {1}, train_rec_error: {2}, valid_rec_error: {3}, free_energy_train: {4}, free_energy_valid: {5}, free_energy_dif: {6}".format(i, self.learn_rate_W[0], rec_errors_train_str, rec_errors_valid_str, mfe_train, mfe_validate, free_energy_dif))
                else:
                    print("{0}-th iteration, learn_rate_W: {1}, train_rec_error: {2}, free_energy_train: {3}".format(i, self.learn_rate_W[0], rec_errors_train_str, mfe_train))
            elif not track_reconstruct_error and track_free_energy:
                if self.X_validate is not None:
                    free_energy_dif=mfe_train - mfe_validate
                    print("{0}-th iteration, learn_rate_W: {1}, free_energy_train: {2}, free_energy_valid: {3}, free_energy_dif: {4}".format(i, self.learn_rate_W[0], mfe_train, mfe_validate, free_energy_dif))
                else:
                    print("{0}-th iteration, learn_rate_W: {1}, free_energy_train: {2}".format(i, self.learn_rate_W[0], mfe_train))
            elif track_reconstruct_error and not track_free_energy:
                if self.X_validate is not None:
                    print("{0}-th iteration, learn_rate_W: {1}, train_rec_error: {2}, valid_rec_error: {3}".format(i, self.learn_rate_W[0], rec_errors_train_str, rec_errors_valid_str))
                else:
                    print("{0}-th iteration, learn_rate_W: {1}, train_rec_error: {2}".format(i, self.learn_rate_W[0], rec_errors_train_str))
            elif not track_reconstruct_error and not track_free_energy:
                print("{0}-th iteration, learn_rate_W: {1}".format(i, self.learn_rate_W[0]))
                
        if if_plot_error_free_energy:
            self.plot_error_free_energy(dir_save, prefix=prefix, figwidth=figwidth, figheight=figheight)
        
        end_time = time.clock()
        self.train_time= end_time - start_time
        print("The training (fine-tuning) of multimodal DBN is finished!")
        return self.train_time


    def sample_xh_wake(self, X, compute_HMg=True, missing_view=None, pcdk=None, NMF=None, value_or_mean="value", clamp_observed_view=False, generate_joint_fantacy=True):
        """
        Use the recognition parameters and the top MDBM to sample hidden states.
        value_or_mean: whether use value of mean when pass the data up using recognition parameters.
        """
        Hr_view=[None]*self.num_views # sampled by recognition parameters
        HMr_view=[None]*self.num_views # the mean of H computed by recognition parameters, to compute the free energy
        HMg_view=[None]*self.num_views
        XMg_view=[None]*self.num_views
        a_hat_gen_view=[None]*self.num_views
        b_hat_gen_view=[None]*self.num_views
        
        if missing_view is None:
            missing_view=self.missing_view_train
        if pcdk is None:
            pcdk=self.pcdk
        if NMF is None:
            NMF=self.NMF
            
        # view-specific hidden states
        for v in range(self.num_views):
            if self.NK_view[v]>0 and not missing_view[v]: # view-specific HM and not missing data
                XMg_v,HMg_v,Hr_v,HMr_v,a_hat_gen_v,b_hat_gen_v=self.hm_view[v].sample_xh_wake(X=X[v], compute_HMg=compute_HMg)
                if value_or_mean=="value":
                    Hr_view[v]=Hr_v
                if value_or_mean=="mean":
                    Hr_view[v]=HMr_v
                HMr_view[v]=HMr_v
                HMg_view[v]=HMg_v
                XMg_view[v]=XMg_v
                a_hat_gen_view[v]=a_hat_gen_v
                b_hat_gen_view[v]=b_hat_gen_v
            
        # joint hidden states inferred by mean-field, the X_joint[v] will be None if data for view v is missing
        X_joint=self.get_X_joint(X, Hr_view)
        
        # mean-field to approxiamte hidden states and missing views in X_joint in the joint MDBM
        XR_joint,XRM_joint,XRP_joint,H_view_joint,HP_view_joint,X_joint_joint,H_joint_joint,HP_joint_joint=self.mdbm_joint.mean_field_approximate_inference(XbatchOrg=X_joint, missing_view=missing_view, NMF=NMF, rand_init_H=False, rand_init_missing_view=True, only_update_view_spec_component_with_observed_data=False)
        
        for v in range(self.num_views):
            if missing_view[v]:
                X_joint[v]=XRM_joint[v]
        
        for v in range(self.num_views):
            if self.NK_view[v]==0:
                XMg_view[v]=XRM_joint[v]
        
        # run the cd-k Gibbs sampling on the top MDBM to generate fantacy
        # use the mean-field result to initialize the Markov chain
        if generate_joint_fantacy:
            for v in range(self.num_views):
                if missing_view[v]==False:
                    if self.visible_types[v]=="Multinoulli":
                        NS=X[v][0].shape[1]
                    else:
                        NS=X[v].shape[1]
            Xfantacy_joint,Hfantacy_view_joint,Xfantacy_joint_joint,Hfantacy_joint_joint,XMfantacy_joint,XPfantacy_joint,HPfantacy_view_joint,XPfantacy_joint_joint,HPfantacy_joint_joint,chain_length = self.mdbm_joint.pcd_sampling(pcdk=pcdk, NS=NS, X0=X_joint, H0_view=H_view_joint, H0_joint=H_joint_joint, missing_view=missing_view, clamp_observed_view=clamp_observed_view, only_update_view_spec_component_with_observed_data=False, persistent=True, init_sampling=True, rand_init_X=False, rand_init_missing_view=False, rand_init_H=False) # do not initialize the sampling using mean!
        else: # do not run Gibbs sampling, this function is probably called by generate_missing_x using mean-field method
            Xfantacy_joint=XRM_joint # let the joint "fantacies" equal to the mean-field results
            XMfantacy_joint=XRM_joint
            Hfantacy_view_joint=HP_view_joint
            Xfantacy_joint_joint=X_joint_joint
            Hfantacy_joint_joint=HP_joint_joint
            
        #if self.NK_joint==1: # joint RBM, do not use mean
        #    HP_view_joint=H_view_joint
        #    HP_joint_joint=H_joint_joint
            
        return XMg_view,HMg_view,Hr_view,HMr_view, X_joint,HP_view_joint,X_joint_joint,HP_joint_joint, Xfantacy_joint,XMfantacy_joint,Hfantacy_view_joint,Xfantacy_joint_joint,Hfantacy_joint_joint,a_hat_gen_view,b_hat_gen_view

        
    def compute_gradient_wake(self, Xbatch, XbatchMg_view, HbatchMg_view, Hbatchr_view, Xbatch_joint, Hbatch_view_joint, Xbatch_joint_joint, Hbatch_joint_joint, Xfantacy_joint, Hfantacy_view_joint, Xfantacy_joint_joint, Hfantacy_joint_joint):
        """
        Compute gradient in the wake phase to update the generative parameters.
        """
        for v in range(self.num_views):
            if self.NK_view[v]>0:
                self.hm_view[v].compute_gradient_wake(Xbatch[v], XbatchMg_view[v], Hbatchr_view[v], HbatchMg_view[v])
        
        # joint DBM
        self.mdbm_joint.compute_gradient( Xbatch_joint, Hbatch_view_joint, Xbatch_joint_joint, Hbatch_joint_joint, Xfantacy_joint, Hfantacy_view_joint, Xfantacy_joint_joint, Hfantacy_joint_joint)
        
            
    def update_param_wake(self):
        
        # update view-specific HMs
        for v in range(self.num_views):
            if self.NK_view[v]>0:
                self.hm_view[v].update_param_wake()
                self.hm_view[v].update_rbms()
                a,b,W,_,_=self.hm_view[v].get_param()
                self.a_view[v]=a
                self.b_view[v]=b
                self.W_view[v]=W
        
        # update joint MDBM
        self.get_a_joint() # a_view[v] for non-trivial case has been updated above, but will not be updated again below
        self.mdbm_joint.a_joint=self.a_joint
        self.mdbm_joint.a_view2joint=self.a_view2joint
        self.mdbm_joint.update_param(update_dbms=True)
        
        a_view_jc,b_view_jc,W_view_jc,a_joint_jc,b_joint_jc,W_joint_jc=self.mdbm_joint.get_param()
        self.a_joint=a_joint_jc
        self.b_joint=b_joint_jc
        self.W_joint=W_joint_jc
        self.get_W_view2joint() # update W_view2joint
        self.get_a_view2joint() # update a_view2joint and a_view[v] v is trivial
        
        # update the view-specific DBMs with no hidden layer
        for v in range(self.num_views):
            if self.K_view[v] is None:
                self.b_view[v]=[self.b_joint[0]] # update b_view[v] if view v is trivial
                if not self.fix_a_view[v]:
                    a=self.a_view2joint[v]# or self.a_view[v]
                else:
                    a=None
                b=[self.b_joint[0]]
                W=[self.W_view2joint[v]]
                self.hm_view[v].set_param(a=a, b=b, W=W, update_rbms=False)
                
        
    def sample_xh_sleep(self, Xfantacy_joint, XMfantacy_joint, value_or_mean="value", compute_HMr=True):
        """
        Use the generative parameters to sample hidden states and visible states.
        """
        Xg=[None]*self.num_views
        XMg=[None]*self.num_views
        Hg=[None]*self.num_views
        HMr=[None]*self.num_views
        #HMg=[None]*self.num_views
        for v in range(self.num_views):
            if self.NK_view[v]>0:
                Hg_v=[None]*self.NK_view[v]
                Hg_v[-1]=Xfantacy_joint[v]
                Xg_v,XMg_v,Hg_v,HMr_v=self.hm_view[v].sample_xh_sleep(NS=None, Hg=Hg_v, value_or_mean=value_or_mean, compute_HMr=compute_HMr)
                Xg[v]=Xg_v
                XMg[v]=XMg_v
                HMr[v]=HMr_v
                Hg[v]=Hg_v
            else: # trivial
                Xg[v]=Xfantacy_joint[v]
                XMg[v]=XMfantacy_joint[v]
            
        return Xg,XMg,Hg,HMr
        
        
    def compute_gradient_sleep(self, Xfantacy, Hfantacy, HfantacyMr):
        """
        Compute gradient in the sleep phase to update the recognition parameters.
        """
        for v in range(self.num_views):
            if self.NK_view[v]>0:
                self.hm_view[v].compute_gradient_sleep(Xfantacy=Xfantacy[v], Hfantacy=Hfantacy[v], HfantacyMr=HfantacyMr[v])
    
    
    def update_param_sleep(self):
        """
        Update the recognition parameters.
        """
        for v in range(self.num_views):
            if self.NK_view[v]>0:
                self.hm_view[v].update_param_sleep()
                _,_,W,br,Wr=self.hm_view[v].get_param()
                self.br_view[v]=br
                self.Wr_view[v]=Wr 
         

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
        

    def plot_error_free_energy(self, dir_save="./", prefix="MDBN", mean_over=5, figwidth=5, figheight=3):
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
        #plt.ion()
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


    def estimate_log_likelihood(self, X=None, NMF=100, base_rate_type="prior", beta=None, step_base=0.999, T=10000, stepdist="even", S=100, sumout="auto", dir_save="/.", prefix="MDBN"):
        """
        Estimate the log-likelihood of MDBN.
        """
        if X is None:
            X=self.X
        
        print("I am estimating the log-likelihood...")
        
        # compute E_q[log p*(x,h)]
        XMg_view,HMg_view,Hr_view,HMr_view,X_joint,H_view_joint,X_joint_joint,H_joint_joint, Xfantacy_joint,XMfantacy_joint,Hfantacy_view_joint,Xfantacy_joint_joint,Hfantacy_joint_joint,a_hat_gen_view,b_hat_gen_view=self.sample_xh_wake(X=X, compute_HMg=True)
        # free energy of view-specific HMs
        # E_q [log p] + H(q) for each view-specific HM
        loglh_hm=0
        for v in range(self.num_views):
            if self.NK_view[v]>0:
                mfe_hmv,_=self.hm_view[v].compute_free_energy(X=X[v], HMr=HMr_view[v], a_hat_gen=a_hat_gen_view[v], b_hat_gen=b_hat_gen_view[v], in_mdbn=True)             
                loglh_hm=loglh_hm-mfe_hmv
        # l(x) for the joint MDBM
        loglh_jc,logZ,logZ_std,mean_energy_jc,mean_entropy_jc=self.mdbm_joint.estimate_log_likelihood(X=X_joint, NMF=NMF, base_rate_type=base_rate_type, beta=beta, step_base=step_base, T=T, stepdist=stepdist, S=S, sumout=sumout, dir_save=dir_save, prefix="jointMDBM")
        
        loglh = loglh_hm + loglh_jc
        
        print("log-likelihood:{0}, logZ:{1}, logZ_std:{2}".format(loglh, logZ, logZ_std))
        
        # save results
        result=numpy.zeros(shape=(3,2),dtype=object)
        result[:,0]=numpy.array(["log-likelihood","logZ","logZ_std"])
        result[:,1]=numpy.array([loglh,logZ,logZ_std])
        filename=dir_save + prefix + "_estimated_log_likelihood.txt"
        numpy.savetxt(filename, result, delimiter="\t", fmt="%s")
        
        return loglh,logZ,logZ_std
        

    def compute_free_energy(self, X=None, HMr_view=None, a_hat_gen_view=None, b_hat_gen_view=None, X_joint=None, H_view_joint=None, H_joint_joint=None): 
        """
        Compute "free" energy E() with some layers summed out. 
        """
                    
        if HMr_view is None:
            _,_,_,HMr_view,X_joint,H_view_joint,_,H_joint_joint, _,_,_,_,_,a_hat_gen_view,b_hat_gen_view=self.sample_xh_wake(X=X, compute_HMg=False)
        
        mfe=0
        # free energy of view-specific HMs
        for v in range(self.num_views):
            if self.NK_view[v]>0:
                mfe_hmv,_=self.hm_view[v].compute_free_energy(X=X[v], HMr=HMr_view[v], a_hat_gen=a_hat_gen_view[v], b_hat_gen=b_hat_gen_view[v], in_mdbn=True)       
                mfe=mfe+mfe_hmv
                
        # free energy of the joint MDBM
        mfe_mdbm,_=self.mdbm_joint.compute_free_energy(X=X_joint, H_view=H_view_joint, H_joint=H_joint_joint)
        mfe=mfe+mfe_mdbm
        
        return mfe


    def compute_reconstruction_error(self, X0, X0RM):
        if X0RM is None:
            X0RM,_,_,_,_,_,_,_, _,_,_,_,_,_,_=self.sample_xh_wake(X=X0, compute_HMg=False)
            
        rec_errors=[0]*self.num_views
        for v in range(self.num_views):
            if self.NK_view[v]>0:
                rec_errors[v],_,_,_=self.hm_view[v].compute_reconstruction_error(X0[v], X0RM[v])
            else:
                rec_errors[v]=self.hm_view[v].compute_reconstruction_error(X0[v], X0RM[v])
        return rec_errors


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
        """
        Randomly sample a minibatch from the training data.
        INPUTS:
        batch_size: the size of minibatch.
        OUTPUTS:
        Xbatch: list of matrices.
        """
        ind_batch=self.rng.choice(self.N,size=batch_size,replace=False)
        Xbatch=[0]*self.num_views # just initialize it
        for v in range(self.num_views):
            if self.X[v] is not None:
                if self.visible_types[v]!="Multinoulli":
                    Xbatch[v]=self.X[v][:,ind_batch]
                    if batch_size==1:
                        Xbatch[v].shape=(self.M[v],1)
                elif self.visible_types[v]=="Multinoulli":
                    Xbatch[v]=[None]*self.M[v]
                    for m in range(self.M[v]):
                        Xbatch[v][m]=self.X[v][m][:,ind_batch]
                        if batch_size==1:
                            Xbatch[v][m].shape=(self.visible_type_fixed_param[v][m],1)
            else:
                Xbatch[v]=None
        return Xbatch
        

    def generate_x(self, pcdk=1000, NMF=100, NS=100, X0=None, init_sampling=False, save_prob=False, dir_save="./", prefix="MDBN"):
        """
        Generate multi-view X.
        If X0 is given, use mean-field to approximate the hidden states of the joint MDBM, use them to initialize the Gibbs sampling, use view-specific generative parameters to generate x. 
        If X0 is None (not given), use randomly initialized Gibbs sampling in the joint MDBM, and then use the view-specific generative parameters to generate x. 
        """
        if init_sampling:
            if X0 is not None: # X0 is completely observed.
                if self.NK_joint==1:
                    NMF=1
                _,_,_,_, X_joint,H_view_joint,_,H_joint_joint, Xfantacy_joint,XMfantacy_joint,Hfantacy_view_joint,_,Hfantacy_joint_joint,_,_=self.sample_xh_wake(X0, compute_HMg=False, NMF=NMF, pcdk=pcdk) # NMF and Gibbs sampling are used here, the Markov chain is initialized within this function
            else: # X0 is not given
                # initialize the Markov chain
                Xfantacy_joint,_,_,_,XMfantacy_joint,_,_,_,_,_=self.mdbm_joint.pcd_sampling(pcdk=pcdk, NS=NS, X0=None, H0_view=None, H0_joint=None, missing_view=None, clamp_observed_view=False, only_update_view_spec_component_with_observed_data=False, persistent=True, init_sampling=True, rand_init_X=True, rand_init_missing_view=False, rand_init_H=False)
            
            # use the view-specific generative parameters to generate X         
            Xg,XMg,_,_=self.sample_xh_sleep(Xfantacy_joint, XMfantacy_joint, value_or_mean="value", compute_HMr=False)
                
        # PCD sampling on the top MDBM
        if not init_sampling:
            Xfantacy_joint,_,_,_,XMfantacy_joint,_,_,_,_,_=self.mdbm_joint.pcd_sampling(pcdk=pcdk, init_sampling=False)
        
            # use the view-specific generative parameters to generate X 
            Xg,XMg,_,_=self.sample_xh_sleep(Xfantacy_joint, XMfantacy_joint, value_or_mean="value", compute_HMr=False)
        
        if save_prob:
            for v in range(self.num_views):
                filename=dir_save+prefix+"_sampling_xh_given_x_infer_prob_view_"+str(v)+".txt"
                numpy.savetxt(filename,XMg[v],fmt="%.5f",delimiter="\t")
        
        return Xg,XMg


    def generate_missing_x(self, pcdk=1000, NMF=100, X=None, missing_view=None, method="Gibbs_sampling", init_sampling=True, save_prob=False, dir_save="./", prefix="MDBN"):
        """
        Generate the missing views in x, while fixing some other views.
        When method is "mean_field", use mean-field to generate hidden states of the view-specific HMs, then use mean-field in the joint MDBM to generate hidden states and missing views in x_joint, then use mean-field to flow down them to generate missing views in x.
        When method is "Gibbs_sampling", use mean-field to generate hidden states of the view-specific HMs, then use Gibbs sampling while fixing the observed views in the joint MDBM to generate hidden states and missing views in x_joint, then use the values to flow down each view-specific HMs to generate missing views in x. 
        """
        
        if method=="mean_field":
            init_sampling=True
            
        if init_sampling:
            self.missing_view=missing_view
            # process multinoulli data            
            for v in range(self.num_views):
                if missing_view[v]==False:
                    if self.visible_types[v]=="Multinoulli":
                        X_binary=[None]*self.M[v]
                        for m in range(self.M[v]):
                            Z,_=cl.membership_vector_to_indicator_matrix(X[v][m,:],z_unique=list(range(self.visible_type_fixed_param[v][m])))
                            X_binary[m]=Z.transpose()
                            X[v]=X_binary
                else:
                    print("View {0} is missing, to be inferred ...".format(v))
                    
            # pass observed data upward using mean-field to the joint component, then use mean-field or Gibbs sampling in the joint component to generate fantacies of the joint component, finally generate x through flowing the generated joint fantacies using mean-field or Gibbs sampling. 
            if method=="mean_field":
                generate_joint_fantacy=False
            if method=="Gibbs_sampling":
                generate_joint_fantacy=True
            _,_,_,_,_,_,_,_, Xfantacy_joint,XMfantacy_joint,_,_,_,_,_=self.sample_xh_wake(X, compute_HMg=False, missing_view=missing_view, NMF=NMF, pcdk=pcdk, value_or_mean="mean", clamp_observed_view=True, generate_joint_fantacy=generate_joint_fantacy) # NMF and Gibbs sampling are used here
            
            if method=="mean_field":
                # use the mean of each to pass down
                # use the view-specific parameters to generate X 
                Xg,XMg,_,_=self.sample_xh_sleep(Xfantacy_joint, XMfantacy_joint, value_or_mean="mean", compute_HMr=False) # Xfantacy_joint actually is the mean-field result
            if method=="Gibbs_sampling":
                # use the view-specific generative parameters to generate X 
                Xg,XMg,_,_=self.sample_xh_sleep(Xfantacy_joint, XMfantacy_joint, value_or_mean="value", compute_HMr=False)
 
        if not init_sampling:
            Xfantacy_joint,_,_,_,_,_,_,_,_,_=self.mdbm_joint.pcd_sampling(pcdk=pcdk, init_sampling=init_sampling)
            # use the view-specific generative parameters to generate X 
            Xg,XMg,_,_=self.sample_xh_sleep(Xfantacy_joint, XMfantacy_joint, value_or_mean="value", compute_HMr=False)
        
        if save_prob:
            for v in range(self.num_views):
                if self.visible_types[v]!="Multinoulli":
                    filename=dir_save+prefix+"_sampling_missing_views_infer_prob_view_"+str(v)+".txt"
                    numpy.savetxt(filename,XMg[v],fmt="%.5f",delimiter="\t")
                else:
                    for m in range(self.M[v]):
                        filename=dir_save+prefix+"_sampling_missing_views_infer_prob_view_"+str(v)+"_var_"+str(m)+".txt"
                        numpy.savetxt(filename,XMg[v][m],fmt="%.5f",delimiter="\t")
        
        return Xg,XMg


    def get_param(self): 
        return self.a_view,self.b_view,self.W_view,self.br_view,self.Wr_view,self.a_joint,self.b_joint,self.W_joint
 

    def set_param(self, a_view, b_view, W_view, br_view, Wr_view, a_joint, b_joint, W_joint, update_hmmdbm=True):
        """
        Set the parameters of multimodal DBN.
        """
        # update view-specific components        
        for v in range(self.num_views):
            self.a_view[v]=a_view[v]
            self.b_view[v]=b_view[v]
            self.W_view[v]=W_view[v]
            self.br_view[v]=br_view[v]
            self.Wr_view[v]=Wr_view[v]
            if self.K_view[v] is not None and update_hmmdbm:
                a=self.a_view[v]
                b=self.b_view[v]
                W=self.W_view[v]
                br=self.br_view[v]
                Wr=self.Wr_view[v]
                self.hm_view[v].set_param(a=a, b=b, W=W, br=br, Wr=Wr, update_rbms=False)
        
        # update the joint component
        self.get_a_joint()
        #self.a_joint=param["a_joint"]
        self.b_joint=b_joint
        self.W_joint=W_joint
        self.get_a_view2joint()
        self.get_W_view2joint()
        if update_hmmdbm:
            a_view=self.a_view2joint
            b_view=[self.b_joint[0]]*self.num_views
            W_view=self.W_view2joint
            a_joint=self.a_joint
            b_joint=self.b_joint
            W_joint=self.W_joint
            self.mdbm_joint.set_param(a_view=a_view, b_view=b_view, W_view=W_view, a_joint=a_joint, b_joint=b_joint, W_joint=W_joint, update_rbms=False)
        
        # update dbm/rbm for trivial views
        for v in range(self.num_views):
            if self.K_view[v] is None and update_hmmdbm:
                a=self.a_view2joint[v]
                b=self.b_joint[0]
                W=self.W_view2joint[v]
                self.hm_view[v].set_param(a=a, b=b, W=W, update_rbms=False)


    def make_dir_save(self,parent_dir_save, prefix):
        K_view_str=""
        for v in range(self.num_views):
            if self.K_view[v] is None:
                K_view_v=[None]
            else:
                K_view_v=self.K_view[v]
            K_view_str=K_view_str+"_V"+str(v)+"_X_"+self.visible_types[v]+":"+str(self.M[v])+"_H_"+"_".join(numpy.array(K_view_v,dtype=str)) 
        K_joint_str="J_"+"_".join(numpy.array(self.K_joint,dtype=str))
        foldername=prefix + K_view_str + "_" + K_joint_str+ time.strftime("_%Y%b%d%X") + "/"
        dir_save=parent_dir_save+foldername
        self.dir_save=dir_save
        try:
            os.makedirs(dir_save)
        except OSError:
            #self.dir_save=parent_dir_save
            pass
        print("The results will be saved in " + self.dir_save)
        return self.dir_save


    def save_sampling(self, XM, if_sort=None, dir_save="./", prefix="MDBN"):
        """
        Save the sampling results for bag of word data.
        XM is a list, XM[v] is for the v-th view.
        if_sort is a list of bool types.
        """
        for v in range(self.num_views): 
            num_features=XM[v].shape[0]
            num_samples=XM[v].shape[1]
            XM_v_sorted=numpy.zeros(shape=(num_features,num_samples),dtype=float)
            features_v_sorted=numpy.zeros(shape=(num_features,num_samples),dtype=object)
            for n in range(num_samples):
                if if_sort[v]:
                    x=XM[v][:,n]
                    ind=numpy.argsort(x,kind="mergesort")
                    ind=ind[::-1]
                    XM_v_sorted[:,n]=x[ind]
                    features_v_sorted[:,n]=self.features[v][ind]
                else:
                    XM_v_sorted[:,n]=x
                    features_v_sorted[:,n]=self.features[v]
                
            filename=dir_save + prefix + "_view_" + str(v) +"_sampled_XM_sorted.txt"
            numpy.savetxt(filename,XM_v_sorted, fmt="%.4f", delimiter="\t")
            filename=dir_save + prefix + "_view_" + str(v) + "_sampled_features_sorted.txt"
            numpy.savetxt(filename,features_v_sorted, fmt="%s", delimiter="\t")
            

    def predict_sample_specific(self,XYP_infer,XY_test,tops=[10,20,50,100], save_result=True, dir_save="./", prefix="MDBM"):
        """
        This function is used to compute the sample specific prediction accuracy for the Ideation project only.
        """
        print("Computing sample specific prediction accuracy...")
        num_test=XYP_infer[1].shape[1] # number of test samples
        num_feat=XYP_infer[1].shape[0]
        scores_sorted=numpy.zeros(shape=(num_feat,num_test))
        Ytest_sorted=numpy.zeros(shape=(num_feat,num_test))
        features_sorted=numpy.zeros(shape=(num_feat,num_test),dtype=object)
        accs=numpy.zeros(shape=(num_test,))
        num_rank=len(tops)
        accss=numpy.zeros(shape=(num_rank,num_test))
        accs_means=[None]*num_rank
        accs_stds=[None]*num_rank
        for t in range(num_rank):
            top=tops[t]
            for n in range(num_test):
                prob1_n=XYP_infer[1][:,n]
                #prob0_n=XYP_infer[0][:,n]
                scores_n=prob1_n
                #scores_n=prob1_n*(prob1_n-prob0_n)
                ind=numpy.argsort(scores_n,kind="mergesort")
                ind=ind[::-1] # incremental order
                scores_sorted[:,n]=scores_n[ind]
                features_sorted[:,n]=self.features[0][ind]
                Y_n=XY_test[1][ind,n]
                Ytest_sorted[:,n]=Y_n
                accs[n]=numpy.mean(Y_n[0:top]>0)
            accs_mean=numpy.mean(accs)
            accs_std=numpy.std(accs)
            accss[t,:]=accs
            accs_means[t]=accs_mean
            accs_stds[t]=accs_std
            
            if False and save_result: # not save these results as it not very useful, and take many spaces
                filename=dir_save+prefix+"_sample_specific_performance_top_"+str(top)+".txt"
                file_handle=open(filename,'w') # create a new file
                file_handle.close()
                file_handle=open(filename,'a')
                numpy.savetxt(file_handle, ["Sample Specific Accuracies:"], fmt="%s",delimiter="\t")
                numpy.savetxt(file_handle, accs, fmt="%.4f",delimiter="\t")
                numpy.savetxt(file_handle, ["Mean Accuracy:"], fmt="%s",delimiter="\t")            
                numpy.savetxt(file_handle, [accs_mean], fmt="%.4f",delimiter="\t")
                numpy.savetxt(file_handle, ["STD:"], fmt="%s",delimiter="\t")            
                numpy.savetxt(file_handle, [accs_std], fmt="%.4f",delimiter="\t")
                file_handle.close()
        if save_result:
            filename=dir_save+prefix+"_sample_specific_scores_sorted.txt"
            numpy.savetxt(filename, scores_sorted, fmt="%.4f",delimiter="\t")
            filename=dir_save+prefix+"_sample_specific_Ytest_sorted.txt"
            numpy.savetxt(filename, Ytest_sorted, fmt="%d",delimiter="\t")
            filename=dir_save+prefix+"_sample_specific_features_sorted.txt"
            numpy.savetxt(filename, features_sorted, fmt="%s", delimiter="\t")
            filename=dir_save+prefix+"_sample_specific_accss.txt"            
            numpy.savetxt(filename, accss, fmt="%.4f", delimiter="\t")
            filename=dir_save+prefix+"_sample_specific_accs_means.txt"            
            numpy.savetxt(filename, accs_means, fmt="%.4f", delimiter="\t")
            filename=dir_save+prefix+"_sample_specific_accs_stds.txt"            
            numpy.savetxt(filename, accs_stds, fmt="%.4f", delimiter="\t")
            
        return accs_means,accs_stds,accss,scores_sorted,features_sorted
    