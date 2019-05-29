
#from __future__ import division
import deep_boltzmann_machine
import time
import numpy
#import math
import classification as cl
import copy
import os

class multimodaldbm:
    def __init__(self, num_views=2, features=[None,None], visible_types=["Bernoulli","Bernoulli"], visible_type_fixed_param=[0,0], M=[1000,1000],K_view=[[1000,1000],[1000,1000]],K_joint=[2000,2000], fix_a_view=None, tol_poisson_max=numpy.log(255), rng=numpy.random.RandomState(100)):
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
        self.a_joint=None
        self.b_joint=[None]
        self.W_joint=[None]
        self.a_view2joint=[None]*self.num_views
        self.W_view2joint=[None]*self.num_views # list of matrice, should be updated once W.joint gets updated; when the model is a multimodal RBM, it also hold parts of the weight matrix. 
        self.dbm_view=[]
        self.dbm_joint=None

        # initialize parameter of each view
        self.M_joint_component=0 # number of input variables of the joint component
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
                self.M_joint_component=self.M_joint_component+self.M[v] # add the # of input of the trival v-th view
                self.dbm_view.append(dbm_view_no_hidden)
                
            else: # self.K_view[v]!=None
                # there are at least one hidden layers in the v-th view
                self.NK_view[v]=len(self.K_view[v]) # real number of layers for each view
                dbm_model=deep_boltzmann_machine.deep_boltzmann_machine(M=self.M[v], K=self.K_view[v], visible_type=self.visible_types[v], visible_type_fixed_param=self.visible_type_fixed_param[v], if_fix_vis_bias=self.fix_a_view[v], a=None, fix_a_log_ind=None, tol_poisson_max=self.tol_poisson_max, rng=rng) # use DBM to initialize parameters, DBM uses stack of RBMs. :
                a,b,W=dbm_model.get_param() # get initialized parameters
                self.a_view[v]=a # when "Multinoulli", a, that is self.a_view[v] is a list.
                self.W_view[v]=W # when "Multinoulli", W[0], that is self.W_view[v][0], is a list.
                self.b_view[v]=b
                self.visible_type_fixed_param[v]=dbm_model.visible_type_fixed_param #maybe vector
                # the number of input variables for the joint component
                self.M_joint_component=self.M_joint_component+self.K_view[v][-1]
                self.dbm_view.append(dbm_model)

        # initialize parameter of the joint component
        self.num_views_none=0
        self.all_view_same_type=True
        for v in range(self.num_views):
            if self.visible_types[0]!=self.visible_types[v]:
                self.all_view_same_type=False
            if self.K_view[v] is None:
                self.num_views_none=self.num_views_none+1
            #if self.K_view[v] is not None:
            #    self.if_multimodal_RBM=False
                
        #self.if_multimodal_RBM is an indicator whether this multimodal DBM is just a multi-modal RBM: zero view-specific hidden layers and one hidden joint layer. If it is True, no pretraining involved.
        if self.NK_joint==1 and self.num_views_none==self.num_views:
            self.if_multimodal_RBM=True
        else:
            self.if_multimodal_RBM=False
            
        # if this MDBM is just an exp-DBM with all input views the same types
        if self.all_view_same_type and self.num_views_none==self.num_views:
            visible_type_joint=self.visible_types[0]
            visible_type_joint_fixed_param=self.visible_type_fixed_param[0]
        else:
            visible_type_joint="Bernoulli"
            visible_type_joint_fixed_param=0
            
        self.get_a_joint() # to be consistent, concatenate the bias of the last hidden layers of the view-specific components
        dbm_model=deep_boltzmann_machine.deep_boltzmann_machine(M=self.M_joint_component, K=K_joint, visible_type=visible_type_joint, visible_type_fixed_param=visible_type_joint_fixed_param, if_fix_vis_bias=any(self.fix_a_view), a=self.a_joint, fix_a_log_ind=self.fix_a_joint_log_ind, tol_poisson_max=self.tol_poisson_max, rng=rng)
        a,b,W=dbm_model.get_param()
        self.b_joint=b
        self.W_joint=W
        self.dbm_joint=dbm_model
        self.get_a_view2joint()
        self.get_W_view2joint()

        if not self.if_multimodal_RBM:
            for v in range(self.num_views):
                if self.K_view[v]==None:
                    self.dbm_view[v].set_param(a=None,b=[self.b_joint[0]],W=[self.W_view2joint[v]], update_rbms=False)
                    self.W_view[v]=self.W_view2joint[v]
                    self.b_view[v]=[self.b_joint[0]]
        else: # multimodal RBM
            for v in range(self.num_views):
                self.a_view2joint[v]=self.a_view[v]
                self.W_view2joint[v]=self.W_view[0] # self.W_view2joint[v] is a matrix
                # when multimodal RBM, self.b_joint is a [vector] and shared by all views
                if v==0:
                    self.b_joint=self.b_view[v] # b is a list
                else:
                    self.b_view[v]=self.b_joint


    def get_a_joint(self):
        """
        # Define self.fix_a_joint_log_ind and update self.a_joint
        """
        for v in range(self.num_views):
            if v==0:
                if self.K_view[v]==None:
                    self.a_joint=self.a_view[v] # no hidden layers, WHAT IF GAUSSIAN WHICH HAS a1 and a2?
                    self.fix_a_joint_log_ind=numpy.array([self.fix_a_view[v]]*self.M[v],dtype=bool)
                else:
                    self.a_joint=self.b_view[v][-1] # at least one hidden layer
                    self.fix_a_joint_log_ind=numpy.array([True]*self.K_view[v][0],dtype=bool)
            else:
                if self.K_view[v]==None:
                    self.a_joint=numpy.vstack((self.a_joint,self.a_view[v]))
                    self.fix_a_joint_log_ind=numpy.concatenate((self.fix_a_joint_log_ind, numpy.array([self.fix_a_view[v]]*self.M[v],dtype=bool) ))
                else:
                    self.a_joint=numpy.vstack((self.a_joint,self.b_view[v][-1]))
                    self.fix_a_joint_log_ind=numpy.concatenate((self.fix_a_joint_log_ind, numpy.array([True]*self.K_view[v][0],dtype=bool) ))


    def get_grad_a_joint(self,grad_a_view2joint):
        for v in range(self.num_views):
            if v==0:
                grad_a_joint=grad_a_view2joint[v]
            else:
                grad_a_joint=numpy.vstack((grad_a_joint,grad_a_view2joint[v]))
        return grad_a_joint


    def get_X_joint(self, X, H_view):
        # get psuedo input of the joint component
        for v in range(self.num_views):
            if self.NK_view[v]==0:
                X_joint_v=X[v]
            else:
                X_joint_v=H_view[v][-1]
            if v==0:
                X_joint=X_joint_v
            else:
                X_joint=numpy.vstack((X_joint,X_joint_v))
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
        
        # update the corresponding DBM
        self.dbm_view[v].set_param(a=self.a_view[v], b=None, W=None, update_rbms=False)
        self.a_view2joint[v]=self.a_view[v]

            
    def pretrain(self, X=None, just_pretrain_DBM=True, batch_size=10, NMF=10, pcdk=20, NS=10, maxiter=20, learn_rate_a=0.001, learn_rate_b=0.001, learn_rate_W=0.001, change_rate=0.5, adjust_change_rate_at=None, adjust_coef=1.02, change_every_many_iters=20, init_chain_time=100, train_subset_size_for_compute_error=100, valid_subset_size_for_compute_error=100, track_reconstruct_error=True, track_free_energy=True, reinit_a_use_data_stat=False, if_plot_error_free_energy=False, dir_save="./", prefix="DBM", figwidth=5, figheight=3):
        """
        Pretrain the multi-modal DBM using RBMs. 
        learn_rate_a, b, W: respectively a list of length V+1. The last element is for the joint layer
        reinit_a_use_data_stat is a list of bool.
        """
        print("Start pretraining multimodal DBM ...")
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
        
#        #self.dbm_view=[] # list of view-specific DBMS, already assigned in the __init__ function.
#        # pretrain each view-specific DBM
#        for v in range(self.num_views):
#
            # pretrain each view-specific DBM
            # if there is no hidden layers, just ignore/skip this view
            if self.K_view[v]==None:
                if reinit_a_use_data_stat[v]:
                    self.reinit_a(self.X,v)
                if v==0:
                    input_joint=self.X[v] # not work for multinoulli
                else:
                    input_joint=numpy.vstack((input_joint,self.X[v])) # not consider multinoulli types here, here is specific for multinomial input units
                continue
            
            
            # there is at least one hidden layer
            dbm_model=self.dbm_view[v]
            if self.fix_a_view[v]:
                dbm_model.fix_vis_bias(a=None,fix_a_log_ind=None)
            dbm_model.pretrain(X=self.X[v], batch_size=batch_size, pcdk=pcdk, NS=NS ,maxiter=maxiter, learn_rate_a=learn_rate_a[v], learn_rate_b=learn_rate_b[v], learn_rate_W=learn_rate_W[v], change_rate=change_rate, adjust_change_rate_at=adjust_change_rate_at, adjust_coef=adjust_coef, change_every_many_iters=change_every_many_iters, init_chain_time=init_chain_time, train_subset_size_for_compute_error=train_subset_size_for_compute_error, valid_subset_size_for_compute_error=valid_subset_size_for_compute_error, track_reconstruct_error=track_reconstruct_error, track_free_energy=track_free_energy, reinit_a_use_data_stat=reinit_a_use_data_stat[v], if_plot_error_free_energy=if_plot_error_free_energy, dir_save=dir_save, prefix=prefix+"_pretrain_DBM_"+str(v), figwidth=figwidth, figheight=figheight)
            
            # fine-tune each view-specific DBM
            if not just_pretrain_DBM and self.NK_view[v]>1:
                dbm_model.train(X=self.X[v], batch_size=batch_size, NMF=NMF, pcdk=pcdk, NS=NS, maxiter=maxiter, llearn_rate_a=learn_rate_a[v], learn_rate_b=learn_rate_b[v], learn_rate_W=learn_rate_W[v], change_rate=change_rate, adjust_change_rate_at=adjust_change_rate_at, adjust_coef=adjust_coef, change_every_many_iters=change_every_many_iters, init_chain_time=init_chain_time, train_subset_size_for_compute_error=train_subset_size_for_compute_error, valid_subset_size_for_compute_error=valid_subset_size_for_compute_error, track_reconstruct_error=track_reconstruct_error, track_free_energy=track_free_energy, if_plot_error_free_energy=if_plot_error_free_energy, dir_save=dir_save, prefix=prefix+"_train_DBM_"+str(v), figwidth=figwidth, figheight=figheight)    
                
                # I should update H_pretrain here using mean-field, since DBMs are fine-tuned
                _,_,_,_,HP=dbm_model.mean_field_approximate_inference(Xbatch=self.X[v],NMF=NMF,rand_init_H=False)
                dbm_model.H_pretrain=HP
            
            #get parameter
            a,b,W=dbm_model.get_param()
            self.a_view[v]=a
            self.b_view[v]=b
            self.W_view[v]=W
            # update the saved DBMs for each view with at least one hidden layers
            self.dbm_view[v]=dbm_model

            # the bias of the input layer of the joint component, be careful here
            if v==0:
                #self.a_joint=b[-1] # the visible bais of the joint DBM, a_joint can also be obtained by calling get_a_joint()
                input_joint=dbm_model.H_pretrain[-1] # the inputs of the joint DBM
            else:
                #self.a_joint=numpy.vstack((self.a_joint,b[-1]))
                input_joint=numpy.vstack((input_joint,dbm_model.H_pretrain[-1]))

        # pretrain the joint DBM
        #if not self.if_multimodal_RBM:
        if True:
            print("pretraining the joint DBM ...")
            self.get_a_joint()
            dbm_model=self.dbm_joint
            dbm_model.fix_vis_bias(a=self.a_joint,fix_a_log_ind=self.fix_a_joint_log_ind) # fix a_joint, NEED IMPROVEMENT!!!!!
            #dbm_model=deep_boltzmann_machine.deep_boltzmann_machine(M=self.M_joint_component, K=K_joint, visible_type="Bernoulli", rng=rng)
            #print input_joint.shape
            dbm_model.pretrain(X=input_joint, batch_size=batch_size, pcdk=pcdk, NS=NS ,maxiter=maxiter, learn_rate_a=learn_rate_a[self.num_views], learn_rate_b=learn_rate_b[self.num_views], learn_rate_W=learn_rate_W[self.num_views], change_rate=change_rate, adjust_change_rate_at=adjust_change_rate_at, adjust_coef=adjust_coef, change_every_many_iters=change_every_many_iters, init_chain_time=init_chain_time, train_subset_size_for_compute_error=train_subset_size_for_compute_error, valid_subset_size_for_compute_error=valid_subset_size_for_compute_error, track_reconstruct_error=track_reconstruct_error, track_free_energy=track_free_energy, if_plot_error_free_energy=if_plot_error_free_energy, dir_save=dir_save, prefix=prefix+"_pretrain_DBM_joint", figwidth=figwidth, figheight=figheight )
    
            # fine-tune the joint DBM
            if not just_pretrain_DBM and self.NK_joint>1:
                dbm_model.train(X=input_joint,batch_size=batch_size, NMF=NMF, pcdk=pcdk, NS=NS, maxiter=maxiter, learn_rate_a=learn_rate_a[self.num_views], learn_rate_b=learn_rate_b[self.num_views], learn_rate_W=learn_rate_W[self.num_views], change_rate=change_rate, adjust_change_rate_at=adjust_change_rate_at, adjust_coef=adjust_coef, change_every_many_iters=change_every_many_iters, init_chain_time=init_chain_time, train_subset_size_for_compute_error=train_subset_size_for_compute_error, valid_subset_size_for_compute_error=valid_subset_size_for_compute_error, track_reconstruct_error=track_reconstruct_error, track_free_energy=track_free_energy, if_plot_error_free_energy=if_plot_error_free_energy, dir_save=dir_save, prefix=prefix+"_train_DBM_joint", figwidth=figwidth, figheight=figheight )
            a,b,W=dbm_model.get_param()
            
            # update the paramter of the joint component
            self.a_joint=a
            self.b_joint=b
            self.W_joint=W
            # update another copy of the W_joint[0]; every time if W_joint is updated, update self.W_view2joint        
            self.get_W_view2joint()
            # update a_view[v] is view v has no hidden layer
            self.get_a_view2joint()
            # update DBM for the joint component
            self.dbm_joint=dbm_model       
            
            # update the DBMs for views with no hidden layer
            for v in range(self.num_views):
                if self.K_view[v] is None:
                    a=self.a_view2joint[v]# or self.a_view[v]
                    b=[self.b_joint[0]]
                    W=[self.W_view2joint[v]]
                    self.dbm_view[v].set_param(a=a, b=b, W=W, update_rbms=False)
        
        end_time = time.clock()
        self.pretrain_time=end_time - start_time
        print("Finished pretraining multimodal DBM.")
        return self.pretrain_time


    def train(self, X=None, X_validate=None, batch_size=10, NMF=10, pcdk=20, NS=10, maxiter=20, learn_rate_a=0.001, learn_rate_b=0.001, learn_rate_W=0.001, change_rate=0.5, adjust_change_rate_at=None, adjust_coef=1.02, change_every_many_iters=20, init_chain_time=100, track_reconstruct_error=True, valid_subset_size_for_compute_error=100, track_free_energy=False, if_plot_error_free_energy=False, dir_save="./", prefix="MDBM", figwidth=5, figheight=3 ):
        """
        In future improvement, it will be able to handle missing views.
        learn_rate_a, b, W: each is a list of length V+1, the last emelent is for the joint component.
        """
        print("Start training multimodal DBM...")
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
        
        if self.if_multimodal_RBM:
            end_time = time.clock()
            self.train_time= end_time - start_time
            print("No need to fine-tune, because it is a multimodal-RBM!")
            return self.train_time
            
        # actually a (multi-modal) RBM
        if self.num_views_none==self.num_views and len(self.K_joint)==1:
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

        self.batch_size=batch_size
        if self.batch_size>self.N:
            self.batch_size=1
            
        ## all view-specific components are None, thus, leave right now
        ## sometimes, for multi-view data, the situation is not this simple, the DBM pretraining is not the same as MDBM training, even for this trivial case
        #if self.all_view_same_type and self.num_views_none==self.num_views:    
        #    end_time = time.clock()
        #    self.train_time= end_time - start_time
        #    print "All view-specific components are None, and have the same type of input. I do not need to go through the training step again. Thus, I decide to leave now."
        #    return self.train_time
        
        # even if this is a multi-modal RBM, we still need to continue training
            
               # initialize Markov chains
        _,_,_,_,_,_,_,_,_,_=self.pcd_sampling(pcdk=init_chain_time*pcdk,NS=NS,persistent=True, init_sampling=True,rand_init_X=True,rand_init_H=False) # initialize pcd

        for i in range(self.maxiter):
            # get mini-batch
            Xbatch=self.sample_minibatch(self.batch_size)
            
            # mean-field approximation
            _,XbatchRM,_,Hbatch_view,HbatchP_view,Xbatch_joint,Hbatch_joint,HbatchP_joint=self.mean_field_approximate_inference(XbatchOrg=Xbatch,NMF=self.NMF,rand_init_H=False)
            print("mean-field samples:")
            print(Xbatch)
            print(HbatchP_view)

            # pcd sampling
            XS,HS_view,XS_joint,HS_joint,_,_,_,_,_,_=self.pcd_sampling(pcdk=self.pcdk,init_sampling=False)
            print("pcd samples:")
            print(XS)
            print(HS_view)

            # compute gradient
            if False and self.if_multimodal_RBM:
                self.compute_gradient(Xbatch,Hbatch_view,Xbatch_joint,Hbatch_joint,XS,HS_view,XS_joint,HS_joint)
            else:
                self.compute_gradient(Xbatch,HbatchP_view,Xbatch_joint,HbatchP_joint,XS,HS_view,XS_joint,HS_joint)
                
            if adjust_change_rate_at is not None:
                if i==adjust_change_rate_at[0]:
                    change_rate=change_rate*adjust_coef # increast change_rate
                    change_rate=1.0 if change_rate>1.0 else change_rate # make sure not greater than 1
                    if len(adjust_change_rate_at)>1:
                        adjust_change_rate_at=adjust_change_rate_at[1:] # delete the first element
                    else:
                        adjust_change_rate_at=None
                        
            self.learn_rate_a=self.change_learning_rate(current_learn_rate=self.learn_rate_a, change_rate=change_rate, current_iter=i, change_every_many_iters=self.change_every_many_iters)
            self.learn_rate_b=self.change_learning_rate(current_learn_rate=self.learn_rate_b, change_rate=change_rate, current_iter=i, change_every_many_iters=self.change_every_many_iters)
            self.learn_rate_W=self.change_learning_rate(current_learn_rate=self.learn_rate_W, change_rate=change_rate, current_iter=i, change_every_many_iters=self.change_every_many_iters)

            # update parameters
            self.update_param()
            
            # compute reconstruct error and free energy on training subset
            if track_reconstruct_error:
                rec_errors_train=self.compute_reconstruction_error(Xbatch, XbatchRM)
                rec_errors_train_str="_".join(numpy.array(rec_errors_train,dtype=str))
                self.rec_errors_train.append(numpy.mean(rec_errors_train))
            if track_free_energy:
                if False and self.if_multimodal_RBM:
                    mfe_train,_=self.compute_free_energy(X=Xbatch, H_view=Hbatch_view, H_joint=Hbatch_joint)
                else:
                    mfe_train,_=self.compute_free_energy(X=Xbatch, H_view=HbatchP_view, H_joint=HbatchP_joint)
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
                        mfe_validate,_=self.compute_free_energy(X=X_validate_subset, H_view=None, H_joint=None)
                else:
                    if track_reconstruct_error:                    
                        rec_errors_valid=self.compute_reconstruction_error(X0=self.X_validate, X0RM=None)
                    if track_free_energy:
                        mfe_validate,_=self.compute_free_energy(X=self.X_validate, H_view=None, H_joint=None)
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
        print("The training (fine-tuning) of multimodal DBM is finished!")
        return self.train_time


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
        

    def plot_error_free_energy(self, dir_save="./", prefix="MDBM", mean_over=5, figwidth=5, figheight=3):
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


    def estimate_log_likelihood(self, X=None, NMF=100, base_rate_type="prior", beta=None, step_base=0.999, T=10000, stepdist="even", S=100, sumout="auto", dir_save="/.", prefix="MDBM"):
        """
        Estimate the log-likelihood of MDBM.
        """
        if X is None:
            X=self.X
        # trivial case, just a multimodal RBM, set NMF=1
        if numpy.max(self.NK_view)==0 and self.NK_joint==1:
            NMF=1
            
        print("I am estimating the log-likelihood...")
        # E_q[-E(x,h)]
        _,_,_,_,HP_view,X_joint,_,HP_joint=self.mean_field_approximate_inference(XbatchOrg=X, missing_view=None, NMF=NMF, rand_init_H=False, rand_init_missing_view=False,  only_update_view_spec_component_with_observed_data=False)
        
        # compute energy
        mean_energy,_=self.compute_energy(X, HP_view, HP_joint)
        
        # compute entropy of approximate distributions
        #print HP_view
        #print HP_joint
        mean_entropy=self.compute_entropy(HP_view, HP_joint)
        
        # estimate logZ
        logZ,logZ_std,logws,log_ratio_AIS_mean,log_ratio_AIS_std=self.estimate_logZ(base_rate_type=base_rate_type, beta=beta, step_base=step_base, T=T, stepdist=stepdist, S=S, sumout=sumout)
        
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
        

    def compute_energy(self, X, H_view, H_joint):
        """
        Compute energy E(x,h) given x and h. 
        """
        print("I am computing energy E(X,h)...")
        if self.visible_types[0]!="Multinoulli":
            num_samples=X[0].shape[1]
        else:
            num_samples=X[0][0].shape[1]
        
        Es=0 # initialize energy
        X_joint=self.get_X_joint(X, H_view)
        
        for n in range(num_samples): # compute it for each sample is better when the number of samples is large
            En=0
            # view specific networks
            for v in range(self.num_views):
                Xvn=X[v][:,[n]]
                Hvn=[None]*self.NK_view[v]
                for l in range(self.NK_view[v]):
                    Hvn[l]=H_view[v][l][:,[n]]
                En = En - self.dbm_view[v].zeta(X=Xvn, H=Hvn, a=self.a_view[v], b=self.b_view[v], visible_type=self.visible_types[v], visible_type_fixed_param=self.visible_type_fixed_param[v], NK=self.NK_view[v], sumout=None)
                for l in range(self.NK_view[v]): # do not compute the interaction if this is a trivial view, the interaction will be computed in the joint part
                    if l==0:
                        if self.visible_types[v]=="Multinoulli":
                            for m in range(self.M):
                                En = En - numpy.dot(X[v][m][:,[n]].transpose(),self.W_view[v][l][m]).dot(H_view[v][l][:,[n]])
                        else:
                            En = En - numpy.dot(X[v][:,[n]].transpose(),self.W_view[v][l]).dot(H_view[v][l][:,[n]])
                    else: # not first layer
                        En = En - numpy.dot(H_view[v][l-1][:,[n]].transpose(),self.W_view[v][l]).dot(H_view[v][l][:,[n]])
                        
            # joint, even if this is a multimodal RBM
            for l in range(self.NK_joint):
                if l==0:
                    En = En - numpy.dot(H_joint[l][:,[n]].transpose(),self.b_joint[l]) - numpy.dot(X_joint[:,[n]].transpose(),self.W_joint[l]).dot(H_joint[l][:,[n]])
                else: # not first layer
                    En = En - numpy.dot(H_joint[l][:,[n]].transpose(),self.b_joint[l]) - numpy.dot(H_joint[l-1][:,[n]].transpose(),self.W_joint[l]).dot(H_joint[l][:,[n]])
                    
            Es=Es+En
        Es=Es[0,0] # take off [[]]
        ME=Es/num_samples # mean energy
        return ME,Es
    
    
    def compute_entropy(self, HP_view, HP_joint):
        """
        Compute the entropy of approximate distribution q(h).
        Only work for Bernoulli and Multinoulli distributions.
        HP: each column of HP[l] is a sample.
        """
        print("I am computing entropy...")
        mean_entropy=0
        # view-specific
        for v in range(self.num_views):
            if self.NK_view[v]>0: # not a trivial view
                ev,_=self.dbm_view[v].compute_entropy(HP_view[v])
                mean_entropy = mean_entropy + ev
        # joint
        #print "computing entropy for the joint component..."
        ev,_=self.dbm_joint.compute_entropy(HP_joint)
        mean_entropy = mean_entropy + ev
        #print "the mean entropy for MDBM is {}".format(mean_entropy)
        return mean_entropy
        
    
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
        
        a_view_A,b_view_A,a_joint_A,b_joint_A,logZ_A=self.compute_logZ_A(base_rate_type=base_rate_type)
        
        print("logZ_A={0}".format(logZ_A))
        
        print("I need to run AIS for {0} times.".format(S))
        logws=numpy.zeros(shape=(S,),dtype=float)
        #ws=numpy.zeros(shape=(S,),dtype=float)
        for s in range(S):
            print("I am running the {0}-th AIS...".format(s))
            # Markov chain
            x_t=[None]*self.num_views
            h_view_A_t=[None]*self.num_views
            h_view_B_t=[None]*self.num_views
            
            log_p_star_diff=numpy.zeros(shape=(T-1,),dtype=float)
            
            # just initialize h_B_t, for the first iteration where beta_t=0, it is not use
            h_view_B_t,_,h_joint_B_t,_=self.sample_hidden_MDBM(X=None, H_view=None, H_joint=None, a_view=self.a_view, b_view=self.b_view, W_view=None, a_joint=self.a_joint, b_joint=self.b_joint, W_joint=None, W_view2joint=None, beta_t=0)
            x_t,_,_=self.sample_visible_MDBM(visible_types=self.visible_types, visible_type_fixed_param=self.visible_type_fixed_param, a_view=self.a_view, W_view=None, H_view=None, NS=1, rng=self.rng)

            for t in range(T-1):
                # sample hidden
                # sample from model A
                h_view_A_t,_,h_joint_A_t,_=self.sample_hidden_MDBM(X=None, H_view=None, H_joint=None, a_view=a_view_A, b_view=b_view_A, W_view=None, a_joint=a_joint_A, b_joint=b_joint_A, W_joint=None, W_view2joint=None, beta_t=1-beta[t])
                # sample from model B
                h_view_B_t,_,h_joint_B_t,_=self.sample_hidden_MDBM(X=x_t, H_view=h_view_B_t, H_joint=h_joint_B_t, a_view=self.a_view, b_view=self.b_view, W_view=self.W_view, a_joint=self.a_joint, b_joint=self.b_joint, W_joint=self.W_joint, W_view2joint=self.W_view2joint, beta_t=beta[t])
                
                # sample visible
                a_view_B_hat,_,_=self.compute_posterior_bias(visible_types=self.visible_types, visible_type_fixed_param=self.visible_type_fixed_param, NK_view=self.NK_view, NK_joint=self.NK_joint, a_view=self.a_view, b_view=self.b_view, W_view=self.W_view, b_joint=self.b_joint, W_joint=self.W_joint, W_view2joint=self.W_view2joint, X=x_t, H_view=h_view_B_t, H_joint=h_joint_B_t, scale=1, compute_a_only=True)
                a_view_hat_t=self.combine_a_A_a_B_hat(a_view_A, a_view_B_hat, beta[t], self.visible_types)
                x_t,_,_=self.sample_visible_MDBM(visible_types=self.visible_types, visible_type_fixed_param=self.visible_type_fixed_param, a_view=a_view_hat_t, W_view=None, H_view=None, NS=1, rng=self.rng)
                
                # log p*()
                log_p_star_t_xh_t=self.log_p_star_t(x=x_t, h_view_A=h_view_A_t, h_joint_A=h_joint_A_t, h_view_B=h_view_B_t, h_joint_B=h_joint_B_t, beta_t=beta[t], a_view_A=a_view_A, b_view_A=b_view_A, b_joint_A=b_joint_A, a_view_B=self.a_view, b_view_B=self.b_view, b_joint_B=self.b_joint, W_view_B=self.W_view, W_joint_B=self.W_joint, W_view2joint_B=self.W_view2joint)
                
                log_p_star_tplus1_xh_t=self.log_p_star_t(x=x_t, h_view_A=h_view_A_t, h_joint_A=h_joint_A_t, h_view_B=h_view_B_t, h_joint_B=h_joint_B_t, beta_t=beta[t+1], a_view_A=a_view_A, b_view_A=b_view_A, b_joint_A=b_joint_A, a_view_B=self.a_view, b_view_B=self.b_view, b_joint_B=self.b_joint, W_view_B=self.W_view, W_joint_B=self.W_joint, W_view2joint_B=self.W_view2joint)
                
                log_p_star_diff[t]=log_p_star_tplus1_xh_t-log_p_star_t_xh_t
            
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
        
    
    def sample_visible_MDBM(self, visible_types, visible_type_fixed_param=None, a_view=None, W_view=None, H_view=None, NS=1, rng=numpy.random.RandomState(100)):
        X=[None]*self.num_views
        XM=[None]*self.num_views
        XP=[None]*self.num_views
        for v in range(self.num_views):
            if H_view is not None:
                Xv,XMv,XPv=self.dbm_view[v].sample_visible(visible_type=visible_types[v], a=a_view[v], W=W_view[v], H=H_view[v], visible_type_fixed_param=visible_type_fixed_param[v], tie_W_for_pretraining_DBM_top=False, NS=1, rng=rng)
            else:
                Xv,XMv,XPv=self.dbm_view[v].sample_visible(visible_type=visible_types[v], a=a_view[v], W=None, H=None, visible_type_fixed_param=visible_type_fixed_param[v], tie_W_for_pretraining_DBM_top=False, NS=1, rng=rng)
            X[v]=Xv
            XM[v]=XMv
            XP[v]=XPv
        return X,XM,XP


    def sample_hidden_MDBM(self, X, H_view, H_joint, a_view, b_view, W_view, a_joint, b_joint, W_joint, W_view2joint, beta_t):
        # sample view-specific
        if H_view is None:
            H_view=[None]*self.num_views
        HP_view=[None]*self.num_views
        for v in range(self.num_views):
            if self.NK_view[v]==0: # do not sample the joint hidden layer, when there is no view-specific hidden layer
                continue
            if self.NK_view[v]%2==0: # odd number of layer
                even_or_odd_first="odd"
            if self.NK_view[v]%2==1: # odd number of layer
                even_or_odd_first="even"
            if X is not None:
                Hv,HPv=self.sample_hidden(visible_type=self.visible_types[v], visible_type_fixed_param=self.visible_type_fixed_param[v], NK=self.NK_view[v], NS=1, X=X[v], H=H_view[v], H_joint_0=H_joint[0], a=a_view[v], b=b_view[v], W=W_view[v], W_view2joint_v=W_view2joint[v], beta_t=beta_t, even_or_odd_first=even_or_odd_first, rng=self.rng)
            else:
                Hv,HPv=self.sample_hidden(visible_type=self.visible_types[v], visible_type_fixed_param=self.visible_type_fixed_param[v], NK=self.NK_view[v], NS=1, X=None, H=None, H_joint_0=None, a=a_view[v], b=b_view[v], W=None, W_view2joint_v=None, beta_t=beta_t, even_or_odd_first=even_or_odd_first, rng=self.rng)
            H_view[v]=Hv
            HP_view[v]=HPv
        
        # sample joint
        if X is not None:
            X0=self.get_X_joint(X,H_view)
            H_joint,HP_joint=self.dbm_joint.sample_hidden(visible_type="Bernoulli", visible_type_fixed_param=0, hidden_type="Bernoulli", hidden_type_fixed_param=0, NS=1, X=X0, H=H_joint, a=a_joint, b=b_joint, W=W_joint, beta_t=beta_t, even_or_odd_first="even", rng=self.rng)
        else:
            H_joint,HP_joint=self.dbm_joint.sample_hidden(visible_type="Bernoulli", visible_type_fixed_param=0, hidden_type="Bernoulli", hidden_type_fixed_param=0, NS=1, X=None, H=None, a=a_joint, b=b_joint, W=None, beta_t=beta_t, even_or_odd_first="even", rng=self.rng)
            
        # for trivial view, the view-specific hidden states are identifical to the first joint layer
        for v in range(self.num_views):
            if self.NK_view[v]==0:
                H_view[v]=[H_joint[0]]
                HP_view[v]=[HP_joint[0]]
        return H_view,HP_view,H_joint,HP_joint
        

    def compute_logZ_A(self, base_rate_type="prior"):
        """
        Compute the log-partition function of the prior or uniform base-rate model A.
        a_B, b_B: the biases of model B.
        """
        a_view_A=[None]*self.num_views
        b_view_A=[None]*self.num_views
        logZ_A=0
        
        for v in range(self.num_views):
            a_Av,b_Av,logZ_Av=self.dbm_view[v].compute_logZ_A(a_B=self.a_view[v], b_B=self.b_view[v], visible_type=self.visible_types[v], visible_type_fixed_param=self.visible_type_fixed_param[v], hidden_type="Bernoulli", hidden_type_fixed_param=None, base_rate_type="prior")
            a_view_A[v]=a_Av
            b_view_A[v]=b_Av
            logZ_A=logZ_A+logZ_Av

        a_joint_A,b_joint_A,logZ_Aj=self.dbm_joint.compute_logZ_A(a_B=self.a_joint, b_B=self.b_joint, visible_type="Bernoulli", visible_type_fixed_param=None, hidden_type="Bernoulli", hidden_type_fixed_param=None, base_rate_type="prior")
        logZ_A=logZ_A+logZ_Aj
        
        return a_view_A,b_view_A,a_joint_A,b_joint_A,logZ_A 
 
 
    def combine_a_A_a_B_hat(self, a_A, a_B_hat, beta_t, visible_types):
        a_hat_t=[None]*self.num_views
        for v in range(self.num_views):
            if visible_types[v]=="Bernoulli" or visible_types[v]=="Poisson" or visible_types[v]=="NegativeBinomial" or visible_types[v]=="Multinomial" or visible_types[v]=="Gaussian_FixPrecision1" or visible_types[v]=="Gaussian_FixPrecision2":
                #print "a_A:{}".format(v)
                #print a_A[v]
                #print "a_B_hat:{}".format(v)
                #print a_B_hat[v]
                a_hat_t[v]=(1-beta_t)*a_A[v] + beta_t*a_B_hat[v]
            elif visible_types[v]=="Gaussian" or visible_types[v]=="Gamma":
                a_hat_t_v=[None,None]*2
                a_hat_t_v[0]=(1-beta_t)*a_A[v][0] + beta_t*a_B_hat[v][0]
                a_hat_t_v[1]=(1-beta_t)*a_A[v][1] + beta_t*a_B_hat[v][1]
                a_hat_t[v]=a_hat_t_v
        return a_hat_t
      

    def log_p_star_t(self, x, h_view_A, h_joint_A, h_view_B, h_joint_B, beta_t, a_view_A, b_view_A, b_joint_A, a_view_B, b_view_B, b_joint_B, W_view_B, W_joint_B, W_view2joint_B):
        
        fta=self.compute_reduced_energy(X=x, H_view=h_view_A, H_joint=h_joint_A, a_view=a_view_A, b_view=b_view_A, b_joint=b_joint_A, W_view=None, W_joint=None, W_view2joint=None, beta_t=1-beta_t)
        ftb=self.compute_reduced_energy(X=x, H_view=h_view_B, H_joint=h_joint_B, a_view=a_view_B, b_view=b_view_B,  b_joint=b_joint_B, W_view=W_view_B, W_joint=W_joint_B, W_view2joint=W_view2joint_B, beta_t=beta_t)
        log_p_star_t_xh= -fta -ftb
        
        return log_p_star_t_xh
        
    
    def compute_reduced_energy(self,X=None, H_view=None, H_joint=None, a_view=None, b_view=None, W_view=None, b_joint=None, W_joint=None, W_view2joint=None, beta_t=1): 
        """
        Compute "free" energy E() with some layers summed out. For AIS.
        """
        if W_view is not None: # model B
            a_view_hat,b_view_hat,b_joint_hat=self.compute_posterior_bias(visible_types=self.visible_types, visible_type_fixed_param=self.visible_type_fixed_param, NK_view=self.NK_view, NK_joint=self.NK_joint, a_view=a_view, b_view=b_view, W_view=W_view, b_joint=b_joint, W_joint=W_joint, W_view2joint=W_view2joint, X=X, H_view=H_view, H_joint=H_joint, scale=beta_t)
        else:
            a_view_hat=a_view
            b_view_hat=b_view
            b_joint_hat=b_joint
        
        fes=0
        # view-specific
        for v in range(self.num_views):
            if self.NK_view[v]%2==1:# NK is odd, such as x, h1, h2, h3, h4, h5
                sumout="odd" # sum out h1, h3, h5; equlvalently, x, h2, h4 can be summed out
            if self.NK_view[v]%2==0: # NK is even, such as x, h1, h2, h3, h4
                sumout="even" # sum out x, h2, h4
            # zeta
            z_v=self.dbm_view[v].zeta(X=X[v], H=H_view[v], a=a_view[v], b=b_view[v], visible_type=self.visible_types[v], visible_type_fixed_param=self.visible_type_fixed_param[v], sumout=sumout)
            # A()
            # when v==0 and if_multimodal_RBM, compuate the logPar once only
            if (v>0 and self.if_multimodal_RBM) or (not self.if_multimodal_RBM and self.K_view is None):
                logPar_v=0
            else:
                logPar_v=self.dbm_view[v].A(a_hat=a_view_hat[v], b_hat=b_view_hat[v], visible_type=self.visible_types[v], visible_type_fixed_param=self.visible_type_fixed_param[v], hidden_type="Bernoulli", hidden_type_fixed_param=0, sumout=sumout )
                
            fes_v = -beta_t*z_v - logPar_v
            fes = fes + fes_v
            
        # joint
        if not self.if_multimodal_RBM:
            X_joint=self.get_X_joint(X, H_view)
            sumout="even"
            z_j=self.dbm_joint.zeta(X=X_joint, H=H_joint, a=None, b=b_joint, visible_type=self.dbm_joint.visible_type, visible_type_fixed_param=self.dbm_joint.visible_type_fixed_param, sumout=sumout)
            logPar_j=self.dbm_joint.A(a_hat=None, b_hat=b_joint_hat, visible_type=self.dbm_joint.visible_type, visible_type_fixed_param=self.dbm_joint.visible_type_fixed_param, hidden_type="Bernoulli", hidden_type_fixed_param=0, sumout=sumout )
            fes_j = -beta_t*z_j - logPar_j
            fes = fes + fes_j
        
        mfe=numpy.mean(fes) # average over N samples
        return mfe


    def compute_free_energy(self,X=None, H_view=None, H_joint=None): 
        """
        Compute "free" energy E() with some layers summed out. 
        """
                    
        if H_view is None:
            XR,XRM,XRP,H_view,HP_view,X_joint,H_joint,HP_joint=self.mean_field_approximate_inference(X,NMF=self.NMF,rand_init_H=False)
            if self.if_multimodal_RBM:
                H_view=HP_view
                H_joint=HP_joint
        
        a_view_hat,b_view_hat,b_joint_hat=self.compute_posterior_bias(visible_types=self.visible_types, visible_type_fixed_param=self.visible_type_fixed_param, NK_view=self.NK_view, NK_joint=self.NK_joint, a_view=self.a_view, b_view=self.b_view, W_view=self.W_view, b_joint=self.b_joint, W_joint=self.W_joint, W_view2joint=self.W_view2joint, X=X, H_view=H_view, H_joint=H_joint, scale=1)
        
        fes=0
        # view-specific
        for v in range(self.num_views):
            if self.NK_view[v]%2==1:# NK is odd, such as x, h1, h2, h3, h4, h5
                sumout="odd" # sum out h1, h3, h5; equlvalently, x, h2, h4 can be summed out
            if self.NK_view[v]%2==0: # NK is even, such as x, h1, h2, h3, h4
                sumout="even" # sum out x, h2, h4
            # zeta
            z_v=self.dbm_view[v].zeta(X=X[v], H=H_view[v], a=self.a_view[v], b=self.b_view[v], visible_type=self.visible_types[v], visible_type_fixed_param=self.visible_type_fixed_param[v], sumout=sumout)
            # A()
            if (v>0 and self.if_multimodal_RBM) or (not self.if_multimodal_RBM and self.K_view is None):
                logPar_v=0
            else:
                logPar_v=self.dbm_view[v].A(a_hat=a_view_hat[v], b_hat=b_view_hat[v], visible_type=self.visible_types[v], visible_type_fixed_param=self.visible_type_fixed_param[v], hidden_type="Bernoulli", hidden_type_fixed_param=0, sumout=sumout )
                
            fes_v = -z_v - logPar_v
            fes = fes + fes_v
            
        # joint
        if not self.if_multimodal_RBM:
            X_joint=self.get_X_joint(X, H_view)
            sumout="even"
            z_j=self.dbm_joint.zeta(X=X_joint, H=H_joint, a=None, b=self.b_joint, visible_type=self.dbm_joint.visible_type, visible_type_fixed_param=self.dbm_joint.visible_type_fixed_param, sumout=sumout)
            logPar_j=self.dbm_joint.A(a_hat=None, b_hat=b_joint_hat, visible_type=self.dbm_joint.visible_type, visible_type_fixed_param=self.dbm_joint.visible_type_fixed_param, hidden_type="Bernoulli", hidden_type_fixed_param=0, sumout=sumout )
            fes_j = -z_j - logPar_j
            fes = fes + fes_j
        
        mfe=numpy.mean(fes) # average over N samples
        return mfe,fes


    def compute_reconstruction_error(self, X0, X0RM):
        if X0RM is None:
            _,X0RM,_,_,_,_,_,_=self.mean_field_approximate_inference(X0,NMF=self.NMF,rand_init_H=False)
        rec_errors=[0]*self.num_views
        for v in range(self.num_views):
            rec_errors[v]=self.dbm_view[v].compute_reconstruction_error(X0[v], X0RM[v])
        return rec_errors


    def compute_posterior_bias(self, visible_types=["Bernoulli"], visible_type_fixed_param=[0], NK_view=None, NK_joint=None, a_view=None, b_view=None, W_view=None, b_joint=None, W_joint=None, W_view2joint=None, X=None, H_view=None, H_joint=None, scale=1, compute_a_only=False):
        """
        Compute the posterior bias given parameters and data.
        """
        num_views=len(X)
        #num_joint=len(b_joint)
        a_view_hat=[None]*num_views
        b_view_hat=[None]*num_views
        b_joint_hat=[None]*NK_joint
        
        for v in range(num_views):
            # visible
            if visible_types[v]=="Bernoulli" or visible_types[v]=="Poisson" or visible_types[v]=="NegativeBinomial" or visible_types[v]=="Multinomial" or visible_types[v]=="Gaussian_FixPrecision1" or visible_types[v]=="Gaussian_FixPrecision2":
                if H_view is not None and W_view is not None:
                    a_hat=a_view[v] + numpy.dot(W_view[v][0],H_view[v][0])
                else:
                    a_hat=a_view[v]
                a_view_hat[v]= scale*a_hat
            elif visible_types[v]=="Gaussian":
                if H_view is not None and W_view is not None:
                    a_hat1=a_view[v][0] + numpy.dot(W_view[v][0],H_view[v][0])
                else:
                    a_hat1=a_view[v][0]
                a_hat2=a_view[v][1]
                a_view_hat[v]=[scale*a_hat1, scale*a_hat2]
            elif visible_types[v]=="Gamma":
                a_hat1=a_view[v][0]
                if H_view is not None and W_view is not None:
                    a_hat2=a_view[v][1] + numpy.dot(W_view[v][0],H_view[v][0])
                else:
                    a_hat2=a_view[v][1]            
                a_view_hat[v]=[scale*a_hat1, scale*a_hat2]
            elif visible_types[v]=="Multinoulli":
                M=len(a_view[v])
                a_hat=[None]*M
                for m in range(M):
                    if H_view is not None and W_view is not None:
                        a_hat[m]=scale* (a_view[v][m] + numpy.dot(W_view[v][0][m],H_view[v][0]) )
                    else:
                        a_hat[m]=scale* a_view[v][m]
                a_view_hat[v]= a_hat
                
            if v==num_views-1 and compute_a_only:
                return a_view_hat,None,None
            
            # hidden        
            NK=NK_view[v]#len(b_view[v])
            b_hat=[None]*NK
            for nk in range(NK):
                if NK>1:
                    if nk==0:
                        if visible_types[v]=="Multinoulli":
                            if X[v] is not None and W_view is not None:
                                b_hat[nk]= scale*( b_view[v][nk] + numpy.dot(W_view[v][nk+1], H_view[v][nk+1] ) )
                                for m in range(M):
                                    b_hat[nk]= b_hat[nk] + scale*( numpy.dot(W_view[v][nk][m].transpose(), X[v][m] ) )
                            else: # H is None
                                b_hat[nk]= scale*b_view[v][nk]
                        elif visible_types[v]=="Gaussian_FixPrecision1":
                            if X[v] is not None and W_view is not None:
                                b_hat[nk]= scale*( b_view[v][nk] + numpy.dot(W_view[v][nk].transpose(), visible_type_fixed_param[v]*X[v] ) + numpy.dot(W_view[v][nk+1], H_view[v][nk+1] ) )
                            else:
                                b_hat[nk]= scale* b_view[v][nk]
                        else: # not Multinoulli, not Gaussian1
                            if X[v] is not None and W_view is not None:
                                b_hat[nk]= scale*( b_view[v][nk] + numpy.dot(W_view[v][nk].transpose(), X[v] ) + numpy.dot(W_view[v][nk+1], H_view[v][nk+1] ) )
                            else:
                                b_hat[nk]= scale* b_view[v][nk]
                    elif nk==NK-1:
                        if X[v] is not None and W_view is not None:
                            b_hat[nk]= scale*( b_view[v][nk] + numpy.dot(W_view[v][nk].transpose(), H_view[v][nk-1] ) + numpy.dot(W_view2joint[v], H_joint[0]) )
                        else:
                            b_hat[nk]= scale*b_view[v][nk]
                    else: # middle
                        if X[v] is not None and W_view is not None:
                            b_hat[nk]= scale*( b_view[v][nk] + numpy.dot(W_view[v][nk].transpose(), H_view[v][nk-1] ) + numpy.dot(W_view[v][nk+1], H_view[v][nk+1]) )
                        else:
                            b_hat[nk]= scale* b_view[v][nk]
                else: # NK=1 # multimodal RBM
                    if visible_types[v]=="Multinoulli":
                        b_hat[nk]= scale* b_view[v][nk]
                        if X[v] is not None and W_view is not None:
                            for m in range(M):
                                b_hat[nk]= b_hat[nk] + scale*( numpy.dot(W_view[v][nk][m].transpose(), X[v][m] ) + numpy.dot(W_view2joint[v], H_joint[0]) )
                    elif visible_types[v]=="Gaussian_FixPrecision1":
                        b_hat[nk]= scale* b_view[v][nk]
                        if X[v] is not None and W_view is not None:
                            b_hat[nk]= b_hat[nk] + scale*( numpy.dot(W_view[v][nk].transpose(), visible_type_fixed_param[v]*X[v] ) + numpy.dot(W_view2joint[v], H_joint[0]) )
                    else: # not Gaussian1, not Multinoulli
                        b_hat[nk]= scale* b_view[v][nk]
                        if X[v] is not None and W_view is not None:
                            b_hat[nk]= b_hat[nk] + scale*( numpy.dot(W_view[v][nk].transpose(), X[v] ) + numpy.dot(W_view2joint[v], H_joint[0]) )
            b_view_hat[v]=b_hat
            
            # get psuedo input of the joint component
            if NK_view[v]==0:
                X_joint_v=X[v]
            else:
                X_joint_v=H_view[v][-1]
            if v==0:
                X_joint=X_joint_v
            else:
                X_joint=numpy.vstack((X_joint,X_joint_v))
            
        # joint component
        b_hat=[None]*NK_joint
        NK=NK_joint
        for nk in range(NK):
            if NK>1:
                if nk==0:
                    if X_joint is not None and W_joint is not None:
                        b_hat[nk]= scale*( b_joint[nk] + numpy.dot(W_joint[nk].transpose(), X_joint ) + numpy.dot(W_joint[nk+1], H_joint[nk+1] ) )
                    else:
                        b_hat[nk]= scale* b_joint[nk]
                elif nk==NK-1:
                    if X_joint is not None and W_joint is not None:
                        b_hat[nk]= scale*( b_joint[nk] + numpy.dot(W_joint[nk].transpose(), H_joint[nk-1] ) )
                    else:
                        b_hat[nk]= scale*b_joint[nk]
                else: # middle
                    if X_joint is not None and W_joint is not None:
                        b_hat[nk]= scale*( b_joint[nk] + numpy.dot(W_joint[nk].transpose(), H_joint[nk-1] ) + numpy.dot(W_joint[nk+1], H_joint[nk+1]) )
                    else:
                        b_hat[nk]= scale* b_joint[nk]
            else: # NK=1
                b_hat[nk]= scale* b_joint[nk]
                if X_joint is not None and W_joint is not None:
                    b_hat[nk]= b_hat[nk] + scale*( numpy.dot(W_joint[nk].transpose(), X_joint ) )
            b_joint_hat=b_hat
            
        return a_view_hat,b_view_hat,b_joint_hat


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


    def sample_hidden(self, visible_type="Bernoulli", visible_type_fixed_param=1, NK=0, NS=100, X=None, H=None, H_joint_0=None, a=None, b=None, W=None, W_view2joint_v=None, beta_t=1, even_or_odd_first="even", rng=numpy.random.RandomState(100)):
        """
        Sample the hidden variables of a (view-specific or joint) subnetwork.
        If X is None, it will sample NS samples using marginals.
        H is a list of vectors/matrices.
        H_joint_0 is a vector/matrix, the values of the first joint layer. 
        W is a list of matrices.
        W_view2joint_v is a matrix, it is in fact the self.W_view2joint[v]
        """
        #NK=len(b) # NK is the number of hidden layers,for trivial view-specific DBM, NK=0
        # get indices for hidden layers
        if NK==0:
            return [H_joint_0],[H_joint_0]
            
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
                if visible_type=="Bernoulli" or visible_type=="Gaussian" or visible_type=="Poisson" or visible_type=="NegativeBinomial" or visible_type=="Multinomial" or visible_type=="Gamma" or visible_type=="Gaussian_FixPrecision1" or visible_type=="Gaussian_FixPrecision2":
                    if X is not None:
                        b_hat= b[nk] + numpy.dot(numpy.transpose(W[nk]),X) + numpy.dot(W[nk+1],H[nk+1])
                    else:
                        b_hat=numpy.repeat(b[nk], NS, axis=1)
                    b_hat=beta_t * b_hat
                    P=cl.sigmoid( b_hat )
                elif visible_type=="Multinoulli":
                    if X is not None:
                        b_hat=b[nk] + numpy.dot(W[nk+1],H[nk+1])
                        M=len(X)
                        for m in range(M):
                            b_hat=b_hat + numpy.dot(numpy.transpose(W[nk][m]),X[m])
                    else:
                        b_hat=numpy.repeat(b[nk], NS, axis=1)
                    b_hat=beta_t * b_hat
                    P=cl.sigmoid(b_hat)
                elif visible_type=="Gaussian_Hinton":
                    if X is not None:
                        b_hat=b[nk] + numpy.dot(numpy.transpose(W[nk]),numpy.diag(a[1]).dot(X)) + numpy.dot(W[nk+1],H[nk+1])
                    else:
                        b_hat=numpy.repeat(b[nk], NS, axis=1)
                    b_hat=beta_t * b_hat
                    P=cl.sigmoid( b_hat )
            elif nk==NK-1 and NK>1: # last but not the first hidden layer
                if X is not None:
                    b_hat=b[nk] +  numpy.dot(numpy.transpose(W[nk]),H[nk-1]) + numpy.dot(W_view2joint_v,H_joint_0) 
                else:
                    b_hat=numpy.repeat(b[nk], NS, axis=1)
                b_hat=beta_t * b_hat
                P=cl.sigmoid( b_hat )
            elif nk==0 and NK==1: # there is only one hidden layer in this DBM, actually it is a RBM!
                if visible_type=="Bernoulli" or visible_type=="Gaussian" or visible_type=="Poisson" or visible_type=="NegativeBinomial" or visible_type=="Multinomial" or visible_type=="Gamma" or visible_type=="Gaussian_FixPrecision1" or visible_type=="Gaussian_FixPrecision2":
                    if X is not None:
                        b_hat=b[nk] + numpy.dot(numpy.transpose(W[nk]),X) + numpy.dot(W_view2joint_v,H_joint_0)
                    else:
                        b_hat=numpy.repeat(b[nk], NS, axis=1)
                    b_hat=beta_t * b_hat
                    P=cl.sigmoid( b_hat )
                elif visible_type=="Multinoulli":
                    if X is not None:
                        b_hat=b[nk] + numpy.dot(W_view2joint_v,H_joint_0)
                        for m in range(M):
                            b_hat= b_hat + numpy.dot(numpy.transpose(W[nk][m]),X[m])
                    else:
                        b_hat=numpy.repeat(b[nk], NS, axis=1)    
                    b_hat=beta_t * b_hat
                    P=cl.sigmoid(b_hat)
                elif visible_type=="Gaussian_Hinton":
                    if X is not None:
                        b_hat=b[nk] + numpy.dot(numpy.transpose(W[nk]),numpy.diag(a[1]).dot(X)) + numpy.dot(W_view2joint_v,H_joint_0)
                    else:
                        b_hat=numpy.repeat(b[nk], NS, axis=1)
                    b_hat=beta_t * b_hat
                    P=cl.sigmoid( b_hat )
            else: # in the middle
                if X is not None:
                    b_hat=b[nk] + numpy.dot(numpy.transpose(W[nk]),H[nk-1]) + numpy.dot(W[nk+1],H[nk+1])
                else:
                    b_hat=numpy.repeat(b[nk], NS, axis=1)
                b_hat=beta_t * b_hat
                P=cl.sigmoid( b_hat )
            HP[nk]=P
            H[nk]=cl.Bernoulli_sampling(P=P,rng=rng)
                
        return H,HP


    def mean_field_approximate_inference(self, XbatchOrg=None, missing_view=None, NMF=20, rand_init_H=False, rand_init_missing_view=False,  only_update_view_spec_component_with_observed_data=False):
        """
        Mean-field method to approximate the data dependent expectation. This function can also be used for infere p(h|v).
        XbatchOrg: list of numpy matrices with each sample in a row, a batch of training samples. When the self.visibletypes[v] is "Multinoulli", XbatchOrg[v] is a list of matrices.
        NMF: number of iterations of the mean-field approximation.
        rand_init_H: bool, whether randomly initialize the hidden layers or use marginals from RBMs.
        only_update_view_spec_component_with_observed_data: whether only update view-specific components with observed data and ignore updating view-specific components with missing views and ignore updating the joint DBM; if False, initialize the missing views with random numbers, update the view-specific components with missing views and the joint component, and infer the missing views.
        """
        ####### initialization #####################
        Xbatch=copy.deepcopy(XbatchOrg)
        #self.NMF=NMF # number of iterations in mean-field approximation

        # when missing_view indicators are given, we just take it. If not given, we need to find it out from Xbatch 
        if missing_view is None:
            missing_view=[True]*self.num_views
            for v in range(self.num_views):
                if Xbatch[v] is not None:
                    missing_view[v]=False
                        
        # get batch_size
        for v in range(self.num_views):
            
            if missing_view[v]==False:
                if self.visible_types[v]!="Multinoulli":
                    batch_size=Xbatch[v].shape[1]
                else:
                    batch_size=Xbatch[v][0].shape[1]

        # all views observed, of course, update the whole model
        if numpy.sum(missing_view)==0:
            only_update_view_spec_component_with_observed_data=False
            
        #
        if numpy.sum(missing_view)>0 and not rand_init_missing_view:
            Xbatch_rand_sampled=self.sample_minibatch(batch_size=batch_size)

        # if wanting to infer using the whole model, initialize the missing views by (1) given initial values, (2) sampled values from the training data, or (3) randomly generated samples using marginals
        if not only_update_view_spec_component_with_observed_data:
            for v in range(self.num_views):
                if missing_view[v]:
                    if rand_init_missing_view:
                        XvR,XvRM,XvRP=self.dbm_view[v].sample_visible(visible_type=self.visible_types[v], a=self.a_view[v], W=None, H=None, visible_type_fixed_param=self.visible_type_fixed_param[v], tie_W_for_pretraining_DBM_top=False, NS=batch_size, rng=self.rng )
                        Xbatch[v]=XvR
                    else: # use given initial values or sample from the training data to initialize
                        if Xbatch[v] is None:
                            Xbatch[v]=Xbatch_rand_sampled[v]
                                       
        XbatchR=[None]*self.num_views # store the recovered value
        XbatchRM=[None]*self.num_views # store mean
        XbatchRP=[None]*self.num_views # store prob
        # initialize Hbatch_view
        Hbatch_view=[None]*self.num_views # Hbatch_view[v] is a list of matrices
        HbatchP_view=[None]*self.num_views 
        Xbatch_joint=[None]
        # joint
        Hbatch_joint=[None]*self.NK_joint
        HbatchP_joint=[None]*self.NK_joint
        
        ############### for multimodal RBM ##############################
        #print "in mean-field ..."
        #print missing_view
        if False and self.if_multimodal_RBM:
            for i in range(NMF):
                # sample the joint hidden states
                #b_hat=self.b_joint[0] # the bias of the joint layer
                #for v in range(self.num_views):
                #    _,b_hat_v=self.dbm_view[v].compute_posterior_bias(visible_type=self.visible_types[v], a=self.a_view[v], b=self.b_joint, W=[self.W_view2joint[v]], X=Xbatch[v], H=None, scale=1)
                #    b_hat=b_hat + b_hat_v[0] - self.b_joint[0] # remove additional self.b_joint[0] from b_hat_v 
                _,_,b_joint_hat=self.compute_posterior_bias(visible_types=self.visible_types, visible_type_fixed_param=self.visible_type_fixed_param, NK_view=self.NK_view, NK_joint=self.NK_joint, a_view=self.a_view, b_view=self.b_view, W_view=self.W_view, b_joint=self.b_joint, W_joint=self.W_joint, W_view2joint=self.W_view2joint, X=Xbatch, H_view=Hbatch_view, H_joint=Hbatch_joint, scale=1)
                
                H,HP=self.dbm_view[0].sample_hidden(visible_type=self.visible_types[0], NS=1, X=None, H=None, a=None, b=b_joint_hat, W=None, beta_t=1, even_or_odd_first="odd", rng=self.rng) # b_hat is already posterior, the visible_type does not matter as we only want to call this function
                print("In mean-field of MDBM..., HP:")
                print(Xbatch[0])
                print(Xbatch[1])
                print(HP)
                
                Hbatch_joint=H
                HbatchP_joint=HP
                Hbatch_view=[H]*self.num_views # the view-specific hidden layer has the same state as the joint hidden layer
                HbatchP_view=[HP]*self.num_views
            
                # sample the missing views
                for v in range(self.num_views):
                    if (i==NMF-1) or (missing_view[v] and not only_update_view_spec_component_with_observed_data):
                        X,XM,XP=self.dbm_view[v].sample_visible(visible_type=self.visible_types[v], a=self.a_view[v], W=self.W_view2joint[v], H=HbatchP_joint[0], visible_type_fixed_param=self.visible_type_fixed_param[v], tie_W_for_pretraining_DBM_top=False, NS=None, rng=self.rng)
                        if missing_view[v]:
                            #print "missing view {0} filled in".format(v)
                            Xbatch[v]=XM
                    
                        XbatchR[v]=X
                        XbatchRM[v]=XM
                        XbatchRP[v]=XP
                    
            # return results
            return XbatchR,XbatchRM,XbatchRP,Hbatch_view,HbatchP_view,Xbatch_joint,Hbatch_joint,HbatchP_joint
        
        ############ the model is a multimodal DBM #########################
        for v in range(self.num_views):
            # if there is no hidden layer, just skip
            if self.K_view[v] is None:
                if v==0:
                    Xbatch_joint=Xbatch[v]
                else:
                    Xbatch_joint=numpy.vstack( (Xbatch_joint,Xbatch[v]) )
                continue
            
            # at least one hidden layer
            if (not missing_view[v]) or (missing_view[v] and not only_update_view_spec_component_with_observed_data):
                # initialize Hbatch_view using marginals, this can also be realized by inferring H_view before the training, in order to save time
                _,_,_,Hbatch_v,HbatchP_v=self.dbm_view[v].mean_field_approximate_inference(Xbatch=Xbatch[v], NMF=1, rand_init_H=rand_init_H) # just initialize hidden states
                Hbatch_view[v]=Hbatch_v
                HbatchP_view[v]=HbatchP_v
                if not only_update_view_spec_component_with_observed_data:
                    if v==0:
                        Xbatch_joint=HbatchP_v[-1] # the input of the joint DBM
                    else:
                        Xbatch_joint=numpy.vstack( (Xbatch_joint,HbatchP_v[-1]) )
        
        ##### initialize joint DBM
        if not only_update_view_spec_component_with_observed_data:
            # initialize Hbatch_joint using marginals
            _,_,_,Hbatch_joint,HbatchP_joint=self.dbm_joint.mean_field_approximate_inference(Xbatch=Xbatch_joint,NMF=1,rand_init_H=rand_init_H)
            # for trival view, use the first joint hidden layer to fill in Hbatch_view[v]
            for v in range(self.num_views):
                if self.K_view[v] is None:
                    Hbatch_view[v]=[Hbatch_joint[0]]
                    HbatchP_view[v]=[HbatchP_joint[0]]
        
        ############ iterative alternating updates #########################
        for i in range(NMF):
            # for each view-specific DBM
            for v in range(self.num_views):
                # no hidden layer
                if self.K_view[v] is None:
                    if v==0:
                        Xbatch_joint=Xbatch[v]
                    else:
                        Xbatch_joint=numpy.vstack( (Xbatch_joint,Xbatch[v]) )
                    continue
                # at least one hidden layer
                else:
                    if (not missing_view[v]) or (missing_view[v] and not only_update_view_spec_component_with_observed_data):
                        a=self.a_view[v]
                        b=self.b_view[v]
                        W=self.W_view[v]
                        HbatchP_v=HbatchP_view[v] # Hbatch_v is a list of matrices, each for a hidden layer
                        Hbatch_v=[None]*self.NK_view[v]
                        
                        # if data of this view is missing and we want to to infer in the whole network, infer data of the missing views
                        # in the last iteration, reconstruct all the observed views, useful to compare with the original one
                        if missing_view[v] and not only_update_view_spec_component_with_observed_data:
                            # if missing view, update the X in this view too.
                            XvR,XvRM,XvRP=self.dbm_view[v].sample_visible(visible_type=self.visible_types[v], a=a, W=W[0], H=HbatchP_v[0], visible_type_fixed_param=self.visible_type_fixed_param[v], tie_W_for_pretraining_DBM_top=False, NS=None, rng=self.rng )
                            XbatchR[v]=XvR
                            Xbatch[v]=XvR # mean
                            #print "updated the missing view"
                        
                        # sample hidden states
                        #if self.K_view[v] is not None:# already not None 
                        if self.NK_view[v]%2==1: # even number of hidden layers
                            even_or_odd_first="even"
                        else: # odd number of hidden layers
                            even_or_odd_first="odd"
                        Hbatch_v,HbatchP_v=self.sample_hidden(visible_type=self.visible_types[v], visible_type_fixed_param=self.visible_type_fixed_param[v], NK=self.NK_view[v], NS=None, X=Xbatch[v], H=HbatchP_v, H_joint_0=HbatchP_joint[0], a=a, b=b, W=W, W_view2joint_v=self.W_view2joint[v], beta_t=1, even_or_odd_first=even_or_odd_first, rng=self.rng)
                        
                        # update the hidden layers of the v-th view
                        HbatchP_view[v]=HbatchP_v
                        Hbatch_view[v]=Hbatch_v

                    # last run, sample Xbatch
                    if i==NMF-1:
                        XvR,XvRM,XvRP=self.dbm_view[v].sample_visible(visible_type=self.visible_types[v], a=a, W=W[0], H=HbatchP_v[0], visible_type_fixed_param=self.visible_type_fixed_param[v], tie_W_for_pretraining_DBM_top=False, NS=None, rng=self.rng )
                        XbatchR[v]=XvRM
                        XbatchRM[v]=XvRM
                        XbatchRP[v]=XvRP
                        
                    if not only_update_view_spec_component_with_observed_data:
                        if v==0:
                            Xbatch_joint=HbatchP_v[-1] # the input of the joint DBM
                        else:
                            Xbatch_joint=numpy.vstack((Xbatch_joint,HbatchP_v[-1]))
            
            # joint DBM
            if not only_update_view_spec_component_with_observed_data:
                # run on the joint DBM
                Hbatch_joint,HbatchP_joint=self.dbm_joint.sample_hidden(visible_type="Bernoulli", NS=None, X=Xbatch_joint, H=HbatchP_joint, a=self.a_joint, b=self.b_joint, W=self.W_joint, beta_t=1, even_or_odd_first="even", rng=self.rng)
                
                # sample Xbatch for trivial views
                for v in range(self.num_views):
                    if self.K_view[v] is None:
                        if missing_view[v] or i==NMF-1:
                            a=self.a_view2joint[v]
                            b=self.b_joint[0]
                            W=self.W_view2joint[v]
                            XvR,XvRM,XvRP=self.dbm_view[v].sample_visible(visible_type=self.visible_types[v], a=a, W=W, H=HbatchP_joint[0], visible_type_fixed_param=self.visible_type_fixed_param[v], tie_W_for_pretraining_DBM_top=False, NS=None, rng=self.rng )
                            if missing_view[v]:
                                Xbatch[v]=XvRM # fill in missing view
                            XbatchR[v]=XvRM
                            XbatchRM[v]=XvRM
                            XbatchRP[v]=XvRP
                            #print "updated the missing view with no hidden layer"
                            
                # for trival view, use the first joint hidden layer to fill in Hbatch_view[v]
                for v in range(self.num_views):
                    if self.K_view[v] is None:
                        Hbatch_view[v]=[Hbatch_joint[0]]
                        HbatchP_view[v]=[HbatchP_joint[0]]
        # return means
        return XbatchR,XbatchRM,XbatchRP,Hbatch_view,HbatchP_view,Xbatch_joint,Hbatch_joint,HbatchP_joint


    def pcd_sampling(self, pcdk=20, NS=20, X0=None, H0_view=None, H0_joint=None, missing_view=None, clamp_observed_view=False, only_update_view_spec_component_with_observed_data=False, persistent=True, init_sampling=False, rand_init_X=False, rand_init_missing_view=False, rand_init_H=False): 
        """
        Persistent contrastive divergence sampling. This function can be used for learning and sampling after learning.
        INPUT: 
        pcdk: integer, steps of a Markov chain to generate a sample.
        NS: integer, number of Markov chains.
        rand_init_X: bool, whether randomly initialize the visible variables or sample some training samples as initial points.
        missing_view: list or None.
        init_sampling: bool, whether call this function to initialize the Markov chains.
        rand_init_H: bool, whether randomly initialize the hidden layers or use the marginals from RBMs.
        OUTPUT: 
        self.chainX: list of numpy arrays, the final states of the visible samples of the Markov chains; chainX[nk] of size M by NS: the final states of visible samples.
        self.chainH: list of numpy arrays, the final states of the latent samples of the Markov chains; chainH[nk] of size K[nk] by NS: the final states of the nk-th hidden layer.
        self.chain_length: the length of Markov chains.
        """

        if not persistent:
            if X0 is not None and pcdk>0: # X0 should be the current mini-batch
                init_sampling=True
            else:
                print("Error! You want to use CD-k sampling, but you either did not give me a batch of training samples (the same samples as in the mean-field approximation) or mistakenly set pck=0.")
                exit()

        if init_sampling:
            # initialize Markov chains
            #self.pcdk=pcdk
            self.NS=NS
            self.clamp_observed_view=clamp_observed_view
            self.only_update_view_spec_component_with_observed_data=only_update_view_spec_component_with_observed_data
            # randomly initialize H0 from Bernoulli distributions
            #self.chainX=self.rng.binomial(n=1,p=0.5,size=(self.M,self.NS)) # randomly initialize data
            # random initialize X
            self.chainX=[None]*self.num_views
            self.chainXM=[None]*self.num_views
            self.chainXP=[None]*self.num_views
            if rand_init_X: # is randomly initialize it, the X0 is not used anyway
                for v in range(self.num_views):
                    #P=0.5*numpy.ones(shape=(self.M[v],NS))
                    XvR,XvRM,XvRP=self.dbm_view[v].sample_visible(visible_type=self.visible_types[v], a=self.a_view[v], W=None, H=None, visible_type_fixed_param=self.visible_type_fixed_param[v], tie_W_for_pretraining_DBM_top=False, NS=self.NS, rng=self.rng )
                    self.chainX[v]=XvR # value
                    self.chainXM[v]=XvRM
                    self.chainXP[v]=XvRP # mean
                self.missing_view=self.missing_view_train
            else: # use training samples to initialize X
                #P=0.5*numpy.ones(shape=(self.M,NS))
                #P=self.rng.random_sample(size=(self.M[v],NS)) # this is not used anyway
                if X0 is None: # the initial sample is not given
                    self.chainX=copy.deepcopy(self.sample_minibatch(self.NS))
                else:
                    self.chainX=copy.deepcopy(X0)
                    
                if not rand_init_missing_view:
                    Xbatch_rand_sampled=copy.deepcopy(self.sample_minibatch(batch_size=self.NS))

                # check if there are missing values
                if missing_view is None:
                    self.missing_view=[True]*self.num_views 
                    for v in range(self.num_views):
                        if self.chainX[v] is None:
                            self.missing_view[v]=True
                else:
                    self.missing_view=missing_view
                    
                for v in range(self.num_views):
                    # if missing view, random initialize it
                    XbatchR,XbatchRM,XbatchRP=self.dbm_view[v].sample_visible(visible_type=self.visible_types[v], a=self.a_view[v], W=None, H=None, visible_type_fixed_param=self.visible_type_fixed_param[v], tie_W_for_pretraining_DBM_top=False, NS=self.NS, rng=self.rng )
                    if self.missing_view[v]==False:
                        # do not update chainX[v] because it is already assigned
                        self.chainXM[v]=XbatchRM # just initialize
                        self.chainXP[v]=XbatchRP # just initialize
                    # if missing view, initialize it by (1) randomly sampled from marginal, (2) sampled from training data, or (3) given initial values
                    else:
                        if not only_update_view_spec_component_with_observed_data:
                            if rand_init_missing_view: # random initial values
                                self.chainX[v]=XbatchR # initialize chainX[v] because it is missing
                            else:
                                if self.chainX[v] is None: # sampled from training data
                                    self.chainX[v]=Xbatch_rand_sampled[v]
                            self.chainXM[v]=XbatchRM
                            self.chainXP[v]=XbatchRP

                #self.chainXP=[0]*self.num_views # not used anyway
                #self.chainXM=[0]*self.num_views # not used anyway

            self.chainH_view=[None]*self.num_views # a list of lists of matrices: [[matrix_layer1, matrix_layer2, matrix_layer3],...]
            self.chainHP_view=[None]*self.num_views # corresponding probabilities
            self.chainX_joint=None
            self.chainXP_joint=None
            self.chainH_joint=[None]*self.NK_joint # a list of matrices
            self.chainHP_joint=[None]*self.NK_joint # corresonding probabilities
            # initialize chainH_view
            for v in range(self.num_views):
                if (not self.missing_view[v]) or (self.missing_view[v] and not only_update_view_spec_component_with_observed_data):
                    if self.K_view[v] is not None:
                        _,H0,_,_,P0,_=self.dbm_view[v].pcd_sampling(pcdk=0, NS=NS, X0=self.chainX[v], persistent=persistent, init_sampling=True, rand_init_X=False, rand_init_H=rand_init_H)
                    else:
                        _,H0,_,_,P0,_=self.dbm_view[v].pcd_sampling(pcdk=0, NS=NS, X0=self.chainX[v], persistent=persistent, init_sampling=True, rand_init_X=False, rand_init_H=True)
                    if H0_view is not None:
                        self.chainH_view[v]=H0_view[v]
                        H0=H0_view[v]
                    else:
                        self.chainH_view[v]=H0
                    self.chainHP_view[v]=P0

                    if not only_update_view_spec_component_with_observed_data:
                        if v==0:
                            if self.K_view[v] is None:
                                X0_joint=self.chainX[v] # the input of the joint DBM
                            else:
                                X0_joint=H0[-1] # the input of the joint DBM
                        else:
                            if self.K_view[v] is None:
                                X0_joint=numpy.vstack( (X0_joint,self.chainX[v]) )
                            else:
                                X0_joint=numpy.vstack( (X0_joint,H0[-1]) )

            #if not only_update_view_spec_component_with_observed_data and not self.if_multimodal_RBM:
            if not only_update_view_spec_component_with_observed_data:
                # initialize chainH_joint
                _,chainH_joint,_,_,chainHP_joint,_=self.dbm_joint.pcd_sampling(pcdk=0, NS=NS, X0=X0_joint, persistent=persistent, init_sampling=True, rand_init_X=False, rand_init_H=rand_init_H)
                if H0_joint is not None:
                    self.chainH_joint=H0_joint
                else:
                    self.chainH_joint=chainH_joint
                self.chainHP_joint=chainHP_joint
                # for trival view, use the first joint hidden layer to fill in chainH_view[v]
                for v in range(self.num_views):
                    if self.K_view[v] is None:
                        self.chainH_view[v]=[self.chainH_joint[0]]
                        self.chainHP_view[v]=[self.chainHP_joint[0]]
                
            # initialize the length of Markov chains
            self.chain_length=0            
            
        ############ multimodal RBM ################
        #print "in pcd ..."
        #print self.missing_view
        if False and self.if_multimodal_RBM:
            for c in range(pcdk):
                # sample the joint hidden states
                #b_hat=self.b_joint[0] # the bias of the joint layer
                #for v in range(self.num_views):
                #    _,b_hat_v=self.dbm_view[v].compute_posterior_bias(visible_type=self.visible_types[v], a=self.a_view[v], b=self.b_joint, W=[self.W_view2joint[v]], X=self.chainX[v], H=None, scale=1)
                #    b_hat=b_hat + b_hat_v[0] - self.b_joint[0] # remove additional self.b_joint[0] from b_hat_v 
                _,_,b_joint_hat=self.compute_posterior_bias(visible_types=self.visible_types, visible_type_fixed_param=self.visible_type_fixed_param, NK_view=self.NK_view, NK_joint=self.NK_joint, a_view=self.a_view, b_view=self.b_view, W_view=self.W_view, b_joint=self.b_joint, W_joint=self.W_joint, W_view2joint=self.W_view2joint, X=self.chainX, H_view=self.chainH_view, H_joint=self.chainH_joint, scale=1)
                
                H,HP=self.dbm_view[v].sample_hidden(visible_type=self.visible_types[v], NS=1, X=None, H=None, a=None, b=b_joint_hat, W=None, beta_t=1, even_or_odd_first="odd", rng=self.rng) # b_hat is already posterior, the visible_type does not matter as we only want to call this function
                self.chainH_joint=H
                self.chainHP_joint=HP
                self.chainH_view=[H]*self.num_views
                self.chainHP_view=[HP]*self.num_views
                
                #if (c==pcdk-1) or (not self.missing_view[v] and not self.clamp_observed_view) or (self.missing_view[v] and not self.only_update_view_spec_component_with_observed_data):
                    # sample the missing views
                for v in range(self.num_views):
                    if (c==pcdk-1) or self.missing_view[v] or (not self.missing_view[v] and not self.clamp_observed_view):
                        X,XM,XP=self.dbm_view[v].sample_visible(visible_type=self.visible_types[v], a=self.a_view[v], W=self.W_view2joint[v], H=self.chainH_joint[0], visible_type_fixed_param=self.visible_type_fixed_param[v], tie_W_for_pretraining_DBM_top=False, NS=None, rng=self.rng)
                        if self.missing_view[v] or (not self.missing_view[v] and not self.clamp_observed_view): # if this view is missing, update it in each iteration
                            self.chainX[v]=X
                            #print "missing view {0} filled in".format(v)
                    
                        self.chainXM[v]=XM
                        self.chainXP[v]=XP
                    
                self.chain_length=self.chain_length+1
            
            return self.chainX,self.chainH_view,self.chainX_joint,self.chainH_joint,self.chainXM,self.chainXP,self.chainHP_view,self.chainXP_joint,self.chainHP_joint,self.chain_length
                    

        ############ multimodal DBM ################
        # start sampling
        for c in range(pcdk): # for each step

            # sample from each view-specific DBM first
            for v in range(self.num_views):
                if (not self.missing_view[v]) or (self.missing_view[v] and not self.only_update_view_spec_component_with_observed_data):
                    
                    # sample hidden
                    if self.NK_view[v]%2==1: # even number of hidden layers
                        even_or_odd_first="even"
                    else: # odd number of hidden layers
                        even_or_odd_first="odd"
                    #print "v={0}".format(v)
                    if self.K_view[v] is not None:
                        chainH_view_v,chainHP_view_v=self.sample_hidden(visible_type=self.visible_types[v], visible_type_fixed_param=self.visible_type_fixed_param[v], NK=self.NK_view[v], NS=None, X=self.chainX[v], H=self.chainH_view[v], H_joint_0=self.chainH_joint[0], a=self.a_view[v], b=self.b_view[v], W=self.W_view[v], W_view2joint_v=self.W_view2joint[v], beta_t=1, even_or_odd_first=even_or_odd_first, rng=self.rng)
                        self.chainH_view[v]=chainH_view_v
                        self.chainHP_view[v]=chainHP_view_v
                    
                    # sample visible
                    # in the last iteration, always sample the observed views to compare with the original
                    if (c==pcdk-1) or (not self.missing_view[v] and not self.clamp_observed_view) or (self.missing_view[v] and not self.only_update_view_spec_component_with_observed_data):
                        if self.K_view[v] is not None: # there are at least one hidden layers
                            chainXv,chainXMv,chainXPv=self.dbm_view[v].sample_visible(visible_type=self.visible_types[v], a=self.a_view[v], W=self.W_view[v][0], H=self.chainH_view[v][0], visible_type_fixed_param=self.visible_type_fixed_param[v], tie_W_for_pretraining_DBM_top=False, NS=None, rng=self.rng )
                            self.chainX[v]=chainXv
                            self.chainXM[v]=chainXMv
                            self.chainXP[v]=chainXPv
                        else: # no hidden layers
                            chainXv,chainXMv,chainXPv =  self.dbm_view[v].sample_visible(visible_type=self.visible_types[v], a=self.a_view[v], W=self.W_view2joint[v], H=self.chainH_joint[0], visible_type_fixed_param=self.visible_type_fixed_param[v], tie_W_for_pretraining_DBM_top=False, NS=None, rng=self.rng )
                            self.chainX[v]=chainXv
                            self.chainXM[v]=chainXMv
                            self.chainXP[v]=chainXPv
                
                # get the input of the joint component
                if not self.only_update_view_spec_component_with_observed_data:
                    if v==0:
                        if self.K_view[v] is None:
                            self.chainX_joint=self.chainX[v] # the visible layer of the joint DBM
                            self.chainXP_joint=self.chainXP[v]
                        else:
                            self.chainX_joint=self.chainH_view[v][-1] # the visible layer of the joint DBM
                            self.chainXP_joint=self.chainHP_view[v][-1]
                    else:
                        if self.K_view[v] is None:
                            self.chainX_joint=numpy.vstack( (self.chainX_joint,self.chainX[v]) )
                            self.chainXP_joint=numpy.vstack( (self.chainXP_joint,self.chainXP[v]) )
                        else:
                            self.chainX_joint=numpy.vstack( (self.chainX_joint,self.chainH_view[v][-1]) )
                            self.chainXP_joint=numpy.vstack( (self.chainXP_joint,self.chainHP_view[v][-1]) )

            if not self.only_update_view_spec_component_with_observed_data:
                # sample from joint DBM
                chainH_joint,chainHP_joint=self.dbm_joint.sample_hidden(visible_type="Bernoulli", NS=None, X=self.chainX_joint, H=self.chainH_joint, a=self.a_joint, b=self.b_joint, W=self.W_joint, beta_t=1, even_or_odd_first="even", rng=self.rng)
                self.chainH_joint=chainH_joint
                self.chainHP_joint=chainHP_joint
                # for trival view, use the first joint hidden layer to fill in chainH_view[v]
                for v in range(self.num_views):
                    if self.K_view[v] is None:
                        self.chainH_view[v]=[self.chainH_joint[0]]
                        self.chainHP_view[v]=[self.chainHP_joint[0]]

            self.chain_length=self.chain_length+1
        return self.chainX,self.chainH_view,self.chainX_joint,self.chainH_joint,self.chainXM,self.chainXP,self.chainHP_view,self.chainXP_joint,self.chainHP_joint,self.chain_length


    def sample_xh_given_x(self, X, method="mean_field", num_iter=20, init_chain_time=100, save_prob=True, dir_save="./", prefix="MDBM"):
        """
        sample X and H given a complete X without missing views.
        INPUTS:
        X: list of numpy arrays; the observed multi-view data.
        method: string, method to sample/infer H; can be one from {"mean_field","Gibbs_sampling"}.
        """
        for v in range(self.num_views):
            if self.visible_types[v]=="Multinoulli": # convert to binary, may cause problems when there is only one sample in X
                X_binary=[None]*self.M[v]
                for m in range(self.M[v]):
                    Z,_=cl.membership_vector_to_indicator_matrix(X[v][m,:],z_unique=list(range(self.visible_type_fixed_param[v][m])))
                    X_binary[m]=Z.transpose()
                    X[v]=X_binary
                num_samples=X[0][0].shape[1]
            else:
                num_samples=X[0].shape[1]
                
        if method=="mean_field":
            X_view,XM_view,XP_view,H_view,HP_view,XM_joint,H_joint,HP_joint=self.mean_field_approximate_inference(XbatchOrg=X, NMF=num_iter, rand_init_H=False, only_update_view_spec_component_with_observed_data=False)
        elif method=="Gibbs_sampling":
            _,_,_,_,_,_,_,_,_,_=self.pcd_sampling(pcdk=init_chain_time*num_iter, NS=num_samples, X0=X, clamp_observed_view=True, only_update_view_spec_component_with_observed_data=False, persistent=True, init_sampling=True, rand_init_X=False, rand_init_H=False)
            X_view,H_view,X_joint,H_joint,XM_view,XP_view,HP_view,XP_joint,HP_joint,chain_length=self.pcd_sampling(pcdk=num_iter, clamp_observed_view=True,init_sampling=False)
            
        if save_prob:
            for v in range(self.num_views):
                filename=dir_save+prefix+"_sampling_xh_given_x_infer_prob_view_"+str(v)+".txt"
                numpy.savetxt(filename,XM_view[v],fmt="%.5f",delimiter="\t")
                
        return X_view,H_view,H_joint,XM_view,XP_view,HP_view,HP_joint


    def select_features(self, method="mean_difference", if_save=True, dir_save="./", prefix="MDBM"):
        """
        Feature selection based on probabilities of input nodes. Now, only applicable for two views of the same feature set.
        """
        print("Selecting features ...")
        if method=="mean_difference":
            # generate mean and probs
            _,_,_,XM_view,XP_view,_,_=self.sample_xh_given_x(X=self.X, method="mean_field", num_iter=20, save_prob=True, dir_save=dir_save, prefix=prefix)
            # mean difference
            self.feature_freqs0=numpy.mean(self.X[0],axis=1)
            self.feature_freqs1=numpy.mean(self.X[1],axis=1)
            self.feature_freqdif=self.feature_freqs1-self.feature_freqs0
            self.feature_probs0=numpy.mean(XM_view[0],axis=1)
            self.feature_probs1=numpy.mean(XM_view[1],axis=1)
            self.feature_probdif=self.feature_probs1-self.feature_probs0
            #self.feature_scores=self.feature_probs1*self.feature_probdif
            self.feature_scores=self.feature_probs1
            
            # sort scores
            ind=numpy.argsort(self.feature_scores,kind="mergesort")
            ind=ind[::-1]            
            
            numbers=numpy.array(range(len(self.features[0])))            
            inds_features_scores=numpy.vstack((numpy.array(numbers,dtype=str), self.features[0], numpy.array(self.feature_scores,dtype=str), self.feature_probs1, self.feature_probs0, self.feature_probdif, self.feature_freqs1, self.feature_freqs0, self.feature_freqdif))            

            inds_features_scores=inds_features_scores.transpose()

            inds_features_scores_sorted=inds_features_scores[ind,:]

        
        if if_save:
            for v in range(self.num_views):
                filename=dir_save+prefix+"_infer_XP_view_"+str(v)+".txt"
                numpy.savetxt(filename,XP_view[v],fmt="%.5f",delimiter="\t")            
                filename=dir_save+prefix+"_infer_XM_view_"+str(v)+".txt"
                numpy.savetxt(filename,XM_view[v],fmt="%.5f",delimiter="\t") 
                
                # save the probabilities
                filename=dir_save+prefix+"_conditional_probabilities_"+str(v)+".txt"
                numpy.savetxt(fname=filename, X=XP_view[v], fmt="%.5f", delimiter="\t")
                filename=dir_save+prefix+"_conditional_probabilities_sorted_"+str(v)+".txt"
                numpy.savetxt(fname=filename, X=XP_view[v][ind,:], fmt="%.5f", delimiter="\t")

                # save the 0s and 1s
                filename=dir_save+prefix+"_original_values_"+str(v)+".txt"
                numpy.savetxt(fname=filename, X=self.X[v], fmt="%s", delimiter="\t")
                filename=dir_save+prefix+"_original_values_X_sorted_"+str(v)+".txt"
                numpy.savetxt(fname=filename, X=self.X[v][ind,:], fmt="%s", delimiter="\t")
            
            # save the scores
            # unsorted scores
            filename=dir_save+prefix+"_feature_scores.txt"
            numpy.savetxt(fname=filename, X=inds_features_scores, fmt="%s", delimiter="\t")
            # sorted scores
            filename=dir_save+prefix+"_feature_scores_sorted.txt"
            numpy.savetxt(fname=filename, X=inds_features_scores_sorted, fmt="%s", delimiter="\t")
        print("Finished selecting features.")
        return inds_features_scores_sorted


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
        
    
    def predict_based_on_frequency(self,XY_train,XY_test):
        pass
        
    def sample_missing_views(self, X=None, missing_view=None, method="mean_field", num_iter=20, init_chain_time=100, rand_init_missing_view=False,  save_prob=True, dir_save="./", prefix="MDBM"):
        """
        Sample the missing views given observed views.
        INPUTS:
        X: list of numpy arrays, the observed data.
        method: string, method to sample/infer the missing values, can be one from {"mean_field","Gibbs_sampling"}.
        """
        start_time = time.clock()
        # which views are missing
        if missing_view is None:
            missing_view=[True]*self.num_views
            for v in range(self.num_views):
                if X[v] is not None:
                    missing_view[v]=False
        
        # process multinoulli data            
        for v in range(self.num_views):
            if missing_view[v]==False:
                if self.visible_types[v]=="Multinoulli":
                    X_binary=[None]*self.M[v]
                    for m in range(self.M[v]):
                        Z,_=cl.membership_vector_to_indicator_matrix(X[v][m,:],z_unique=list(range(self.visible_type_fixed_param[v][m])))
                        X_binary[m]=Z.transpose()
                        X[v]=X_binary
                    NS=X[v][0].shape[1]
                else:
                    NS=X[v].shape[1]
            else:
                print("View {0} is missing, to be inferred ...".format(v))
            
        if method=="mean_field":
            print(missing_view)
            X_view,XM_view,XP_view,H_view,HP_view,XP_joint,H_joint,HP_joint=self.mean_field_approximate_inference(XbatchOrg=X, missing_view=missing_view, NMF=num_iter, rand_init_missing_view=rand_init_missing_view, rand_init_H=False, only_update_view_spec_component_with_observed_data=False)
        elif method=="Gibbs_sampling":
            print(missing_view)
            _,_,_,_,_,_,_,_,_,_=self.pcd_sampling(pcdk=init_chain_time*num_iter, NS=NS, X0=X, missing_view=missing_view, clamp_observed_view=True, only_update_view_spec_component_with_observed_data=False, persistent=True, init_sampling=True, rand_init_X=False, rand_init_missing_view=rand_init_missing_view, rand_init_H=False)
            X_view,chainH_view,chainX_joint,chainH_joint,XM_view,XP_view,chainHP_view,chainXP_joint,chainHP_joint,chain_length=self.pcd_sampling(pcdk=num_iter, clamp_observed_view=True, init_sampling=False)
        
        if save_prob:
            for v in range(self.num_views):
                if self.visible_types[v]!="Multinoulli":
                    filename=dir_save+prefix+"_sampling_missing_views_infer_prob_view_"+str(v)+".txt"
                    numpy.savetxt(filename,XM_view[v],fmt="%.5f",delimiter="\t")
                else:
                    for m in range(self.M[v]):
                        filename=dir_save+prefix+"_sampling_missing_views_infer_prob_view_"+str(v)+"_var_"+str(m)+".txt"
                        numpy.savetxt(filename,XM_view[v][m],fmt="%.5f",delimiter="\t")
                        
        end_time = time.clock()
        self.sample_missing_view_time= end_time - start_time
        return X_view,XM_view,XP_view,self.sample_missing_view_time


    def compute_gradient(self,Xbatch,Hbatch_view,Xbatch_joint,Hbatch_joint,XS,HS_view,XS_joint,HS_joint):
        """
        Compute gradient.
        """
        # view-specific
        grad_a_view=[0]*self.num_views
        grad_b_view=[0]*self.num_views
        grad_W_view=[0]*self.num_views
        
        if False and self.if_multimodal_RBM:
            for v in range(self.num_views):
                if self.visible_types[v]=="Bernoulli" or self.visible_types[v]=="Poisson" or self.visible_types[v]=="NegativeBinomial" or self.visible_types[v]=="Multinomial" or self.visible_types[v]=="Gaussian_FixPrecision2":
                    # gradient of a: data_dep - data_indep
                    data_dep=-numpy.mean(Xbatch[v],axis=1)
                    data_indep=-numpy.mean(XS[v],axis=1)
                    grad_a_view[v]=data_dep - data_indep
                    grad_a_view[v].shape=(self.M[v],1)
    
                    # gradient of b
                    grad_b=[]
                    data_dep=-numpy.mean(Hbatch_view[v][0],axis=1)
                    data_indep=-numpy.mean(HS_view[v][0],axis=1)
                    grad_b_0=data_dep - data_indep
                    grad_b_0.shape=(self.K_joint[0],1)
                    grad_b.append(grad_b_0)
                    grad_b_view[v]=grad_b
    
                    # gradient of W
                    grad_W=[]
                    data_dep=-numpy.dot(Xbatch[v],Hbatch_view[v][0].transpose())/self.batch_size
                    data_indep=-numpy.dot(XS[v],HS_view[v][0].transpose())/self.NS                
                    grad_W_0=data_dep - data_indep
                    grad_W.append(grad_W_0) # gradient of the negative log-likelihood
                    grad_W_view[v]=grad_W

                elif self.visible_types[v]=="Gaussian_FixPrecision1":
                    # gradient of a: data_dep - data_indep
                    data_dep=-numpy.mean(self.visible_type_fixed_param[v]*Xbatch[v],axis=1)
                    data_indep=-numpy.mean(self.visible_type_fixed_param[v]*XS[v],axis=1)
                    grad_a_view[v]=data_dep - data_indep
                    grad_a_view[v].shape=(self.M[v],1)
    
                    # gradient of b
                    grad_b=[]
                    data_dep=-numpy.mean(Hbatch_view[v][0],axis=1)
                    data_indep=-numpy.mean(HS_view[v][0],axis=1)
                    grad_b_0=data_dep - data_indep
                    grad_b_0.shape=(self.K_joint[0],1)
                    grad_b.append(grad_b_0)
                    grad_b_view[v]=grad_b
    
                    # gradient of W
                    grad_W=[]
                    data_dep=-numpy.dot(self.visible_type_fixed_param[v]*Xbatch[v],Hbatch_view[v][0].transpose())/self.batch_size
                    data_indep=-numpy.dot(self.visible_type_fixed_param[v]*XS[v],HS_view[v][0].transpose())/self.NS                
                    grad_W_0=data_dep - data_indep
                    grad_W.append(grad_W_0) # gradient of the negative log-likelihood
                    grad_W_view[v]=grad_W                
                
                elif self.visible_types[v]=="Gaussian":
                    # gradient of a1, a2: data_dep - data_indep
                    data_dep=-numpy.mean(Xbatch[v],axis=1)
                    data_indep=-numpy.mean(XS[v],axis=1)
                    grad_a1_view_v=data_dep - data_indep
                    grad_a1_view_v.shape=(self.M[v],1)
    
                    data_dep=-numpy.mean(Xbatch[v]**2,axis=1)
                    data_indep=-numpy.mean(XS[v]**2,axis=1)
                    grad_a2_view_v=data_dep - data_indep
                    grad_a2_view_v.shape=(self.M[v],1)
                    
                    grad_a_view=[grad_a1_view_v,grad_a2_view_v]
                    
                    # gradient of b
                    grad_b=[]
                    data_dep=-numpy.mean(Hbatch_view[v][0],axis=1)
                    data_indep=-numpy.mean(HS_view[v][0],axis=1)
                    grad_b_0=data_dep - data_indep
                    grad_b_0.shape=(self.K_joint[0],1)
                    grad_b.append(grad_b_0)
                    grad_b_view[v]=grad_b
    
                    # gradient of W
                    grad_W=[]                    
                    data_dep=-numpy.dot(Xbatch[v],Hbatch_view[v][0].transpose())/self.batch_size
                    data_indep=-numpy.dot(XS[v],HS_view[v][0].transpose())/self.NS                
                    grad_W_0=data_dep - data_indep
                    grad_W.append(grad_W_0) # gradient of the negative log-likelihood
                    grad_W_view[v]=grad_W
                
                elif self.visible_types[v]=="Gaussian_Hinton":
                    # gradient of a1, a2: data_dep - data_indep
                    data_dep_x=-numpy.mean(Xbatch[v],axis=1)
                    data_indep_x=-numpy.mean(XS[v],axis=1)
                    grad_a1=self.a_view[v][1] * (data_dep_x - data_indep_x)
                    grad_a1.shape=(self.M[v],1)
                    
                    data_dep_a2=numpy.mean((Xbatch[v]-self.a_view[v][0])**2,axis=1) - numpy.mean(1/self.a_view[v][1]*Xbatch[v]*self.W_view[v][0].dot(Hbatch_view[v][0]),axis=1)
                    data_indep_a2=numpy.mean((XS[v]-self.a_view[v][0])**2,axis=1) - numpy.mean(1/self.a_view[v][1]*XS[v]*self.W_view[v][0].dot(HS_view[v][0]),axis=1)
                    grad_a2=data_dep_a2 - data_indep_a2
                    grad_a2.shape=(self.M,1)
                
                    grad_a_view=[grad_a1_view_v,grad_a2_view_v]
                    
                    # gradient of b
                    grad_b=[]
                    data_dep=-numpy.mean(Hbatch_view[v][0],axis=1)
                    data_indep=-numpy.mean(HS_view[v][0],axis=1)
                    grad_b_0=data_dep - data_indep
                    grad_b_0.shape=(self.K_joint[0],1)
                    grad_b.append(grad_b_0)
                    grad_b_view[v]=grad_b
    
                    # gradient of W
                    grad_W=[]
                    data_dep=-numpy.dot(numpy.sqrt(self.a_view[v][1])*Xbatch[v],Hbatch_view[v][0].transpose())/self.batch_size
                    data_indep=-numpy.dot(numpy.sqrt(self.a_view[v][1])*XS[v],HS_view[v][0].transpose())/self.NS
                    grad_W_0=data_dep - data_indep
                    grad_W.append(grad_W_0) # gradient of the negative log-likelihood
                    grad_W_view[v]=grad_W
                    
                elif self.visible_types[v]=="Multinoulli":
                    # gradient of a: data_dep - data_indep
                    grad_a_v=[None]*self.M[v]
                    for m in range(self.M[v]):
                        data_dep=-numpy.mean(Xbatch[v][m],axis=1)
                        data_indep=-numpy.mean(XS[v][m],axis=1)
                        grad_a_v[m]=data_dep - data_indep
                        grad_a_v[m].shape=(self.visible_type_fixed_param[v][m],1)
                    grad_a_view[v]=grad_a_v
    
                    # gradient of b
                    grad_b=[]
                    data_dep=-numpy.mean(Hbatch_view[v][0],axis=1)
                    data_indep=-numpy.mean(HS_view[v][0],axis=1)
                    grad_b_0=data_dep - data_indep
                    grad_b_0.shape=(self.K_joint[0],1)
                    grad_b.append(grad_b_0)
                    grad_b_view[v]=grad_b
    
                    # gradient of W
                    grad_W=[]
                    grad_W_0=[None]*self.M[v]
                    for m in range(self.M[v]):
                        data_dep=-numpy.dot(Xbatch[v][m],Hbatch_view[v][0].transpose())/self.batch_size
                        data_indep=-numpy.dot(XS[v][m],HS_view[v][0].transpose())/self.NS
                        grad_W_0[m]=data_dep - data_indep
                    grad_W.append(grad_W_0) # gradient of the negative log-likelihood
                    grad_W_view[v]=grad_W
    
                else:
                    print("Error! Visible type not defined!")
                    exit()

            self.grad_a_view=grad_a_view
            self.grad_W_view=grad_W_view
            self.grad_b_view=grad_b_view
        
            print("grad_a_view:")
            print(grad_a_view[0].transpose()[0,0:10])
            print(grad_a_view[1].transpose()[0,0:10])
            print("grad_W_view:")
            print(grad_W_view[0][0])
            print(grad_W_view[1][0])
            print("grad_b_view:")
            print(grad_b_view[0][0].transpose()[0,0:10])
            print(grad_b_view[1][0].transpose()[0,0:10])

            # if this is just a multimodal RBM, get the gradient of the joint layer and leave
            self.grad_b_joint=self.grad_b_view[0]
            return
        
        ################# multimodal DBM ###################3
        for v in range(self.num_views):
            if self.K_view[v] is None:
                continue
            
            if self.visible_types[v]=="Bernoulli" or self.visible_types[v]=="Poisson" or self.visible_types[v]=="NegativeBinomial" or self.visible_types[v]=="Multinomial" or self.visible_types[v]=="Gaussian_FixPrecision2":
                # gradient of a: data_dep - data_indep
                data_dep=-numpy.mean(Xbatch[v],axis=1)
                data_indep=-numpy.mean(XS[v],axis=1)
                grad_a_view[v]=data_dep - data_indep
                grad_a_view[v].shape=(self.M[v],1)

                # gradient of b
                grad_b=[]
                for nk in range(self.NK_view[v]):
                    data_dep=-numpy.mean(Hbatch_view[v][nk],axis=1)
                    data_indep=-numpy.mean(HS_view[v][nk],axis=1)
                    grad_b_nk=data_dep - data_indep
                    grad_b_nk.shape=(self.K_view[v][nk],1)
                    grad_b.append(grad_b_nk)
                grad_b_view[v]=grad_b

                # gradient of W
                grad_W=[]
                for nk in range(self.NK_view[v]):
                    if nk==0:
                        data_dep=-numpy.dot(Xbatch[v],Hbatch_view[v][nk].transpose())/self.batch_size
                        data_indep=-numpy.dot(XS[v],HS_view[v][nk].transpose())/self.NS
                    else:
                        data_dep=-numpy.dot(Hbatch_view[v][nk-1],Hbatch_view[v][nk].transpose())/self.batch_size
                        data_indep=-numpy.dot(HS_view[v][nk-1],HS_view[v][nk].transpose())/self.NS
                    grad_W_nk=data_dep - data_indep
                    grad_W.append(grad_W_nk) # gradient of the negative log-likelihood
                grad_W_view[v]=grad_W
                
            elif self.visible_types[v]=="Gaussian_FixPrecision1":
                # gradient of a: data_dep - data_indep
                data_dep=-numpy.mean(self.visible_type_fixed_param[v]*Xbatch[v],axis=1)
                data_indep=-numpy.mean(self.visible_type_fixed_param[v]*XS[v],axis=1)
                grad_a_view[v]=data_dep - data_indep
                grad_a_view[v].shape=(self.M[v],1)

                # gradient of b
                grad_b=[]
                for nk in range(self.NK_view[v]):
                    data_dep=-numpy.mean(Hbatch_view[v][nk],axis=1)
                    data_indep=-numpy.mean(HS_view[v][nk],axis=1)
                    grad_b_nk=data_dep - data_indep
                    grad_b_nk.shape=(self.K_view[v][nk],1)
                    grad_b.append(grad_b_nk)
                grad_b_view[v]=grad_b

                # gradient of W
                grad_W=[]
                for nk in range(self.NK_view[v]):
                    if nk==0:
                        data_dep=-numpy.dot(self.visible_type_fixed_param[v]*Xbatch[v],Hbatch_view[v][nk].transpose())/self.batch_size
                        data_indep=-numpy.dot(self.visible_type_fixed_param[v]*XS[v],HS_view[v][nk].transpose())/self.NS
                    else:
                        data_dep=-numpy.dot(Hbatch_view[v][nk-1],Hbatch_view[v][nk].transpose())/self.batch_size
                        data_indep=-numpy.dot(HS_view[v][nk-1],HS_view[v][nk].transpose())/self.NS
                    grad_W_nk=data_dep - data_indep
                    grad_W.append(grad_W_nk) # gradient of the negative log-likelihood
                grad_W_view[v]=grad_W
                
            elif self.visible_types[v]=="Gaussian":
                # gradient of a1, a2: data_dep - data_indep
                data_dep=-numpy.mean(Xbatch[v],axis=1)
                data_indep=-numpy.mean(XS[v],axis=1)
                grad_a1_view_v=data_dep - data_indep
                grad_a1_view_v.shape=(self.M[v],1)

                data_dep=-numpy.mean(Xbatch[v]**2,axis=1)
                data_indep=-numpy.mean(XS[v]**2,axis=1)
                grad_a2_view_v=data_dep - data_indep
                grad_a2_view_v.shape=(self.M[v],1)
                
                grad_a_view=[grad_a1_view_v,grad_a2_view_v]
                
                # gradient of b
                grad_b=[]
                for nk in range(self.NK_view[v]):
                    data_dep=-numpy.mean(Hbatch_view[v][nk],axis=1)
                    data_indep=-numpy.mean(HS_view[v][nk],axis=1)
                    grad_b_nk=data_dep - data_indep
                    grad_b_nk.shape=(self.K_view[v][nk],1)
                    grad_b.append(grad_b_nk)
                grad_b_view[v]=grad_b

                # gradient of W
                grad_W=[]
                for nk in range(self.NK_view[v]):
                    if nk==0:
                        data_dep=-numpy.dot(Xbatch[v],Hbatch_view[v][nk].transpose())/self.batch_size
                        data_indep=-numpy.dot(XS[v],HS_view[v][nk].transpose())/self.NS
                    else:
                        data_dep=-numpy.dot(Hbatch_view[v][nk-1],Hbatch_view[v][nk].transpose())/self.batch_size
                        data_indep=-numpy.dot(HS_view[v][nk-1],HS_view[v][nk].transpose())/self.NS
                    grad_W_nk=data_dep - data_indep
                    grad_W.append(grad_W_nk) # gradient of the negative log-likelihood
                grad_W_view[v]=grad_W
            
            elif self.visible_types[v]=="Gaussian_Hinton":
                # gradient of a1, a2: data_dep - data_indep
                data_dep_x=-numpy.mean(Xbatch[v],axis=1)
                data_indep_x=-numpy.mean(XS[v],axis=1)
                grad_a1=self.a_view[v][1] * (data_dep_x - data_indep_x)
                grad_a1.shape=(self.M[v],1)
                
                data_dep_a2=numpy.mean((Xbatch[v]-self.a_view[v][0])**2,axis=1) - numpy.mean(1/self.a_view[v][1]*Xbatch[v]*self.W_view[v][0].dot(Hbatch_view[v][0]),axis=1)
                data_indep_a2=numpy.mean((XS[v]-self.a_view[v][0])**2,axis=1) - numpy.mean(1/self.a_view[v][1]*XS[v]*self.W_view[v][0].dot(HS_view[v][0]),axis=1)
                grad_a2=data_dep_a2 - data_indep_a2
                grad_a2.shape=(self.M,1)
            
                grad_a_view=[grad_a1_view_v,grad_a2_view_v]
                
                # gradient of b
                grad_b=[]
                for nk in range(self.NK_view[v]):
                    data_dep=-numpy.mean(Hbatch_view[v][nk],axis=1)
                    data_indep=-numpy.mean(HS_view[v][nk],axis=1)
                    grad_b_nk=data_dep - data_indep
                    grad_b_nk.shape=(self.K_view[v][nk],1)
                    grad_b.append(grad_b_nk)
                grad_b_view[v]=grad_b

                # gradient of W
                grad_W=[]
                for nk in range(self.NK_view[v]):
                    if nk==0:
                        data_dep=-numpy.dot(numpy.sqrt(self.a_view[v][1])*Xbatch[v],Hbatch_view[v][nk].transpose())/self.batch_size
                        data_indep=-numpy.dot(numpy.sqrt(self.a_view[v][1])*XS[v],HS_view[v][nk].transpose())/self.NS
                    else:
                        data_dep=-numpy.dot(Hbatch_view[v][nk-1],Hbatch_view[v][nk].transpose())/self.batch_size
                        data_indep=-numpy.dot(HS_view[v][nk-1],HS_view[v][nk].transpose())/self.NS
                    grad_W_nk=data_dep - data_indep
                    grad_W.append(grad_W_nk) # gradient of the negative log-likelihood
                grad_W_view[v]=grad_W
                
            elif self.visible_types[v]=="Multinoulli":
                # gradient of a: data_dep - data_indep
                grad_a_v=[None]*self.M[v]
                for m in range(self.M[v]):
                    data_dep=-numpy.mean(Xbatch[v][m],axis=1)
                    data_indep=-numpy.mean(XS[v][m],axis=1)
                    grad_a_v[m]=data_dep - data_indep
                    grad_a_v[m].shape=(self.visible_type_fixed_param[v][m],1)
                grad_a_view[v]=grad_a_v

                # gradient of b
                grad_b=[]
                for nk in range(self.NK_view[v]):
                    data_dep=-numpy.mean(Hbatch_view[v][nk],axis=1)
                    data_indep=-numpy.mean(HS_view[v][nk],axis=1)
                    grad_b_nk=data_dep - data_indep
                    grad_b_nk.shape=(self.K_view[v][nk],1)
                    grad_b.append(grad_b_nk)
                grad_b_view[v]=grad_b

                # gradient of W
                grad_W=[]
                for nk in range(self.NK_view[v]):
                    if nk==0:
                        grad_W_nk=[None]*self.M[v]
                        for m in range(self.M[v]):
                            data_dep=-numpy.dot(Xbatch[v][m],Hbatch_view[v][nk].transpose())/self.batch_size
                            data_indep=-numpy.dot(XS[v][m],HS_view[v][nk].transpose())/self.NS
                            grad_W_nk[m]=data_dep - data_indep
                    else:
                        data_dep=-numpy.dot(Hbatch_view[v][nk-1],Hbatch_view[v][nk].transpose())/self.batch_size
                        data_indep=-numpy.dot(HS_view[v][nk-1],HS_view[v][nk].transpose())/self.NS
                        grad_W_nk=data_dep - data_indep
                    grad_W.append(grad_W_nk) # gradient of the negative log-likelihood
                grad_W_view[v]=grad_W

            else:
                print("Error! Visible type not defined!")
                exit()

        self.grad_a_view=grad_a_view
        self.grad_W_view=grad_W_view
        self.grad_b_view=grad_b_view 

        # joint
        # gradident of a
        grad_a_joint=numpy.zeros(self.a_joint.shape)
        grad_a_view2joint=[None]*self.num_views
        for v in range(self.num_views):
            # no hidden layer, compute the gradient of a
            if self.K_view[v] is None:
                if self.visible_types[v]=="Gaussian_FixPrecision1":
                    data_dep=-numpy.mean(self.visible_type_fixed_param[v]*Xbatch[v],axis=1)
                    data_indep=-numpy.mean(self.visible_type_fixed_param[v]*XS[v],axis=1)
                    grad_a_joint_v=data_dep - data_indep
                    grad_a_joint_v.shape=(self.M[v],1)
                else:
                    data_dep=-numpy.mean(Xbatch[v],axis=1)
                    data_indep=-numpy.mean(XS[v],axis=1)
                    grad_a_joint_v=data_dep - data_indep
                    grad_a_joint_v.shape=(self.M[v],1)
            else: # at least one hidden layer, set the gradient to zero for views having at least one hidden layer
                grad_a_joint_v=numpy.zeros(shape=(self.K_view[v][-1],1))
            grad_a_view2joint[v]=grad_a_joint_v
        grad_a_joint=self.get_grad_a_joint(grad_a_view2joint)
        
        # gradient of b
        grad_b_joint=[]
        for nk in range(self.NK_joint):
            data_dep=-numpy.mean(Hbatch_joint[nk],axis=1)
            data_indep=-numpy.mean(HS_joint[nk],axis=1)
            grad_b_nk=data_dep - data_indep
            grad_b_nk.shape=(self.K_joint[nk],1)
            grad_b_joint.append(grad_b_nk)

        # gradient of W
        grad_W_joint=[]
        for nk in range(self.NK_joint):
            if nk==0:
                data_dep=-numpy.dot(Xbatch_joint,Hbatch_joint[nk].transpose())/self.batch_size
                data_indep=-numpy.dot(XS_joint,HS_joint[nk].transpose())/self.NS
            else:
                data_dep=-numpy.dot(Hbatch_joint[nk-1],Hbatch_joint[nk].transpose())/self.batch_size
                data_indep=-numpy.dot(HS_joint[nk-1],HS_joint[nk].transpose())/self.NS
            grad_W_nk=data_dep - data_indep
            grad_W_joint.append(grad_W_nk) # gradient of the negtive log-likelihood

        self.grad_a_joint=grad_a_joint
        self.grad_W_joint=grad_W_joint
        self.grad_b_joint=grad_b_joint


    def update_param(self, update_dbms=True):
        """
        Update parameters.
        fix_a_view: is a list of bool types.
        """
        #tol=1e-8
        tol_negbin_max=-1e-8
        tol_negbin_min=-100
        tol_poisson_max=self.tol_poisson_max#16 #numpy.log(255)
        #tol_gamma_min=1e-3
        #tol_gamma_max=1e3
        ############## multimodal RBM ##################        
        if False and self.if_multimodal_RBM:
            for v in range(self.num_views):
                if self.visible_types[v]=="Bernoulli" or self.visible_types[v]=="Multinomial" or self.visible_types[v]=="Gaussian_FixPrecision1" or self.visible_types[v]=="Gaussian_FixPrecision2":
                    if not self.fix_a_view[v]:
                        self.a_view[v]=self.a_view[v] - self.learn_rate_a[v] * self.grad_a_view[v]
                    self.W_view[v][0]=self.W_view[v][0] - self.learn_rate_W[v] * self.grad_W_view[v][0]
                    
                elif self.visible_types[v]=="Poisson":
                    if not self.fix_a_view[v]:
                        self.a_view[v]=self.a_view[v] - self.learn_rate_a[v] * self.grad_a_view[v]
                    self.W_view[v][0]=self.W_view[v][0] - self.learn_rate_W[v] * self.grad_W_view[v][0]
                    # constrain a to make it not wild
                    self.a_view[v][self.a_view[v]>tol_poisson_max]=tol_poisson_max
                    
                elif self.visible_types[v]=="NegativeBinomial":
                    if not self.fix_a_view[v]:
                        self.a_view[v]=self.a_view[v] - self.learn_rate_a[v] * self.grad_a_view[v]
                    self.W_view[v][0]=self.W_view[v][0] - self.learn_rate_W[v] * self.grad_W_view[v][0]
                    # a not too small, not positive,s [-100,0)
                    self.a_view[v][self.a_view[v]>=0]=tol_negbin_max # project a to negative
                    self.a_view[v][self.a_view[v]<tol_negbin_min]=tol_negbin_min
                    self.W_view[v][0][self.W_view[v][0]>0]=0 # project W[0] to negative
                elif self.visible_types[v]=="Gaussian" or self.visible_types[v]=="Gaussian_Hinton" or self.visible_types[v]=="Gamma":
                    if not self.fix_a_view[v]:
                        self.a_view[v][0]=self.a_view[v][0] - self.learn_rate_a[v][0] * self.grad_a_view[v][0]
                        self.a_view[v][1]=self.a_view[v][1] - self.learn_rate_a[v][1] * self.grad_a_view[v][1]
                    self.W_view[v][0]=self.W_view[v][0] - self.learn_rate_W[v] * self.grad_W_view[v][0]
                    
                elif self.visible_types[v]=="Multinoulli":
                    for m in range(self.M[v]):
                        if not self.fix_a_view[v]:
                            self.a_view[v][m]=self.a_view[v][m] - self.learn_rate_a[v] * self.grad_a_view[v][m]
                        for m in range(self.M[v]):
                            self.W_view[v][0][m]=self.W_view[v][0][m] - self.learn_rate_W[v] * self.grad_W_view[v][0][m]
                        
                # update a_view2joint and W_view2joint                  
                self.a_view2joint[v]=self.a_view[v]
                self.W_view2joint[v]= self.W_view[v][0]
            # update the bias of the joint layer
            self.b_joint[0]=self.b_joint[0] - self.learn_rate_b[0] * self.grad_b_joint[0]
            self.b_view=[self.b_joint] * self.num_views
            
            for v in range(self.num_views):
                if not self.fix_a_view[v]:
                    a=self.a_view2joint[v] # or self.a_view[v]
                else:
                    a=None
                b=[self.b_joint[0]]
                W=[self.W_view2joint[v]]
                self.dbm_view[v].set_param(a=a, b=b, W=W, update_rbms=False)
                
            self.W_joint=numpy.vstack(self.W_view2joint)
            self.a_joint=numpy.vstack(self.a_view2joint)
            self.dbm_joint.set_param(a=self.a_joint, b=self.b_joint, W=self.W_joint, update_rbms=False)
            
            print("a_view:")
            print(self.a_view[0].transpose()[0,0:10])
            print(self.a_view[1].transpose()[0,0:10])
            print("W_view:")
            print(self.W_view[0][0])
            print(self.W_view[1][0])
            print("b_view:")
            print(self.b_view[0][0].transpose()[0,0:10])
            print(self.b_view[1][0].transpose()[0,0:10])

            return 
            
            
        ################# multimodal DBM ###########################
        for v in range(self.num_views):
            if self.K_view[v] is not None:
                if self.visible_types[v]=="Bernoulli" or self.visible_types[v]=="Multinomial" or self.visible_types[v]=="Gaussian_FixPrecision1" or self.visible_types[v]=="Gaussian_FixPrecision2":
                    if not self.fix_a_view[v]:
                        self.a_view[v]=self.a_view[v] - self.learn_rate_a[v] * self.grad_a_view[v]
                    for nk in range(self.NK_view[v]):
                        self.W_view[v][nk]=self.W_view[v][nk] - self.learn_rate_W[v] * self.grad_W_view[v][nk]
                        self.b_view[v][nk]=self.b_view[v][nk] - self.learn_rate_b[v] * self.grad_b_view[v][nk]
                        
                elif self.visible_types[v]=="Poisson":
                    if not self.fix_a_view[v]:
                        self.a_view[v]=self.a_view[v] - self.learn_rate_a[v] * self.grad_a_view[v]
                    for nk in range(self.NK_view[v]):
                        self.W_view[v][nk]=self.W_view[v][nk] - self.learn_rate_W[v] * self.grad_W_view[v][nk]
                        self.b_view[v][nk]=self.b_view[v][nk] - self.learn_rate_b[v] * self.grad_b_view[v][nk]
                    # constrain a to make it not wild
                    self.a_view[v][self.a_view[v]>tol_poisson_max]=tol_poisson_max
                    
                elif self.visible_types[v]=="NegativeBinomial":
                    if not self.fix_a_view[v]:
                        self.a_view[v]=self.a_view[v] - self.learn_rate_a[v] * self.grad_a_view[v]
                    for nk in range(self.NK_view[v]):
                        self.W_view[v][nk]=self.W_view[v][nk] - self.learn_rate_W[v] * self.grad_W_view[v][nk]
                        self.b_view[v][nk]=self.b_view[v][nk] - self.learn_rate_b[v] * self.grad_b_view[v][nk]
                    # a not too small, not positive,s [-100,0)
                    self.a_view[v][self.a_view[v]>=0]=tol_negbin_max # project a to negative
                    self.a_view[v][self.a_view[v]<tol_negbin_min]=tol_negbin_min
                    self.W_view[v][0][self.W_view[v][0]>0]=0 # project W[0] to negative
                        
                elif self.visible_types[v]=="Gaussian" or self.visible_types[v]=="Gaussian_Hinton" or self.visible_types[v]=="Gamma":
                    if not self.fix_a_view[v]:
                        self.a_view[v][0]=self.a_view[v][0] - self.learn_rate_a[v][0] * self.grad_a_view[v][0]
                        self.a_view[v][1]=self.a_view[v][1] - self.learn_rate_a[v][1] * self.grad_a_view[v][1]
                    for nk in range(self.NK_view[v]):
                        self.W_view[v][nk]=self.W_view[v][nk] - self.learn_rate_W[v] * self.grad_W_view[v][nk]
                        self.b_view[v][nk]=self.b_view[v][nk] - self.learn_rate_b[v] * self.grad_b_view[v][nk]
                        
                elif self.visible_types[v]=="Multinoulli":
                    for m in range(self.M[v]):
                        if not self.fix_a_view[v]:
                            self.a_view[v][m]=self.a_view[v][m] - self.learn_rate_a[v] * self.grad_a_view[v][m]
                    for nk in range(self.NK_view[v]):
                        if nk==0:
                            for m in range(self.M[v]):
                                self.W_view[v][nk][m]=self.W_view[v][nk][m] - self.learn_rate_W[v] * self.grad_W_view[v][nk][m]
                        else:
                            self.W_view[v][nk]=self.W_view[v][nk] - self.learn_rate_W[v] * self.grad_W_view[v][nk]
                        self.b_view[v][nk]=self.b_view[v][nk] - self.learn_rate_b[v] * self.grad_b_view[v][nk]
            #else: # has no hidden layer
            #    if self.visible_types[v]=="Bernoulli" or self.visible_types[v]=="Poisson" or self.visible_types[v]=="NegativeBinomial" or self.visible_types[v]=="Multinomial":
            #        if not self.fix_a_view[v]:
            #            self.a_view[v]=self.a_view[v] - self.learn_rate_a[v] * self.grad_a_view[v]
            #    elif self.visible_types[v]=="Gaussian" or self.visible_types[v]=="Gaussian_Hinton" or self.visible_types[v]=="Gamma":
            #        if not self.fix_a_view[v]:
            #            self.a_view[v][0]=self.a_view[v][0] - self.learn_rate_a[v][0] * self.grad_a_view[v][0]
            #            self.a_view[v][1]=self.a_view[v][1] - self.learn_rate_a[v][1] * self.grad_a_view[v][1]
            #    elif self.visible_types[v]=="Multinoulli":
            #        for m in range(self.M[v]):
            #            if not self.fix_a_view[v]:
            #                self.a_view[v][m]=self.a_view[v][m] - self.learn_rate_a[v] * self.grad_a_view[v][m]
            
        self.get_a_joint() # must update a_joint, because b_view is updated
        not_fix_a_joint_log_ind=numpy.logical_not(self.fix_a_joint_log_ind)
        not_fix_a_joint_log_ind=numpy.array(not_fix_a_joint_log_ind,dtype=int)
        not_fix_a_joint_log_ind.shape=(len(not_fix_a_joint_log_ind),1)
        # not Gaussian, not Multinoulli, not Gamma for a_joint
        self.a_joint=self.a_joint - self.learn_rate_a[self.num_views] * (not_fix_a_joint_log_ind * self.grad_a_joint)  #SHOULD I UPDATE THIS?! Yes! a_view[v] is updated when it has no hidden layer
        for nk in range(self.NK_joint):
            self.W_joint[nk]=self.W_joint[nk] - self.learn_rate_W[self.num_views] * self.grad_W_joint[nk]
            self.b_joint[nk]=self.b_joint[nk] - self.learn_rate_b[self.num_views] * self.grad_b_joint[nk]

        # must update the separated list of weight matrices from the views to the joint
        self.get_W_view2joint()
        # update another copy of a_joint, update a_view[v] as well
        self.get_a_view2joint()
        
        # update the parameter of the view-specific DBMs
        if update_dbms:
            # update view-specific components
            for v in range(self.num_views):
                if self.K_view[v] is not None:
                    a=self.a_view[v]
                    b=self.b_view[v]
                    W=self.W_view[v]
                    self.dbm_view[v].set_param(a=a, b=b, W=W, update_rbms=False)

            # update joint component
            a=self.a_joint
            b=self.b_joint
            W=self.W_joint
            self.dbm_joint.set_param(a=a, b=b, W=W, update_rbms=False)
            
            # update view-specific components if it has no hidden layer
            for v in range(self.num_views):
                if self.K_view[v] is None:
                    if not self.fix_a_view[v]:
                        a=self.a_view2joint[v] # or self.a_view[v]
                        self.a_view[v]=a
                    else:
                        a=None
                    b=[self.b_joint[0]]
                    self.b_view[v]=b
                    W=[self.W_view2joint[v]]
                    self.W_view[v]=W
                    self.dbm_view[v].set_param(a=a, b=b, W=W, update_rbms=False)

            print("a_view:")
            print(self.a_view[0].transpose()[0,0:10])
            print(self.a_view[1].transpose()[0,0:10])
            print("W_view:")
            print(self.W_view[0][0])
            print(self.W_view[1][0])
            print("W_view2joint:")
            print(self.W_view2joint[0])
            print(self.W_view2joint[1])
            print("b_view:")
            print(self.b_view[0][0].transpose()[0,0:10])
            print(self.b_view[1][0].transpose()[0,0:10])
            print(self.b_joint[0].transpose()[0,0:10])


    def get_param(self): 
        return self.a_view,self.b_view,self.W_view,self.a_joint,self.b_joint,self.W_joint,
 

    def set_param(self, a_view, b_view, W_view, a_joint, b_joint, W_joint, update_dbms=True):
        """
        Set the parameters of multimodal DBM.
        """
        # update view-specific components        
        for v in range(self.num_views):
            self.a_view[v]=a_view[v]
            self.b_view[v]=b_view[v]
            self.W_view[v]=W_view[v]
            if self.K_view[v] is not None and update_dbms:
                a=self.a_view[v]
                b=self.b_view[v]
                W=self.W_view[v]
                self.dbm_view[v].set_param(a=a, b=b, W=W, update_rbms=False)
        
        # update the joint component
        self.get_a_joint()
        #self.a_joint=param["a_joint"]
        self.b_joint=b_joint
        self.W_joint=W_joint
        self.get_a_view2joint()
        self.get_W_view2joint()
        if update_dbms:
            a=self.a_joint
            b=self.b_joint
            W=self.W_joint
            self.dbm_joint.set_param(a=a, b=b, W=W, update_rbms=False)
        
        # update dbm/rbm for trivial views
        for v in range(self.num_views):
            if self.K_view[v] is not None and update_dbms:
                a=self.a_view2joint[v]
                b=self.b_joint[0]
                W=self.W_view2joint[v]
                self.dbm_view[v].set_param(a=a, b=b, W=W, update_rbms=False)


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


    def save_sampling(self, XM, if_sort=None, dir_save="./", prefix="MDBM"):
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
            


