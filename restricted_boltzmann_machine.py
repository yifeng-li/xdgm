
#from __future__ import division
import numpy
import scipy.special
import math
import os
import time
import classification as cl
import sys

class restricted_boltzmann_machine:
    def __init__(self, features=None, M=None, K=None, visible_type="Bernoulli", visible_type_fixed_param=100, hidden_type="Bernoulli", hidden_type_fixed_param=100, tie_W_for_pretraining_DBM_bottom=False, tie_W_for_pretraining_DBM_top=False, if_fix_vis_bias=False, a=None, fix_a_log_ind=None, tol_poisson_max=numpy.log(255), rng=numpy.random.RandomState(100)):
        """
        visible_type: string, can be one from ["Bernoulli", "Gaussian", "HalfNormal", "Laplace", "Exponential", "Poisson", "NegativeBinomial", "Multinomial"]
        visible_type_fixed_param: when visible_type="Multinoulli", it is a vector/list of M numbers. visible_type_fixed_param[m] indicates the variable length of the m-th variable represented in binary code.  
        """
        
        self.features=features
        self.M=M # number of visible variables
        self.K=K # number of latent/hidden variables
        self.rng=rng
        self.visible_type=visible_type
        self.tie_W_for_pretraining_DBM_bottom=tie_W_for_pretraining_DBM_bottom
        self.tie_W_for_pretraining_DBM_top=tie_W_for_pretraining_DBM_top
        self.visible_type_fixed_param=visible_type_fixed_param
        self.mfe_for_loglikelihood_train=None
        self.mfe_for_loglikelihood_test=None
        self.tol_poisson_max=tol_poisson_max   
        
        if self.visible_type=="Bernoulli":
            # this is for Bernoulli-Bernoulli RBM
            #self.W=self.rng.normal(loc=0, scale=0.001, size=(self.M,self.K)) # M by K, initialize weight matrix
            # this is for Bernoulli-Multinomial RBM
            self.W=numpy.zeros(shape=(self.M,self.K),dtype=float)
            self.a=numpy.zeros(shape=(self.M,1)) # length M, a_m=log p_m/(1-p_m) = log 1=0
        elif self.visible_type=="Gaussian":
            #self.W=self.rng.normal(loc=0, scale=0.001, size=(self.M,self.K)) # M by K, initialize weight matrix
            self.W=numpy.zeros(shape=(self.M,self.K),dtype=float)
            self.a=[None]*2
            self.a[0]=self.rng.normal(loc=0, scale=1, size=(self.M,1)) # M X 1  a1=mu*lambda  # coefficient for x
            self.a[1]=-5*numpy.ones(shape=(self.M,1),dtype=float) # M X 1  # a2=-lambda/2<0 coefficient for x^2
        elif self.visible_type=="Gaussian_FixPrecision1":
            #self.W=self.rng.normal(loc=0, scale=0.001, size=(self.M,self.K)) # M by K, initialize weight matrix
            self.W=numpy.zeros(shape=(self.M,self.K),dtype=float)
            #self.a=self.rng.normal(loc=0, scale=0.01, size=(self.M,1)) # M X 1  a=mu  # coefficient for lambda*x
            self.a=numpy.zeros(shape=(self.M,1),dtype=float)
            self.a[self.a<0]=0
            if self.visible_type_fixed_param is None:
                self.visible_type_fixed_param=1*numpy.ones(shape=(self.M,1),dtype=float)
            if numpy.isscalar(self.visible_type_fixed_param):
                self.visible_type_fixed_param=self.visible_type_fixed_param*numpy.ones(shape=(self.M,1),dtype=float)
        elif self.visible_type=="Gaussian_FixPrecision2":
            #self.W=self.rng.normal(loc=0, scale=0.001, size=(self.M,self.K)) # M by K, initialize weight matrix
            self.W=numpy.zeros(shape=(self.M,self.K),dtype=float)   
            #self.a=self.rng.normal(loc=0, scale=0.01, size=(self.M,1)) # M X 1  a=mu*lambda  # coefficient for x
            self.a=numpy.ones(shape=(self.M,1),dtype=float)
            if self.visible_type_fixed_param is None:
                self.visible_type_fixed_param=1*numpy.ones(shape=(self.M,1),dtype=float)
            if numpy.isscalar(self.visible_type_fixed_param):
                self.visible_type_fixed_param=self.visible_type_fixed_param*numpy.ones(shape=(self.M,1),dtype=float)
        elif self.visible_type=="Gaussian_Hinton": 
            self.W=self.rng.normal(loc=0, scale=0.001, size=(self.M,self.K)) # M by K, initialize weight matrix
            self.a=[None]*2
            #self.a[0]=self.rng.normal(loc=0, scale=0.001, size=(self.M,1)) # M X 1, mean, mu
            self.a[0]=self.rng.random_sample(size=(self.M,1))
            self.a[1]=1*numpy.ones(shape=(self.M,1),dtype=float)  # M X 1, precision, lambda>0
        elif self.visible_type=="Poisson":
            #self.W=self.rng.normal(loc=0, scale=0.001, size=(self.M,self.K)) # M by K, initialize weight matrix
            self.W=numpy.zeros(shape=(self.M,self.K),dtype=float)
            #self.a=numpy.zeros(shape=(self.M,1)) # length M, a=log(lambda), log 1 = 0
            self.a=self.rng.normal(loc=0, scale=0.1, size=(self.M,1))
        elif self.visible_type=="NegativeBinomial":
            #self.W=self.rng.normal(loc=0, scale=0.001, size=(self.M,self.K)) # M by K, initialize weight matrix
            #self.W=-0.001*numpy.ones(shape=(self.M,self.K),dtype=float)
            #self.W=-numpy.abs(self.rng.normal(loc=0, scale=0.001, size=(self.M,self.K))) # for 20newsgroups
            self.W=-numpy.abs(self.rng.normal(loc=0, scale=0.00001, size=(self.M,self.K))) # for fashion MNIST
            self.a=numpy.log(0.5)*numpy.ones(shape=(self.M,1)) # length M, since a=log(1-p), a must be <0
            #self.W=self.rng.normal(loc=0, scale=0.0001, size=(self.M,self.K))
            #self.a=self.rng.normal(loc=0, scale=0.0001, size=(self.M,1))
            #self.a=-100*numpy.ones(shape=(self.M,1),dtype=float)
            #self.W[self.W>0]=0
            #self.a[self.a>=0]=-0.00001
            if numpy.isscalar(self.visible_type_fixed_param):
                self.visible_type_fixed_param=self.visible_type_fixed_param*numpy.ones(shape=(self.M,1),dtype=float)
        elif self.visible_type=="Multinoulli":
            self.Ms=self.visible_type_fixed_param # a list of vector, the length of each multinoulli variable.
            self.W=[None]*self.M
            self.a=[None]*self.M
            for m in range(self.M):
                self.W[m]=self.rng.normal(loc=0, scale=0.001, size=(self.Ms[m],self.K))
                self.a[m]=math.log(1/self.Ms[m])*numpy.ones(shape=(self.Ms[m],1))
        elif self.visible_type=="Multinomial":
            #self.W=self.rng.normal(loc=0, scale=0.001, size=(self.M,self.K)) # M by K, initialize weight matrix
            self.W=numpy.zeros(shape=(self.M,self.K),dtype=float)
            #self.a=math.log(1/self.M)*numpy.ones(shape=(self.M,1)) # length M, a_m=log p_m <0
            self.a=numpy.zeros(shape=(self.M,1),dtype=float)
        elif self.visible_type=="Gamma":
            #self.W=-0.0001*numpy.ones(shape=(self.M,self.K),dtype=float) # M by K, initialize weight matrix
            self.W=-0.001*numpy.ones(shape=(self.M,self.K),dtype=float)
            self.a=[None]*2
            self.a[0]=10*numpy.ones(shape=(self.M,1), dtype=float) # M X 1  a1=alpha-1  # bias for log x
            self.a[1]=-numpy.ones(shape=(self.M,1),dtype=float) # M X 1  # a2=-beta<0 coefficient for x
        else:
            print("Error! Please select a correct data type for visible variables from {Bernoulli,Gaussian,Multinoulli,Poisson}.")
            sys.exit(0)
            
        # or call self.fix_via_bias(a, fix_a_log_ind)
        self.if_fix_vis_bias=if_fix_vis_bias
        self.fix_a_log_ind=fix_a_log_ind
        if self.if_fix_vis_bias:
            if self.fix_a_log_ind is None:
                self.fix_a_log_ind=numpy.array([True]*self.M)
            if a is not None:
                self.a=a
                print("I will fix a using the new a in this RBM.")
            else:
                print("I will fix the existing a in this RBM.")
        
        # about hidden nodes
        self.hidden_type=hidden_type
        self.hidden_type_fixed_param=hidden_type_fixed_param
        if self.hidden_type in ["Bernoulli","Binomial","Multinomial"]:
            self.b=numpy.zeros(shape=(self.K,1)) # length K, b=log(p/(1-p))
            if self.visible_type=="NegativeBinomial":
                self.b=numpy.zeros(shape=(self.K,1))
        elif self.hidden_type=="Gaussian":
            self.b=[None,None]
            self.b[0]=self.rng.normal(loc=0, scale=1, size=(self.K,1)) # K X 1  b1=mu*lambda  # coefficient for h
            self.b[1]=-5*numpy.ones(shape=(self.K,1),dtype=float)
        else:
            print("Error! The required distribution for hidden nodes has not implemented yet.")
       

    def fix_vis_bias(self, a=None, fix_a_log_ind=None):
        """
        Fix the visible bias in training.
        For Gaussian, a is a 2-d tuple, a[0] for a1 and a[1] for a2.
        For Multinoulli, a is a M-d tuple or list.
        """
        self.if_fix_bis_bias=True
        self.fix_a_log_ind=fix_a_log_ind
        if self.fix_a_log_ind is None:
            self.fix_a_log_ind=numpy.array([True]*self.M)
        if a is not None:
            self.a=a
            print("I will fix a using the new a in this RBM.")
        else:
            print("I will fix the existing a in this RBM.")

    
    def estimate_log_likelihood(self, X=None, base_rate_type="prior", beta=None, step=0.999, T=10000, stepdist="even", S=100, reuse_mfe=False, train_or_test="train",  dir_save="/.", prefix="RBM"):
        """
        Compute the log-likelihood of RBM.
        """
        if X is None:
            X=self.X
        print("I am estimating the log-likelihood...")
        # compute free energy 
        if reuse_mfe==False:
            mfe,_=self.compute_free_energy(X)
        else:
            if train_or_test=="train":
                if self.mfe_for_loglikelihood_train is not None:
                    mfe=self.mfe_for_loglikelihood_train
                else:
                    mfe,_=self.compute_free_energy(X)
                    self.mfe_for_loglikelihood_train=mfe
            if train_or_test=="test":
                if self.mfe_for_loglikelihood_test is not None:
                    mfe=self.mfe_for_loglikelihood_test
                else:
                    mfe,_=self.compute_free_energy(X)
                    self.mfe_for_loglikelihood_test=mfe
        # estimate logZ
        logZ,logZ_std,logws,log_ratio_AIS_mean,log_ratio_AIS_std=self.estimate_logZ(base_rate_type=base_rate_type, beta=beta, step=step, T=T, stepdist=stepdist, S=S)
        # compute log-likelihood 
        loglh= -mfe - logZ
        # compute the log prior
        #self.compute_log_prior()
        print("log-likelihood:{0}, logZ:{1}, logZ_std:{2}, free_energy:{3}".format(loglh,logZ,logZ_std,mfe))
        
        result=numpy.zeros(shape=(4,2),dtype=object)
        result[:,0]=numpy.array(["log-likelihood","logZ","logZ_std","free_energy"])
        result[:,1]=numpy.array([loglh,logZ,logZ_std,mfe])
        filename=dir_save + prefix + "_" + train_or_test + "_estimated_log_likelihood.txt"
        numpy.savetxt(filename, result, delimiter="\t", fmt="%s")
        return loglh,logZ,logZ_std,mfe
      
      
    def compute_log_prior(self, theta, mu, lambda_G, lambda_L):
        """
        Compute the log of prior p_G(theta|mu,lambda_G)p_L(theta|mu,lambda_L)
        Parameters can be scalars, vectors or matrices.
        """
        log_pG= - (lambda_G/2) * (theta-mu)**2 + 0.5*numpy.log(lambda_G/(2*numpy.pi))
        log_pL= - lambda_L * numpy.abs(theta-mu) + numpy.log(0.5*lambda_L)
        log_pGL=log_pG + log_pL
        log_pGL=log_pGL.sum()
        return log_pGL
      
      
    def estimate_logZ(self, base_rate_type="prior", beta=None, step=0.999, T=10000, stepdist="even", S=100):
        """
        Estimate the (log) partition function.
        base_type: string, the type of model A, either "uniform" or "prior".
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
                beta=1-step**beta
            if stepdist=="even":
                beta=beta/T
            beta=numpy.concatenate((beta,[1]))            
            T=len(beta) # actually, there are T+1 elements in beta
        
        a_A,b_A,logZ_A=self.compute_logZ_A(self.a, self.b, visible_type=self.visible_type, visible_type_fixed_param=self.visible_type_fixed_param, hidden_type=self.hidden_type, hidden_type_fixed_param=self.hidden_type_fixed_param, base_rate_type=base_rate_type)        
        print("logZ_A={0}".format(logZ_A))
        
        print("I need to run AIS for {0} times.".format(S))
        logws=numpy.zeros(shape=(S,),dtype=float)
        #ws=numpy.zeros(shape=(S,),dtype=float)
        for s in range(S):
            print("I am running the {0}-th AIS...".format(s))
            # Markov chain
            x_t=numpy.zeros(shape=(self.M,1),dtype=float) # used to initialize, how about multinoulli types?
            log_p_star_diff=numpy.zeros(shape=(T-1,),dtype=float)
            h_B_t=numpy.zeros(shape=(self.K,1),dtype=float)
            for t in range(T-1):
                # sampling from T(x_{t+1} | x_t)
                # h_A_t is never used as the W_A is zero 
                #b_A_hat=(1-beta[t])*b_A
                #h_A_t=self.sample_h_given_x(X=None, b_hat=b_A_hat, hidden_value_or_meanfield="value")
                a_B_hat=self.compute_posterior_bais(self.a, self.W, h_B_t, visible_or_hidden="visible", distribution=self.visible_type, opposite_type=self.hidden_type, opposite_fixed_param=self.hidden_type_fixed_param)
                a_hat_t=self.combine_a_A_a_B_hat(a_A, a_B_hat, beta[t], self.visible_type)
                x_t,_,_=self.sample_x_given_h(H=None, a_hat=a_hat_t)
                b_B_hat=self.compute_posterior_bais(self.b, self.W, x_t, visible_or_hidden="hidden", distribution=self.hidden_type, opposite_type=self.visible_type, opposite_fixed_param=self.visible_type_fixed_param)
                b_B_hat=beta[t]*b_B_hat
                h_B_t,_=self.sample_h_given_x(X=None, b_hat=b_B_hat, hidden_value_or_meanfield="value")

                # compute log p^*_t(x)
                log_p_star_t_x_t=self.log_p_star_t(x=x_t, beta_t=beta[t], a_A=a_A, b_A=b_A, a_B=self.a, b_B=self.b, W_B=self.W, visible_type=self.visible_type, visible_type_fixed_param=self.visible_type_fixed_param, hidden_type=self.hidden_type, hidden_type_fixed_param=self.hidden_type_fixed_param)
                log_p_star_tplus1_x_t=self.log_p_star_t(x=x_t, beta_t=beta[t+1], a_A=a_A, b_A=b_A, a_B=self.a, b_B=self.b, W_B=self.W, visible_type=self.visible_type, visible_type_fixed_param=self.visible_type_fixed_param, hidden_type=self.hidden_type, hidden_type_fixed_param=self.hidden_type_fixed_param)
                log_p_star_diff[t]=log_p_star_tplus1_x_t-log_p_star_t_x_t
            
            logws[s]=numpy.sum(log_p_star_diff)
        log_ratio_AIS_mean=numpy.mean(logws)
        #ws[s]=numpy.exp(logws[s])
        #ratio_AIS_mean=numpy.mean(ws)
        if S>1:
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


    def combine_a_A_a_B_hat(self, a_A, a_B_hat, beta_t, visible_type):
        if visible_type in ["Bernoulli","Poisson","NegativeBinomial","Multinomial","Gaussian_FixPrecision1","Gaussian_FixPrecision2"]:
            a_hat_t=(1-beta_t)*a_A + beta_t*a_B_hat
        elif visible_type in ["Gaussian","Gamma"]:
            a_hat_t=[None,None]
            a_hat_t[0]=(1-beta_t)*a_A[0] + beta_t*a_B_hat[0]
            a_hat_t[1]=(1-beta_t)*a_A[1] + beta_t*a_B_hat[1]
        return a_hat_t
        
        
    def log_p_star_t(self, x, beta_t, a_A, b_A, a_B, b_B, W_B, visible_type="Bernoulli", visible_type_fixed_param=None, hidden_type="Bernoulli", hidden_type_fixed_param=None):
        """
        log p^*_t(x) = -F_t(x)= zeta(x) + sum_k B_k( (1-beta_t)b_k ) + sum_k B_k( 1-beta_t hat{b}_k )
        """
        # the posterior bias b_B_hat        
        b_B_hat=self.compute_posterior_bais(b_B, W_B, x, visible_or_hidden="hidden", distribution=hidden_type, opposite_type=visible_type, opposite_fixed_param=visible_type_fixed_param)
        
        # -F_t(x)
        log_p_star_t_x=(1-beta_t)*self.zeta(a_A, x, fixed_param=visible_type_fixed_param, distribution=visible_type) + self.A((1-beta_t)*b_A, fixed_param=hidden_type_fixed_param, distribution=hidden_type) + beta_t*self.zeta(a_B, x, fixed_param=visible_type_fixed_param, distribution=visible_type) + self.A(beta_t*b_B_hat, fixed_param=hidden_type_fixed_param, distribution=hidden_type)
        return log_p_star_t_x


    def zeta(self, theta, X, fixed_param=1, distribution="Bernoulli"):
        """
        The Zeta function in the free energy function of exp-RBM.
        INPUTS:
        theta: vector, or list of vectors, the natural prior parameters of the visible units.
        X: vector or matrix, or a list of vector/matrix (for Multinoulli), each column represents a sample.
        OUTPUTS: 
        z: scalar or vector of length N, the value of the zeta function. N is the number of provided samples.
        """
        
        if distribution=="Bernoulli":
            z=numpy.dot(X.transpose(),theta)
        elif distribution=="Gaussian":
            theta1=theta[0]
            theta2=theta[1]
            z=numpy.dot(X.transpose(),theta1) + numpy.dot((X**2).transpose(),theta2)
        elif distribution=="Gaussian_FixPrecision1":
            M=len(theta) # number of visible units
            ONE_M_1=numpy.ones((M,1))
            lambdaX=self.visible_type_fixed_param*X
            hx=numpy.log(numpy.sqrt(fixed_param/(2*numpy.pi))) - 0.5 * fixed_param * X**2
            z=numpy.dot(lambdaX.transpose(),theta) + numpy.dot(hx.transpose(),ONE_M_1) 
        elif distribution=="Gaussian_FixPrecision2":
            M=len(theta) # number of visible units
            ONE_M_1=numpy.ones((M,1))
            hx=numpy.log(numpy.sqrt(fixed_param/(2*numpy.pi))) - 0.5 * fixed_param * X**2
            z=numpy.dot(X.transpose(),theta) + numpy.dot(hx.transpose(),ONE_M_1) 
        elif distribution=="Poisson":
            M=len(theta) # number of visible units
            ONE_M_1=numpy.ones((M,1))
            z=numpy.dot(X.transpose(),theta) - numpy.dot(scipy.special.gammaln(X+1).transpose(),ONE_M_1)
        elif distribution=="Binomial":
            # fixed_param is the total number of trials
            M=len(theta) # number of visible units
            ONE_M_1=numpy.ones((M,1))
            z=numpy.dot(X.transpose(),theta) + numpy.dot(scipy.special.gammaln(fixed_param.transpose()+1) - scipy.special.gammaln(X.transpose()+1) - scipy.special.gammaln(fixed_param.transpose()-X.transpose()+1),ONE_M_1)
        elif distribution=="NegativeBinomial":
            # fixed_param is the number of successes            
            M=len(theta) # number of visible units
            ONE_M_1=numpy.ones((M,1))
            z=numpy.dot(X.transpose(),theta) + numpy.dot(scipy.special.gammaln(X.transpose()+fixed_param.transpose()) - scipy.special.gammaln(fixed_param.transpose()) - scipy.special.gammaln(X.transpose()+1),ONE_M_1)
        elif distribution=="Multinoulli":
            M=len(theta)
            ONE_M_1=numpy.ones((M,1))
            z=0
            for m in range(M):
                z= z + numpy.dot(X[m].transpose(),theta[m])    
        elif distribution=="Multinomial":
            #fixed_param is the total number of counts
            M=len(theta)
            ONE_M_1=numpy.ones((M,1))
            z=numpy.dot(X.transpose(),theta) + scipy.special.gammaln(fixed_param+1) - numpy.dot(scipy.special.gammaln(X.transpose()+1),ONE_M_1)
            #print "atx="
            #print numpy.dot(X.transpose(),theta)
            #print "log part="
            #print scipy.special.gammaln(fixed_param+1) - numpy.dot(scipy.special.gammaln(X.transpose()+1),ONE_M_1)
        elif distribution=="Gamma":
            theta1=theta[0]
            theta2=theta[1]
            z=numpy.dot(numpy.log(X.transpose()), theta1) + numpy.dot(X.transpose(), theta2) 
        else: 
            print("Please specify the correct type of distribution from the exponential family!")
            sys.exit(0)
        
        # posterior
        if z.shape[1]>1:
            z=numpy.diag(z)
            
        #mz=numpy.mean(z)
        z.shape=(z.size,) # reshape NX1 to N
        if len(z)==1:
            z=z[0]
        return z
        

    def compute_posterior_bais(self, theta, W, Z, visible_or_hidden="hidden", distribution="Bernoulli", opposite_type="Bernoulli", opposite_fixed_param=0):
        """
        Compute the posterior biases (either a or b) of an exp-RBM, given data (either X or H).
        theta: vector or a list of vector, either a or b.
        W: the interaction matrix.
        Z: vector (a list of vector) or matrix (a list of matrix), either X or H.
        """
        if opposite_type=="Gaussian_FixPrecision1":
            Z=opposite_fixed_param*Z
            
        if visible_or_hidden=="hidden":
            if distribution in ["Bernoulli", "Poisson", "Binomial", "NegativeBinomial", "Multinomial", "Gaussian_FixPrecision1", "Gaussian_FixPrecision2"]:
                theta_hat=theta + numpy.dot(W.transpose(),Z)
            elif distribution=="Gaussian":
                theta_hat1=theta[0] + numpy.dot(W.transpose(),Z)
                theta_hat2=theta[1]
                theta_hat=[theta_hat1, theta_hat2]
            elif distribution=="Multinoulli":
                theta_hat=theta
                M=len(theta)
                for m in range(M):
                    theta_hat[m]=theta[m] + (W[m].transpose(),Z)
            elif distribution=="Gamma":
                theta_hat1=theta[0]
                theta_hat2=theta[1] + numpy.dot(W.transpose(),Z)
                theta_hat=[theta_hat1, theta_hat2]

        if visible_or_hidden=="visible":
            if distribution in ["Bernoulli", "Poisson", "Binomial", "NegativeBinomial", "Multinomial", "Gaussian_FixPrecision1", "Gaussian_FixPrecision2"]:
                theta_hat=theta + numpy.dot(W,Z)
            elif distribution=="Gaussian":
                theta_hat1=theta[0] + numpy.dot(W,Z)
                theta_hat2=theta[1]
                theta_hat=[theta_hat1, theta_hat2]
            elif distribution=="Multinoulli":
                theta_hat=theta
                M=len(theta)
                for m in range(M):
                    theta_hat[m]=theta[m] + (W[m],Z)
            elif distribution=="Gamma":
                theta_hat1=theta[0]
                theta_hat2=theta[1] + numpy.dot(W,Z)
                theta_hat=[theta_hat1, theta_hat2]
        
        return theta_hat


    def A(self, theta, fixed_param=None, distribution="Bernoulli"):
        """
        log-partition function of an exponential family distribution.
        theta: scalar (list of scalars e.g. for Multinoulli) or column vector (list of column vectors e.g. for Multinoulli) or matrix (a list of matrices e.g. for Multinoulli), the prior or posterior natrual parameters of exponential family distributions.  When theta is a column vector or a list or column vectors, the sum is taken: sum(A(theta)) for the convenience of computing free energy function in exp-RBM. When theta is a matrix or a list of matrices (for Multinoulli), sum(A(theta),axis=0) is taken. 
        """
        if distribution=="Bernoulli":
            #lp=numpy.sum( numpy.log(1+numpy.exp(theta)), axis=0) # old, not robust code
            threshold_max=20 # avoid overflow
            if numpy.isscalar(theta):
                if theta<threshold_max:
                    lp=numpy.log(1+numpy.exp(theta))
                else:
                    lp=theta
            else: # column vector or matrix
                rs,cs=theta.shape
                log_one_plus_exp=numpy.zeros(shape=theta.shape)
                for r in range(rs):
                    for c in range(cs):
                        if theta[r,c]<threshold_max:
                            log_one_plus_exp[r,c]=numpy.log(1+numpy.exp(theta[r,c]))
                        else:
                            log_one_plus_exp[r,c]=theta[r,c]
                lp=numpy.sum( log_one_plus_exp, axis=0)
        elif distribution=="Gaussian":
            theta1=theta[0]
            theta2=theta[1]
            if numpy.any(theta2==0): # for AIS
                lp=numpy.array([0])
            else:
                lp=numpy.sum( 0.5*numpy.log(2*math.pi/(-theta2)) - theta1**2/(4*theta2), axis=0 )
        elif distribution=="Gaussian_FixPrecision1":
            lp=numpy.sum(0.5*fixed_param * theta**2, axis=0)
        elif distribution=="Gaussian_FixPrecision2":
            lp=numpy.sum(theta**2/(2*fixed_param), axis=0)
        elif distribution=="Poisson":
            lp=numpy.sum(numpy.exp(theta), axis=0)
        elif distribution=="Binomial":
            # fixed_param is the total number of trials
            #lp=numpy.sum( fixed_param*numpy.log(1+numpy.exp(theta)), axis=0 )
            threshold_max=20 # avoid overflow
            if numpy.isscalar(theta):
                if theta<threshold_max:
                    lp=numpy.log(1+numpy.exp(theta))
                else:
                    lp=fixed_param*theta
            else: # column vector or matrix
                rs,cs=theta.shape
                log_one_plus_exp=numpy.zeros(shape=theta.shape)
                for r in range(rs):
                    for c in range(cs):
                        if theta[r,c]<threshold_max:
                            log_one_plus_exp[r,c]=numpy.log(1+numpy.exp(theta[r,c]))
                        else:
                            log_one_plus_exp[r,c]=theta[r,c]
                lp=numpy.sum( fixed_param*log_one_plus_exp, axis=0)            
        elif distribution=="NegativeBinomial":
            # fixed_param is the number of successes
            if numpy.any(theta==0): # this is used in AIS
                lp=numpy.array([0])
            else: # this is used in computing free energy
                lp=numpy.sum( -fixed_param*numpy.log(1-numpy.exp(theta)), axis=0 )
        elif distribution=="Multinoulli":
            # when theta is a list of one vector (length of C_m), we have one multinoulli unit; when theta is a list of M vectors (each has length of C_m), we have M multinoulli units; when theta is a list of M matrices (each C_mXN), we have M multinoulli units for N samples.
            M=len(theta) # number of multinoulli units
            N=theta[0].shape[1] # number of samples
            lp=numpy.zeros((N,1))
            for m in range(M):
                lp=lp + numpy.log(numpy.sum(numpy.exp(theta[m]),axis=0))
        elif distribution=="Multinomial":
            #fixed_param is the total number of counts
            lp=fixed_param*numpy.log(numpy.sum(numpy.exp(theta), axis=0) )
        elif distribution=="Gamma":
            theta1=theta[0]
            theta2=theta[1]
            lp=numpy.sum(scipy.special.gammaln(theta1+1) -(theta1+1)*numpy.log(-theta2), axis=0)
        else: 
            print("Please specify the correct type of distribution from the exponential family!")
            sys.exit(0)
            
        lp.shape=(lp.size,) # reshape NX1 to N
        if len(lp)==1:
            lp=lp[0]
        return lp


    def compute_logZ_A(self, a_B, b_B, visible_type="Bernoulli", visible_type_fixed_param=None, hidden_type="Bernoulli", hidden_type_fixed_param=None, base_rate_type="prior"):
        """
        Compute the log-partition function of the prior or uniform base-rate model A.
        a_B, b_B: the biases of model B.
        """
        if base_rate_type=="prior":
            a_A=a_B
            b_A=b_B
        elif base_rate_type=="uniform":  
            
            if visible_type=="Bernoulli" or visible_type=="Poisson" or visible_type=="Multinomial" or visible_type=="Guassian_FixPrecision" or visible_type=="Guassian_FixPrecision2":
                a_A=numpy.zeros(shape=a_B.shape,dtype=float)
            elif visible_type=="Gaussian":
                a_A1=numpy.zeros(shape=a_B[0].shape,dtype=float)
                a_A2=math.pi*numpy.ones(shape=a_B[1].shape,dtype=float)
                a_A=[a_A1,a_A2]
            elif visible_type=="NegativeBinomial":
                a_A=numpy.log(0.5)*numpy.ones(shape=a_B.shape,dtype=float)
            elif visible_type=="Multinoulli":
                M=len(a_B)
                a_A=[None]*M
                for m in range(M):
                    a_A[m]=numpy.zeros(shape=a_B[m].shape,dtype=float)
            elif visible_type=="Gamma":
                a_A1=numpy.zeros(shape=a_B[0].shape,dtype=float)
                a_A2=-numpy.ones(shape=a_B[1].shape,dtype=float)
                a_A=[a_A1,a_A2]

            if hidden_type=="Bernoulli" or hidden_type=="Poisson" or hidden_type=="Multinomial" or  hidden_type=="Gaussian_FixPrecision1" or hidden_type=="Gaussian_FixPrecision2":
                b_A=numpy.zeros(shape=b_B.shape,dtype=float)
            elif hidden_type=="Gaussian":
                b_A1=numpy.zeros(shape=b_B[0].shape,dtype=float)
                b_A2=-math.pi*numpy.ones(shape=b_B[1].shape,dtype=float)
                b_A=[b_A1,b_A2]
            elif hidden_type=="NegativeBinomial":
                b_A=numpy.log(0.5)*numpy.ones(shape=b_B.shape,dtype=float)
            elif hidden_type=="Multinoulli":
                b_A=[]
                M=len(b_B)
                for m in range(M):
                    b_A[m]=numpy.zeros(shape=b_B[m].shape,dtype=float)
            elif hidden_type=="Gamma":
                b_A1=numpy.ones(shape=b_B[0].shape,dtype=float)
                b_A2=-numpy.ones(shape=b_B[1].shape,dtype=float)
                b_A=[b_A1,b_A2]
                
        # mast equation for log-partition function of model A
        logZ_A=self.A(a_A,fixed_param=visible_type_fixed_param, distribution=visible_type) + self.A(b_A,fixed_param=hidden_type_fixed_param, distribution=hidden_type)
        return a_A,b_A,logZ_A


    def repeat_a(self,a,N):
        # repeat a N times column-wise
        if self.visible_type in ["Bernoulli", "Poisson", "NegativeBinomial", "Multinomial", "Gaussian_FixPrecision1", "Gaussian_FixPrecision2"]:
            a_rep=numpy.repeat(a,N,axis=1)
        elif self.visible_type in ["Gaussian", "Gaussian_Hinton"]:
            a_rep=[None,None]
            a_rep[0]=numpy.repeat(a[0],N,axis=1)
            a_rep[1]=a[1]
        elif self.visible_type=="Gamma":
            a_rep=[None,None]
            a_rep[0]=a[0]
            a_rep[1]=numpy.repeat(a[1],N,axis=1)
        elif self.visible_type=="Multinoulli":
            a_rep=[None]*self.M
            for m in range(self.M):
                a_rep[m]=numpy.repeat(a[m],N,axis=1)
        return a_rep
        
        
    def pcd_sampling(self,pcdk=20, NS=20, X0=None, persistent=True, rand_init=True, init=False): 
        "persistent cd sampling"
        if not persistent:
            if X0 is not None: # X0 should be the current mini-batch
                init=True
                rand_init=False
            else:
                print("Error! You want to use CD-k sampling, but you did not give me a batch of training samples.")
                sys.exit(0)

        if init:
            # initialize Markov chains
            self.NS=NS
            a=self.repeat_a(self.a,self.NS)
            Xs,XM,P=self.sample_x_given_h(H=None, a_hat=a)

            if rand_init:
                X0=Xs
            else:
                if X0 is None:
                    X0=self.sample_minibatch(self.NS)
            self.chainX=X0 #numpy.zeros(shape=(self.M,self.NS),dtype=float)
            self.chainXM=XM
            H0,HM=self.sample_h_given_x(X0) #numpy.zeros(shape=(self.K,self.NS),dtype=float)
            self.chainH=H0
            self.chainHM=HM
            self.chain_length=0
            #return self.chainX,self.chainH,self.chainXP,self.chainHP,self.chain_length
        
        for s in range(pcdk):
            self.chainX,self.chainXM,_=self.sample_x_given_h(self.chainH)
            self.chainH,self.chainHM=self.sample_h_given_x(self.chainX)
            self.chain_length=self.chain_length+1

        return self.chainX,self.chainH,self.chainXM,self.chainHM,self.chain_length


    def subgradient(self,theta):
        """
        Compute the subgradient of |theta|.
        theta: vector, or matrix.
        """
        sign_theta=numpy.zeros(shape=theta.shape,dtype=float)
        sign_theta[theta>0]=1
        sign_theta[theta<0]=-1
        return sign_theta        


    def compute_gradient(self,Xbatch,Hbatch,XS,HS):
        """
        Compute gradient.
        """
        if self.visible_type in ["Bernoulli", "Poisson", "NegativeBinomial","Multinomial","Gaussian_FixPrecision2"]:
            # gradient of a
            data_dep=-numpy.mean(Xbatch,axis=1)
            data_indep=-numpy.mean(XS,axis=1)
            grad_a=data_dep - data_indep
            grad_a.shape=(self.M,1)
            #grad_a=grad_a + self.reg_lambda_a*(1-self.reg_alpha_a)*self.a + self.reg_lambda_a*self.reg_alpha_a*self.subgradient(self.a) 

            # gradient of W
            data_dep=-numpy.dot(Xbatch,Hbatch.transpose())/self.batch_size
            data_indep=-numpy.dot(XS,HS.transpose())/self.NS
            grad_W=data_dep - data_indep #+ self.reg_lambda_W*(1-self.reg_alpha_W)*self.W + self.reg_lambda_W*self.reg_alpha_W*self.subgradient(self.W)

            self.grad_a=grad_a
            self.grad_W=grad_W
            
        elif self.visible_type=="Multinoulli":
            grad_a=[None]*self.M
            grad_W=[None]*self.M
            for m in range(self.M):
                # gradient of a
                data_dep=-numpy.mean(Xbatch[m],axis=1)
                data_indep=-numpy.mean(XS[m],axis=1)
                grad_a[m]=data_dep - data_indep
                grad_a[m].shape=(self.Ms[m],1)
                #grad_a[m]=grad_a[m] +  self.reg_lambda_a*(1-self.reg_alpha_a)*self.a[m] + self.reg_lambda_a*self.reg_alpha_a*self.subgradient(self.a[m]) 
                
                # gradient of W
                data_dep=-numpy.dot(Xbatch[m],Hbatch.transpose())/self.batch_size
                data_indep=-numpy.dot(XS[m],HS.transpose())/self.NS
                grad_W[m]=data_dep - data_indep #+ self.reg_lambda_W*(1-self.reg_alpha_W)*self.W[m] + self.reg_lambda_W*self.reg_alpha_W*self.subgradient(self.W[m])

            self.grad_a=grad_a
            self.grad_W=grad_W
            
        elif self.visible_type=="Gaussian_FixPrecision1":
            # gradient of a
            data_dep=-numpy.mean(self.visible_type_fixed_param*Xbatch,axis=1)
            data_indep=-numpy.mean(self.visible_type_fixed_param*XS,axis=1)
            grad_a=data_dep - data_indep
            grad_a.shape=(self.M,1)
            #grad_a=grad_a + self.reg_lambda_a*(1-self.reg_alpha_a)*self.a + self.reg_lambda_a*self.reg_alpha_a*self.subgradient(self.a) 

            # gradient of W
            data_dep=-numpy.dot(self.visible_type_fixed_param*Xbatch,Hbatch.transpose())/self.batch_size
            data_indep=-numpy.dot(self.visible_type_fixed_param*XS,HS.transpose())/self.NS
            grad_W=data_dep - data_indep #+ self.reg_lambda_W*(1-self.reg_alpha_W)*self.W + self.reg_lambda_W*self.reg_alpha_W*self.subgradient(self.W)

            self.grad_a=grad_a
            self.grad_W=grad_W
            
#            print(Hbatch.max())
#            print(Hbatch.min())
#            print(HS.max())
#            print(HS.min())
#            print(Xbatch.max())
#            print(Xbatch.min())
#            print(XS.max())
#            print(XS.min())
#            print(self.grad_a[0:10])
#            print(self.grad_W)
            
        elif self.visible_type=="Gaussian":
            # gradient of a1
            data_dep=-numpy.mean(Xbatch,axis=1)
            data_indep=-numpy.mean(XS,axis=1)
            grad_a1=data_dep - data_indep
            grad_a1.shape=(self.M,1)
            #grad_a1=grad_a1 + self.reg_lambda_a[0]*(1-self.reg_alpha_a[0])*self.a[0] + self.reg_lambda_a[0]*self.reg_alpha_a[0]*self.subgradient(self.a[0])

            # gradient of a2
            data_dep=-numpy.mean(Xbatch**2,axis=1)
            data_indep=-numpy.mean(XS**2,axis=1)
            grad_a2=data_dep - data_indep
            grad_a2.shape=(self.M,1)
            #grad_a2=grad_a2 + self.reg_lambda_a[1]*(1-self.reg_alpha_a[1])*self.a[1] + self.reg_lambda_a[1]*self.reg_alpha_a[1]*self.subgradient(self.a[1])

            # gradient of W
            data_dep=-numpy.dot(Xbatch,Hbatch.transpose())/self.batch_size
            data_indep=-numpy.dot(XS,HS.transpose())/self.NS
            grad_W=data_dep - data_indep #+ self.reg_lambda_W*(1-self.reg_alpha_W)*self.W + self.reg_lambda_W*self.reg_alpha_W*self.subgradient(self.W)

            self.grad_a=[grad_a1,grad_a2]
            self.grad_W=grad_W
            
#            print(Hbatch.max())
#            print(Hbatch.min())
#            print(HS.max())
#            print(HS.min())
#            print(Xbatch.max())
#            print(Xbatch.min())
#            print(XS.max())
#            print(XS.min())
#            print(self.grad_a[0][0:10])
#            print(self.grad_a[1][0:10])
#            print(self.grad_W)

        elif self.visible_type=="Gaussian_Hinton":
            # gradient of a
            data_dep=-numpy.mean(Xbatch,axis=1)
            data_indep=-numpy.mean(XS,axis=1)
            grad_a1=self.a[1]*(data_dep - data_indep)
            grad_a1.shape=(self.M,1)
            #grad_a1=grad_a1 +  self.reg_lambda_a[0]*(1-self.reg_alpha_a[0])*self.a[0] + self.reg_lambda_a[0]*self.reg_alpha_a[0]*self.subgradient(self.a[0])

            # grad of beta
            data_dep_a2=numpy.mean((Xbatch-self.a[0])**2,axis=1) - numpy.mean(1/numpy.sqrt(self.a[1])*Xbatch*self.W.dot(Hbatch),axis=1)
            data_indep_a2=numpy.mean((XS-self.a[0])**2,axis=1) - numpy.mean(1/numpy.sqrt(self.a[1])*XS*self.W.dot(HS),axis=1)
            grad_a2=data_dep_a2 - data_indep_a2
            grad_a2.shape=(self.M,1)
            #grad_a2=grad_a2 +  self.reg_lambda_a[1]*(1-self.reg_alpha_a[1])*self.a[1] + self.reg_lambda_a[1]*self.reg_alpha_a[1]*self.subgradient(self.a[1])

            # gradient of W
            data_dep=-numpy.dot(numpy.sqrt(self.a[1])*Xbatch,Hbatch.transpose())/self.batch_size
            data_indep=-numpy.dot(numpy.sqrt(self.a[1])*XS,HS.transpose())/self.NS
            grad_W=0.5*(data_dep - data_indep) #+ self.reg_lambda_W*(1-self.reg_alpha_W)*self.W + self.reg_lambda_W*self.reg_alpha_W*self.subgradient(self.W)

            self.grad_a=[grad_a1,grad_a2]
            self.grad_W=grad_W
            
        elif self.visible_type=="Gamma":
            # gradient of a1
            data_dep=-numpy.mean(numpy.log(Xbatch),axis=1)
            data_indep=-numpy.mean(numpy.log(XS),axis=1)
            grad_a1=data_dep - data_indep
            grad_a1.shape=(self.M,1)
            #grad_a1=grad_a1 +  self.reg_lambda_a[0]*(1-self.reg_alpha_a[0])*self.a[0] + self.reg_lambda_a[0]*self.reg_alpha_a[0]*self.subgradient(self.a[0])

            # gradient of a2
            data_dep=-numpy.mean(Xbatch,axis=1)
            data_indep=-numpy.mean(XS,axis=1)
            grad_a2=data_dep - data_indep
            grad_a2.shape=(self.M,1)
            #grad_a2=grad_a2 +  self.reg_lambda_a[1]*(1-self.reg_alpha_a[1])*self.a[1] + self.reg_lambda_a[1]*self.reg_alpha_a[1]*self.subgradient(self.a[1])
            
            # gradient of W
            data_dep=-numpy.dot(Xbatch,Hbatch.transpose())/self.batch_size
            data_indep=-numpy.dot(XS,HS.transpose())/self.NS
            grad_W=data_dep - data_indep #+ self.reg_lambda_W*(1-self.reg_alpha_W)*self.W + self.reg_lambda_W*self.reg_alpha_W*self.subgradient(self.W)

            self.grad_a=[grad_a1,grad_a2]
            self.grad_W=grad_W
            
#            print(Hbatch.max())
#            print(Hbatch.min())
#            print(HS.max())
#            print(HS.min())
#            print(Xbatch.max())
#            print(Xbatch.min())
#            print(XS.max())
#            print(XS.min())
#            print(self.grad_a[0][0:10])
#            print(self.grad_a[1][0:10])
#            print(self.grad_W)
        
        # grad of b
        if self.hidden_type=="Bernoulli":
            data_dep=-numpy.mean(Hbatch,axis=1)
            data_indep=-numpy.mean(HS,axis=1)
            grad_b=data_dep - data_indep
            grad_b.shape=(self.K,1)
            #grad_b=grad_b + self.reg_lambda_b*(1-self.reg_alpha_b)*self.b + self.reg_lambda_b*self.reg_alpha_b*self.subgradient(self.b)
        elif self.hidden_type=="Gaussian":
            # gradident of b1
            data_dep=-numpy.mean(Hbatch,axis=1)
            data_indep=-numpy.mean(HS,axis=1)
            grad_b1=data_dep - data_indep
            grad_b1.shape=(self.K,1)
            # gradient of b2
            data_dep=-numpy.mean(Hbatch**2,axis=1)
            data_indep=-numpy.mean(HS**2,axis=1)
            grad_b2=data_dep - data_indep
            grad_b2.shape=(self.K,1)    
            grad_b=[grad_b1, grad_b2]
        self.grad_b=grad_b
        

    def update_param(self):
        tol=1e-8
        tol_negbin_min=-20
        tol_poisson_max=self.tol_poisson_max#16 #numpy.log(255)
        tol_gamma_min=1e-3
        tol_gamma_max=1e3
        if self.if_fix_vis_bias:
            fix_a_log_ind=self.fix_a_log_ind
            not_fix_a_log_ind=numpy.logical_not(fix_a_log_ind)
            not_fix_a_log_ind=numpy.array(not_fix_a_log_ind, dtype=int)
            not_fix_a_log_ind.shape=(len(not_fix_a_log_ind),1)
        # update a and W
        if self.visible_type=="Bernoulli" or self.visible_type=="Multinomial":
            if not self.if_fix_vis_bias:
                self.a=self.a - self.learn_rate_a * self.grad_a
            else: # fix some a
                self.a=self.a - self.learn_rate_a * (not_fix_a_log_ind*self.grad_a)
            self.W=self.W - self.learn_rate_W * self.grad_W
            #print("in update_param ...")
            #print(self.if_fix_vis_bias)
            #print(self.a[0:10])
            #print(self.W)
        elif self.visible_type=="Gaussian_FixPrecision1" or self.visible_type=="Gaussian_FixPrecision2":
            if not self.if_fix_vis_bias:
                self.a=self.a - self.learn_rate_a * self.grad_a
            else: # fix some a
                self.a=self.a - self.learn_rate_a * (not_fix_a_log_ind*self.grad_a)
            self.W=self.W - self.learn_rate_W * self.grad_W
            #self.a[self.a<0]=0
            #print("in update_param ...")
            #print(self.a[0:10])
            #print(self.W)
        elif self.visible_type=="Poisson":
            if not self.if_fix_vis_bias:
                self.a=self.a - self.learn_rate_a * self.grad_a
            else:
                self.a=self.a - self.learn_rate_a * (not_fix_a_log_ind*self.grad_a)
            self.a[self.a>tol_poisson_max]=tol_poisson_max
            self.W=self.W - self.learn_rate_W * self.grad_W
            #self.W[self.W>0]=0 # set W<=0 to avoid overflow
#            print("in update_param ...")
#            print(self.a[0:10])
#            print(self.b[0:10])
#            print(self.W)
        elif self.visible_type=="NegativeBinomial":
            #print "before update ..."
            #print self.a[0:10]
            #print self.grad_a
            if not self.if_fix_vis_bias:
                self.a=self.a - self.learn_rate_a * self.grad_a
            else:
                self.a=self.a - self.learn_rate_a * (not_fix_a_log_ind*self.grad_a)
            # a not too small, not positive,s [-20,0)
            self.a[self.a>=0]=-tol # project a to negative
            self.a[self.a<tol_negbin_min]=tol_negbin_min
            self.W=self.W - self.learn_rate_W * self.grad_W
            self.W[self.W>0]=0 # project W to negative
#            print("in update_param ...")
#            print(self.a[0:10])
#            print(self.W)
        elif self.visible_type=="Multinoulli":
            for m in range(self.M):
                if not self.if_fix_vis_bias:
                    self.a[m]=self.a[m] - self.learn_rate_a * self.grad_a[m]
                self.W[m]=self.W[m] - self.learn_rate_W * self.grad_W[m]
        elif self.visible_type=="Gaussian":
            if not self.if_fix_vis_bias:
                self.a[0]=self.a[0] - self.learn_rate_a[0] * self.grad_a[0]
                #self.a[0][self.a[0]<0]=0                
                self.a[1]=self.a[1] - self.learn_rate_a[1] * self.grad_a[1]
            else:
                self.a[0]=self.a[0] - self.learn_rate_a[0] * (not_fix_a_log_ind*self.grad_a[0]) # update mean
                self.a[1]=self.a[1] - self.learn_rate_a[1] * (not_fix_a_log_ind*self.grad_a[1])
            self.a[1][self.a[1]>=0]=-tol
            #self.a[1][self.a[1]<-50]=-50
            #self.a[1][self.a[1]<-numpy.pi]=-numpy.pi
            #self.a[1][self.a[1]<-20]=-20
            self.W=self.W - self.learn_rate_W * self.grad_W
            #self.W[self.W<0]=0
#            print("in update_param ...")
#            print(self.a[0][0:10])
#            print(self.a[1][0:10])
#            print(self.W)
        elif self.visible_type=="Gaussian_Hinton":
            if not self.if_fix_vis_bias:
                self.a[0]=self.a[0] - self.learn_rate_a[0] * self.grad_a[0]
                #self.a[0][self.a[0]<0]=0                
                self.a[1]=self.a[1] - self.learn_rate_a[1] * self.grad_a[1]
            else:
                self.a[0]=self.a[0] - self.learn_rate_a[0] * (not_fix_a_log_ind*self.grad_a[0]) # update mean
                self.a[1]=self.a[1] - self.learn_rate_a[1] * (not_fix_a_log_ind*self.grad_a[1]) # update precision
                self.a[1][self.a[1]<=0]=tol # precision>0
            self.W=self.W - self.learn_rate_W * self.grad_W
        elif self.visible_type=="Gamma":
            if not self.if_fix_vis_bias:
                self.a[0]=self.a[0] -self.learn_rate_a[0] * self.grad_a[0]
            else:
                self.a[0]=self.a[0] -self.learn_rate_a[0] * (not_fix_a_log_ind*self.grad_a[0])
                self.a[0][self.a[0]<1]= 1
                self.a[0][self.a[0]>tol_gamma_max]=tol_gamma_max
                self.a[1]=self.a[1] -self.learn_rate_a[1] * (not_fix_a_log_ind*self.grad_a[1])
                self.a[1][self.a[1]>=0]=-tol_gamma_min
                self.a[1][self.a[1]<-tol_gamma_max]=-tol_gamma_max
            self.W=self.W - self.learn_rate_W * self.grad_W
            self.W[self.W>0]=0
#            print("in update_param ...")
#            print(self.a[0][0:10])
#            print(self.a[1][0:10])
#            print(self.W)
            
        # update b
        if self.hidden_type in ["Bernoulli", "Binomial", "Multinomial", "Guassian_FixPrecision2"]:
            self.b=self.b - self.learn_rate_b * self.grad_b
        elif self.hidden_type=="Gaussian":
            self.b[0]=self.b[0] - self.learn_rate_b[0] * self.grad_b[0]
            self.b[1]=self.b[1] - self.learn_rate_b[1] * self.grad_b[1]
        else:
            print("Hidden type is not defined!")
        #print(self.b[0:10])
            
#        print self.a[0].transpose()
#        print self.a[1].transpose()
#        print self.W
#        print self.b.transpose()
#        print self.grad_W
#        print self.grad_b.transpose()


    def sample_minibatch(self, batch_size=20):
        ind_batch=self.rng.choice(self.N,size=batch_size,replace=False)
        if self.visible_type=="Multinoulli":
            Xbatch=[None]*self.M
            for m in range(self.M):
                Xbatch[m]=self.X[m][:,ind_batch]
                if self.batch_size==1:
                    Xbatch[m].shape=(self.Ms[m],1)
        else:
            Xbatch=self.X[:,ind_batch]
            if self.batch_size==1:
                Xbatch.shape=(self.M,1)
        return Xbatch


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

    
    def sample_h_given_x(self, X, b_hat=None, hidden_value_or_meanfield="value"):
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
    
        # compute b_hat if not given
        if b_hat is None:    
            if self.hidden_type=="Gaussian":
                b_hat=[None,None]
                if self.visible_type in ["Bernoulli","Binomial","Multinomial","Poisson","NegativeBinomial","Gaussian"]:
                    if self.tie_W_for_pretraining_DBM_bottom:
                        b_hat[0]=self.b[0] + 2*numpy.dot(self.W.transpose(),X)
                    else:
                        b_hat[0]=self.b[0] + numpy.dot(self.W.transpose(),X)
                    b_hat[1]=self.b[1]#copy.deepcopy(self.b[1])
                elif self.visible_type=="Multinoulli":
                    b_hat[0]=self.b[0]
                    for m in range(self.M):
                        if self.tie_W_for_pretraining_DBM_bottom:
                            b_hat[0]=b_hat[0] + 2*numpy.dot(self.W[m].transpose(),X[m])
                        else:
                            b_hat[0]=b_hat[0] + numpy.dot(self.W[m].transpose(),X[m])
                    b_hat[1]=self.b[1]#copy.deepcopy(self.b[1])
                    
            elif self.hidden_type=="Bernoulli":
                if self.visible_type in ["Bernoulli","Binomial","Multinomial","Poisson","NegativeBinomial","Gaussian"]:
                    if self.tie_W_for_pretraining_DBM_bottom:
                        b_hat=self.b + 2*numpy.dot(self.W.transpose(),X)
                    else:
                        b_hat=self.b + numpy.dot(self.W.transpose(),X)
                elif self.visible_type=="Gaussian_Hinton":
                    if self.tie_W_for_pretraining_DBM_bottom:
                        b_hat=self.b + 2*numpy.dot(self.W.transpose(),numpy.sqrt(self.a[1])*X) # in Hinton's model, X must be scaled
                    else: 
                        b_hat=self.b + numpy.dot(self.W.transpose(),numpy.sqrt(self.a[1])*X)
                elif self.visible_type=="Gaussian_FixPrecision1":
                    if self.tie_W_for_pretraining_DBM_bottom:
                        b_hat=self.b + 2*numpy.dot(self.W.transpose(),self.visible_type_fixed_param*X)
                    else:
                        b_hat=self.b + numpy.dot(self.W.transpose(),self.visible_type_fixed_param*X)
        
        # sampling
        if self.hidden_type=="Bernoulli":
            b_hat[b_hat<-200]=-200 # to avoid overflow
            P=cl.sigmoid(b_hat)
            HM=P # mean of hidden variables
            if hidden_value_or_meanfield=="value":
                H=cl.Bernoulli_sampling(P,rng=self.rng)
            elif hidden_value_or_meanfield=="meanfield":
                H=HM
        elif self.hidden_type=="Binomial":
            P=cl.sigmoid(b_hat) # probability
            HM=self.hidden_type_fixed_param*P # mean
            if hidden_value_or_meanfield=="value":
                # hidden_type_fixed_param is the number of trials
                H=cl.Binomial_sampling(self.hidden_type_fixed_param, P, rng=self.rng)
            elif hidden_value_or_meanfield=="meanfield":
                H=HM
        elif self.hidden_type=="Gaussian":
            HM=b_hat[0]/(-2*b_hat[1] + 1e-10) # to prevent overflow
            if hidden_value_or_meanfield=="value":
                H=cl.Gaussian_sampling(HM,-2*b_hat[1] + 1e-10, rng=self.rng)
            elif hidden_value_or_meanfield=="meanfield":
                H=HM
        elif self.hidden_type=="Multinomial":
            P=numpy.exp(b_hat) # probability
            P=cl.normalize_probability(P)
            HM=self.hidden_type_fixed_param*P # mean
            if hidden_value_or_meanfield=="value":
                # hidden_type_fixed_param is the number of trials
                H=cl.Multinomial_sampling(self.hidden_type_fixed_param, P=P, rng=self.rng)
            elif hidden_value_or_meanfield=="meanfield":
                H=HM
                
        return H,HM


    def sample_x_given_h(self, H, a_hat=None):
        """
        If a_hat is given, H will not be used.
        """
        #X=H # initialize X
        #num=H.shape[1]
        #for n in range(num):
            #a=numpy.copy(self.a)
            #a.shape=(self.M,)
            #x_prob=cl.sigmoid(a + numpy.dot(self.W,H[:,n]))
            #x=numpy.zeros(shape=(self.M,),dtype=int)
            #for m in range(self.M):
            #    x[m]=self.rng.binomial(n=1,h_prob[m],size=1)
            #X[:,m]=x
            
        if self.visible_type=="Bernoulli":
            if a_hat is None:
                if self.tie_W_for_pretraining_DBM_top:
                    a_hat=self.a + 2*numpy.dot(self.W,H)
                else:
                    a_hat=self.a + numpy.dot(self.W,H)
            P=cl.sigmoid( a_hat )
            XM=P
            X=cl.Bernoulli_sampling(P,rng=self.rng)
        elif self.visible_type=="Gaussian":
            if a_hat is None:
                a_hat=[None]*2
                if self.tie_W_for_pretraining_DBM_top:
                    a_hat[0]=self.a[0] + 2*numpy.dot(self.W,H)
                else:
                    a_hat[0]=self.a[0] + numpy.dot(self.W,H)
                a_hat[1]=self.a[1]
                
            XM=-a_hat[0]/(2*a_hat[1])
            P=None
            X=cl.Gaussian_sampling(XM,-2*a_hat[1],rng=self.rng)
        elif self.visible_type=="Gaussian_FixPrecision1":
            if a_hat is None:
                if self.tie_W_for_pretraining_DBM_top:
                    a_hat=self.a + 2*numpy.dot(self.W,H)
                else:
                    a_hat=self.a + numpy.dot(self.W,H)
            XM=a_hat
            P=None
            X=cl.Gaussian_sampling(XM,self.visible_type_fixed_param,rng=self.rng)
        elif self.visible_type=="Gaussian_FixPrecision2":
            if a_hat is None:
                if self.tie_W_for_pretraining_DBM_top:
                    a_hat=self.a + 2*numpy.dot(self.W,H)
                else:
                    a_hat=self.a + numpy.dot(self.W,H)
            XM=a_hat/self.visible_type_fixed_param
            P=None
            X=cl.Gaussian_sampling(XM,self.visible_type_fixed_param,rng=self.rng)
        elif self.visible_type=="Gaussian_Hinton":
            if a_hat is None:
                a_hat=[None]*2
                if self.tie_W_for_pretraining_DBM_top:
                    a_hat[0]=self.a[0] + 2/numpy.sqrt(self.a[1])*numpy.dot(self.W,H) # mean
                else:
                    a_hat[0]=self.a[0] + 1/numpy.sqrt(self.a[1])*numpy.dot(self.W,H) 
                a_hat[1]=self.a[1]
            XM=a_hat[0]
            P=None
            X=cl.Gaussian_sampling(a_hat[0],a_hat[1],rng=self.rng)
        elif self.visible_type=="Poisson":
            tol_poisson_max=self.tol_poisson_max
            if a_hat is None:
                if self.tie_W_for_pretraining_DBM_top:
                    a_hat=self.a + 2*numpy.dot(self.W,H)
                else:
                    a_hat=self.a + numpy.dot(self.W,H)
            
            a_hat[a_hat>tol_poisson_max]=tol_poisson_max
            XM=numpy.exp(a_hat) # a_hat here must be small, if very big, overflow problem will raise.
            P=None
            X=cl.Poisson_sampling(XM,rng=self.rng)
        elif self.visible_type=="NegativeBinomial": 
            tol_negbin_max=-1e-8
            tol_negbin_min=-100
            if a_hat is None:
                if self.tie_W_for_pretraining_DBM_top:
                    a_hat= self.a + 2*numpy.dot(self.W,H) 
                else:
                    a_hat= self.a + numpy.dot(self.W,H)
            a_hat[a_hat>=0]=tol_negbin_max
            a_hat[a_hat<tol_negbin_min]=tol_negbin_min
            P_failure=numpy.exp(a_hat)
            P=P_failure # also return the probability of failure
            P_success=1-P_failure
            #print "max: {}".format(numpy.max(P_failure))
            #print "min: {}".format(numpy.min(P_failure))
            XM=self.visible_type_fixed_param*(P_failure/P_success)
            X=cl.NegativeBinomial_sampling(K=self.visible_type_fixed_param,P=P_success,rng=self.rng)
        elif self.visible_type=="Multinomial":
            if a_hat is None:
                if self.tie_W_for_pretraining_DBM_top:
                    a_hat= self.a + 2*numpy.dot(self.W,H) 
                else:
                    a_hat= self.a + numpy.dot(self.W,H)
            P=numpy.exp(a_hat)
            P=cl.normalize_probability(P)
            #print "max: {}".format(numpy.max(P))
            #print "min: {}".format(numpy.min(P))
            XM=self.visible_type_fixed_param*(P)
            X=cl.Multinomial_sampling(N=self.visible_type_fixed_param,P=P,rng=self.rng)
        elif self.visible_type=="Multinoulli":
            P=[None]*self.M
            XM=[None]*self.M
            X=[None]*self.M
            for m in range(self.M):
                if a_hat is None:
                    if self.tie_W_for_pretraining_DBM_top:
                        a_hatm= self.a[m] + 2*numpy.dot(self.W[m],H) 
                    else:
                        a_hatm= self.a[m] + numpy.dot(self.W[m],H)
                else:
                    a_hatm=a_hat[m]
                P[m]=numpy.exp(a_hatm)
                P[m]=cl.normalize_probability(P[m])
                XM[m]=P[m]
                X[m]=cl.Multinomial_sampling(N=1,P=P[m],rng=self.rng)
        elif self.visible_type=="Gamma":
            if a_hat is None:
                a_hat=[None]*2
                a_hat[0]=self.a[0]
                if self.tie_W_for_pretraining_DBM_top:
                    a_hat[1]=self.a[1] + 2*numpy.dot(self.W,H)
                else:
                    a_hat[1]=self.a[1] + numpy.dot(self.W,H)
                
            XM=(a_hat[0]+1)/(-a_hat[1])
            P=None
            X=cl.Gamma_sampling(a_hat[0]+1, -a_hat[1], rng=self.rng)
            
        return X,XM,P


    def compute_reconstruction_error(self,X0=None):
        if X0 is None:
            X0=self.X
        H0,H0M=self.sample_h_given_x(X0)
        X1,X1M,_=self.sample_x_given_h(H0)
        if self.visible_type=="Multinoulli":
            self.rec_error=0
            for m in range(self.M):
                self.rec_error=self.rec_error+numpy.mean(numpy.abs(X1M[m]-X0[m]))
        else:
            self.rec_error=numpy.mean(numpy.abs(X1M-X0))
        return self.rec_error
    

    def compute_free_energy(self,X=None): ##################### need update for Gaussian visible type
        """
        Compute F(x). 
        """
        if X is None:
            X=self.X
        zs=self.zeta(theta=self.a, X=X, fixed_param=self.visible_type_fixed_param, distribution=self.visible_type)
        #print zs
        b_hat=self.compute_posterior_bais(self.b, self.W, X, visible_or_hidden="hidden", distribution=self.hidden_type, opposite_type=self.visible_type, opposite_fixed_param=self.visible_type_fixed_param)
        B=self.A(b_hat, fixed_param=self.hidden_type_fixed_param, distribution=self.hidden_type)
        fes=-zs - B
        #print B
        mfe=numpy.mean(fes) # average over N samples
        return mfe,fes

        
    def compute_free_energy_dif(self,X_train=None,X_validate=None):
        if X_train is None:
            X_train=self.X
        if X_validate is None:
            X_validate=self.X_validate
        mfe_train,_=self.compute_free_energy(X=X_train)
        mfe_validate,_=self.compute_free_energy(X=X_validate)
        #print "mfe_train:{0}, mfe_validate:{1}".format(mfe_train,mfe_validate)
        self.free_energy_dif=mfe_train - mfe_validate
        return self.free_energy_dif


    def reinit_a(self): #### finish this function later
        if self.visible_type=="Bernoulli":
            mean=numpy.mean(self.X, axis=1)
            mean.shape=(mean.size,1)
            var=numpy.var(self.X, axis=1)
            var.shape=(var.size,1)
            self.a=mean
        elif self.visible_type=="Gaussian":
            mean=numpy.mean(self.X, axis=1)
            mean.shape=(mean.size,1)
            var=numpy.var(self.X, axis=1)
            var.shape=(var.size,1)
            precision=1/(var+0.0001)
            #precision[precision>100]=100
            #precision[precision>numpy.pi]=numpy.pi
            precision[precision>10]=10
            self.a[0]=mean*precision
            self.a[1]=-0.5*precision
        elif self.visible_type=="Gaussian_FixPrecision1":
            mean=numpy.mean(self.X, axis=1)
            mean.shape=(mean.size,1)
            self.a=mean
            #print(self.a)
            var=100*numpy.var(self.X, axis=1)
            var.shape=(var.size,1)
            precision=1/var
            self.visible_type_fixed_param=precision
            self.visible_type_fixed_param[self.visible_type_fixed_param>1000]=1000
        elif self.visible_type=="Gaussian_FixPrecision2":
            var=100*numpy.var(self.X, axis=1)
            var.shape=(var.size,1)
            precision=1/var
            self.visible_type_fixed_param=precision
            self.visible_type_fixed_param[self.visible_type_fixed_param>1000]=1000
            mean=numpy.mean(self.X, axis=1)
            mean.shape=(mean.size,1)
            self.a=mean*self.visible_type_fixed_param
            #print(self.a)
        elif self.visible_type=="Poisson":
            mean=numpy.mean(self.X, axis=1)
            mean.shape=(mean.size,1)
            #var=numpy.var(self.X, axis=1)
            #var.shape=(var.size,1)
            self.a=numpy.log(mean)
            self.a[self.a<-2.3]=-2.3 # threshold log(0.1)
        elif self.visible_type=="NegativeBinomial":
            #max_X=numpy.max(self.X, axis=1)
            #max_X.shape=(max_X.size,1)
            #self.visible_type_fixed_param=max_X
            mean=numpy.mean(self.X, axis=1)
            mean.shape=(mean.size,1)
            #var=numpy.var(self.X, axis=1)
            #var.shape=(var.size,1)
            self.a=numpy.log(mean/(self.visible_type_fixed_param+mean))
            
        elif self.visible_type=="Multinoulli":
            for m in range(self.M):
                mean=numpy.mean(self.X[m], axis=1)
                mean.shape=(mean.size,1)
                #var=numpy.var(self.X[m], axis=1)
                #var.shape=(var.size,1)
                self.a[m]=numpy.log(mean/mean.sum())
        elif self.visible_type=="Multinomial":
            mean=numpy.mean(self.X, axis=1)
            mean.shape=(mean.size,1)
            #var=numpy.var(self.X, axis=1)
            #var.shape=(var.size,1)
            #self.a=numpy.log(mean/mean.sum())
            self.a=mean/mean.sum()
        elif self.visible_type=="Gamma":
            mean=numpy.mean(self.X+1, axis=1)
            mean.shape=(mean.size,1)
            var=numpy.var(self.X+1, axis=1)
            var.shape=(var.size,1)
            self.a[0]=mean**2/var - 1
            self.a[1]=-mean/var


    def train(self, X=None, X_validate=None, batch_size=10, pcdk=20, NS=20, maxiter=100, learn_rate_a=0.1, learn_rate_b=0.1, learn_rate_W=0.1, change_rate=0.9, adjust_change_rate_at=None, adjust_coef=1.02, reg_lambda_a=0, reg_alpha_a=0, reg_lambda_b=0, reg_alpha_b=0, reg_lambda_W=0, reg_alpha_W=0,  change_every_many_iters=10, init_chain_time=100, train_subset_size_for_compute_error=100, valid_subset_size_for_compute_error=100, track_reconstruct_error=True, track_free_energy=True, reinit_a_use_data_stat=False, if_plot_error_free_energy=False, dir_save="./", prefix="RBM", figwidth=5, figheight=3):
        """
        X: numpy 2d array of size M by N, each column is a sample. 
        If train_subset_size_for_compute_error and valid_subset_size_for_compute_error are Nones, use all available training and validation samples.
        """
        start_time=time.clock()
        print("training rbm ...")
        if self.visible_type=="Multinoulli": # convert to binary
            self.X=[None]*self.M
            if X_validate is not None:
                self.X_validate=[None]*self.M
            else:
                self.X_validate=None
            for m in range(self.M):
                Z,_=cl.membership_vector_to_indicator_matrix(X[m,:],z_unique=list(range(self.Ms[m])))
                self.X[m]=Z.transpose()
                if X_validate is not None:
                    Z,_=cl.membership_vector_to_indicator_matrix(X_validate[m,:],z_unique=list(range(self.Ms[m])))
                    self.X_validate[m]=Z.transpose()
            self.N=self.X[0].shape[1]
            if X_validate is not None:
                self.N_validate=self.X_validate[0].shape[1] # number of validation samples
        else: # not multinoulli variables
            self.X= X
            self.N=self.X.shape[1] # number of training samples
            self.X_validate=X_validate
            if X_validate is not None:
                self.N_validate=self.X_validate.shape[1] # number of validation samples
            
        self.batch_size=batch_size
        if self.batch_size>self.N:
            self.batch_size=1

        # re-initiate the bias term of visible variables using the statistics of data
        if reinit_a_use_data_stat:
            self.reinit_a()

        # initialize Markov chains
        print("initializing Markov chains...")
        _,_,_,_,_=self.pcd_sampling(NS=NS,pcdk=init_chain_time*pcdk,rand_init=False,init=True) # initialize pcd
        self.maxiter=maxiter
        self.learn_rate_a=learn_rate_a
        self.learn_rate_b=learn_rate_b
        self.learn_rate_W=learn_rate_W
        # regularization coefficient
        self.reg_lambda_a=reg_lambda_a
        self.reg_alpha_a=reg_alpha_a
        self.reg_lambda_b=reg_lambda_b
        self.reg_alpha_b=reg_alpha_b
        self.reg_lambda_W=reg_lambda_W
        self.reg_alpha_W=reg_alpha_W 

        self.rec_errors_train=[]
        self.rec_errors_valid=[]
        self.mfes_train=[]
        self.mfes_valid=[]
        
        for i in range(self.maxiter):
            # get mini-batch
            Xbatch=self.sample_minibatch(self.batch_size)
            Hbatch,_=self.sample_h_given_x(Xbatch)
            #print("in the training of RBM... Hbatch:")
            #print(Hbatch.sum(axis=0))
            #print(Hbatch)

            # pcd-k sampling
            XS,HS,_,_,_=self.pcd_sampling(pcdk,init=False)

            # cd-k sampling
            #_,_,XS,HS,_=self.pcd_sampling(pcdk=pcdk,X0=Xbatch,persistent=False,init=True) # use probabilities insead of binaries
            #self.NS=self.batch_size # for CD-k, they must be equal.
            
            #print("in the training of RBM... HS:")
            #print(HS.sum(axis=0))
            #print(HS)

            # compute gradient
            self.compute_gradient(Xbatch,Hbatch,XS,HS)
            # update parameters
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
            self.update_param()
            
            # compute reconstruction error of the training samples
            # sample some training samples, rather than use all training samples which is time-consuming
            if train_subset_size_for_compute_error is not None:
                train_subset_ind=self.rng.choice(numpy.arange(self.N,dtype=int),size=train_subset_size_for_compute_error)
                if self.visible_type=="Multinoulli":
                    X_train_subset=[None]*self.M
                    for m in range(self.M):
                        X_train_subset[m]=self.X[m][:,train_subset_ind]
                else:
                    X_train_subset=self.X[:,train_subset_ind]
                if track_reconstruct_error:
                    rec_error_train=self.compute_reconstruction_error(X0=X_train_subset)
                    self.rec_errors_train.append(rec_error_train)
                if track_free_energy:
                    mfe_train,_=self.compute_free_energy(X_train_subset)
                    self.mfes_train.append(mfe_train)
            else:
                if track_reconstruct_error:
                    rec_error_train=self.compute_reconstruction_error(X0=self.X)
                    self.rec_errors_train.append(rec_error_train)
                if track_free_energy:
                    mfe_train,_=self.compute_free_energy(self.X)
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
                        rec_error_valid=self.compute_reconstruction_error(X0=X_validate_subset)
                        self.rec_errors_valid.append(rec_error_valid)
                    if track_free_energy:
                        mfe_validate,_=self.compute_free_energy(X_validate_subset)
                        self.mfes_valid.append(mfe_validate)
                else:
                    if track_reconstruct_error:                    
                        rec_error_valid=self.compute_reconstruction_error(X0=self.X_validate)
                        self.rec_errors_valid.append(rec_error_valid)
                    if track_free_energy:
                        mfe_validate,_=self.compute_free_energy(self.X_validate)
                        self.mfes_valid.append(mfe_validate)
                # compute difference of free energy between training set and validation  set
                # the log-likelihood(train_set) - log-likelihood(validate_set) = F(validate_set) - F(train_set), the log-partition function, logZ is cancelled out
#                if track_reconstruct_error and track_free_energy:
#                    free_energy_dif=mfe_train - mfe_validate
#                    print "{0}-th iteration, learn_rate_W: {1}, train_rec_error: {2}, valid_rec_error: {3}, free_energy_train: {4}, free_energy_valid: {5}, free_energy_dif: {6}".format(i, self.learn_rate_W, rec_error_train, rec_error_valid, mfe_train, mfe_validate, free_energy_dif)
#                elif not track_reconstruct_error and track_free_energy:
#                    free_energy_dif=mfe_train - mfe_validate
#                    print "{0}-th iteration, learn_rate_W: {1}, free_energy_train: {2}, free_energy_valid: {3}, free_energy_dif: {4}".format(i, self.learn_rate_W, mfe_train, mfe_validate, free_energy_dif)
#                elif track_reconstruct_error and not track_free_energy:
#                    print "{0}-th iteration, learn_rate_W: {1}, train_rec_error: {2}, valid_rec_error: {3}".format(i, self.learn_rate_W, rec_error_train, rec_error_valid)
#                elif not track_reconstruct_error and not track_free_energy:
#                    print "{0}-th iteration, learn_rate_W: {1}".format(i, self.learn_rate_W)
#            else:
#                if track_reconstruct_error and track_free_energy:
#                    print "{0}-th iteration, learn_rate_W: {1}, train_rec_error: {2}, free_energy_train: {3}".format(i, self.learn_rate_W, rec_error_train, mfe_train)
#                elif not track_reconstruct_error and track_free_energy:
#                    print "{0}-th iteration, learn_rate_W: {1}, free_energy_train: {2}".format(i, self.learn_rate_W, mfe_train)
#                elif track_reconstruct_error and not track_free_energy:
#                    print "{0}-th iteration, learn_rate_W: {1}, train_rec_error: {2}".format(i, self.learn_rate_W, rec_error_train)
#                elif not track_reconstruct_error and not track_free_energy:
#                    print "{0}-th iteration, learn_rate_W: {1}".format(i, self.learn_rate_W)
        
            print("{0}-th iteration, learn_rate_W: {1}".format(i, self.learn_rate_W))
            if track_reconstruct_error:
                if self.X_validate is None:
                    print("train_rec_error: {0}".format(rec_error_train))
                else:
                    print("train_rec_error: {0}, valid_rec_error: {1}".format(rec_error_train, rec_error_valid))
            if track_free_energy:
                if self.X_validate is None:
                    print("free_energy_train:{0}".format(mfe_train))
                else:
                    free_energy_dif=mfe_train - mfe_validate
                    print("free_energy_train: {0}, free_energy_valid: {1}, free_energy_dif: {2}".format(mfe_train, mfe_validate, free_energy_dif))

        if if_plot_error_free_energy:
            self.plot_error_free_energy(dir_save, prefix, figwidth=5, figheight=3)
            
        print("The training of RBM is finished!")
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


#    def plot_error_free_energy(self, dir_save="./", prefix="RBM", mean_over=5, figwidth=5, figheight=3):
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
#            ax.plot(iters,self.rec_errors_train,linestyle="-", color="red", linewidth=0.5, label="RCE:Train")
#        if len(self.mfes_valid)>0:
#            ax.plot(iters,self.rec_errors_valid,linestyle=":",color="darkgoldenrod",linewidth=0.5, label="RCE:Test")
#        ax.set_xlabel("Iteration",fontsize=8)
#        ax.set_ylabel("Reconstruction Error (RCE)",color="red",fontsize=8)
#        for tl in ax.get_yticklabels():
#            tl.set_color("r")
#        plt.setp(ax.get_yticklabels(), fontsize=8)
#        plt.setp(ax.get_xticklabels(), fontsize=8)
#        ax.legend(loc="upper right",fontsize=8)
#            
#        #ax.legend(loc="lower left",fontsize=8)
#        if len(self.mfes_train)>0 or len(self.mfes_valid)>0:
#            ax2=ax.twinx()
#            if len(self.rec_errors_train)>0:
#                ax2.plot(iters,self.mfes_train,linestyle="-",color="blue",linewidth=0.5, label="FE:Train")
#            if len(self.rec_errors_valid)>0:
#                ax2.plot(iters,self.mfes_valid,linestyle=":",color="blueviolet",linewidth=0.5, label="FE:Test")
#            ax2.set_ylabel("Free Energy (FE)", color="blue",fontsize=8)
#            for tl in ax2.get_yticklabels():
#                tl.set_color("b")
#            plt.setp(ax2.get_yticklabels(), fontsize=8)
#            plt.setp(ax2.get_xticklabels(), fontsize=8)
#            # legend
#            ax.legend(loc="upper left",fontsize=8)
#            ax2.legend(loc="upper right",fontsize=8)
#            
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
        

    def get_param(self):
        """
        Get model parameters.
        """
        return self.a,self.b,self.W


    def set_param(self,a, b, W):
        """
        Set model parameters.
        """
        self.a=a
        self.b=b
        self.W=W


    def make_dir_save(self,parent_dir_save, prefix, learn_rate_a, learn_rate_b, learn_rate_W, reg_lambda_W, reg_alpha_W, visible_type_fixed_param, hidden_type_fixed_param, maxiter, normalization_method="None"):
        
        if self.visible_type=="Gaussian" or self.visible_type=="Gamma": 
            foldername=prefix + "_X"+self.visible_type+":" + str(self.M) + "_H" + self.hidden_type + ":" + str(self.K) + "_learnrateabW:" + str(learn_rate_a[0]) + "_" + str(learn_rate_a[1]) + "_" + str(learn_rate_b) + "_:" + str(learn_rate_W) + "_visfix:" + str(visible_type_fixed_param) + "_hidfix:" + str(hidden_type_fixed_param) + "_iter:" + str(maxiter) + "_norm:" + normalization_method + time.strftime("_%Y%b%d%X") + "/"
        else:
            foldername=prefix + "_X"+self.visible_type+":" + str(self.M) + "_H" + self.hidden_type + ":" + str(self.K) + "_learnrateabW:" + str(learn_rate_a) + "_" + str(learn_rate_b) + "_" + str(learn_rate_W) + "_visfix:" + str(visible_type_fixed_param) + "_hidfix:" + str(hidden_type_fixed_param) + "_iter:" + str(maxiter) + "_norm:" + normalization_method + time.strftime("_%Y%b%d%X") + "/"
        dir_save=parent_dir_save+foldername
        self.dir_save=dir_save
        try:
            os.makedirs(dir_save)
        except OSError:
            #self.dir_save=parent_dir_save
            pass
        print("The results will be saved in " + self.dir_save)
        return(self.dir_save)

    
    def save_sampling(self, XM, ifsort=True, dir_save="./", prefix="RBM"):
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
            numpy.savetxt(filename,XM_sorted, fmt="%.2f", delimiter="\t")
            filename=dir_save + prefix + "_features.txt"
            numpy.savetxt(filename,self.features, fmt="%s", delimiter="\t")


    def generate_samples(self, NS=100, sampling_time=4, reinit=False, pcdk=1000, rand_init=True, row=28, col=28, dir_save="./", prefix="CRBM"):
        """
        Use Gibbs sampling.
        """
        
        if reinit or NS!=self.NS:
            self.pcd_sampling(pcdk=pcdk, NS=NS, X0=None, persistent=True, rand_init=rand_init, init=True)
            
        for s in range(sampling_time):
            chainX,chainH,chainXM,chainHM,chain_length=self.pcd_sampling(pcdk=pcdk, init=False)
            
            # plot sampled data
            sample_set_x_3way=numpy.reshape(chainXM,newshape=(row,col,NS))
            cl.plot_image_subplots(dir_save+"fig_"+prefix+"_visibletype_"+self.visible_type +"_generated_samples_randinit_"+str(rand_init)+"_"+str(s)+".pdf", data=sample_set_x_3way, figwidth=6, figheight=6, colormap="gray", aspect="equal", num_col=int(numpy.ceil(numpy.sqrt(NS))), wspace=0.01, hspace=0.001)
            # plot ZM
            HM_3way=self.make_Z_matrix(chainHM)
            self.HM_3way=HM_3way
            cl.plot_image_subplots(dir_save+"fig_"+prefix+"_visibletype_"+self.visible_type +"_generated_samples_randinit_"+str(rand_init)+"_"+str(s)+"_HM.pdf", data=HM_3way, figwidth=6, figheight=6, colormap="cool", aspect="auto", origin="lower", interpolation="nearest", colorbar=True, num_col=int(numpy.ceil(numpy.sqrt(NS))), wspace=0.1, hspace=0.1)
            # save the Z code
            numpy.savetxt(dir_save+"fig_"+prefix+"_visibletype_"+self.visible_type +"_generated_samples_randinit_"+str(rand_init)+"_"+str(s)+"_ZM.txt", chainHM.transpose(),fmt="%.4f",delimiter="\t")
            numpy.savetxt(dir_save+"fig_"+prefix+"_visibletype_"+self.visible_type +"_generated_samples_randinit_"+str(rand_init)+"_"+str(s)+"_Z.txt", chainH.transpose(),fmt="%.4f",delimiter="\t")
            
            # sorted results
            ind=numpy.argsort(chainHM.sum(axis=1))
            ind=ind[::-1]
            chainHM_sorted=chainHM[ind,:]
            chainH_sorted=chainH[ind,:]
            HM_3way_sorted=self.make_Z_matrix(chainHM_sorted)
            cl.plot_image_subplots(dir_save+"fig_"+prefix+"_visibletype_"+self.visible_type +"_generated_samples_randinit_"+str(rand_init)+"_"+str(s)+"_HM_sorted.pdf", data=HM_3way_sorted, figwidth=6, figheight=6, colormap="cool", aspect="auto", origin="lower", interpolation="nearest", colorbar=True, num_col=int(numpy.ceil(numpy.sqrt(NS))), wspace=0.1, hspace=0.1)
            # save the Z code
            numpy.savetxt(dir_save+"fig_"+prefix+"_visibletype_"+self.visible_type +"_generated_samples_randinit_"+str(rand_init)+"_"+str(s)+"_ZM_sorted.txt", chainHM_sorted.transpose(),fmt="%.4f",delimiter="\t")
            numpy.savetxt(dir_save+"fig_"+prefix+"_visibletype_"+self.visible_type +"_generated_samples_randinit_"+str(rand_init)+"_"+str(s)+"_Z_sorted.txt", chainH_sorted.transpose(),fmt="%.4f",delimiter="\t")
        return chainHM,chainHM_sorted
        
        
    def generate_samples_given_h(self, H0=None, NS=100, sampling_time=4, row=28, col=28, dir_save="./", prefix="RBM"):
        for s in range(sampling_time):
            X,XM,_=self.sample_x_given_h(H0)
            # plot sampled data
            sample_set_x_3way=numpy.reshape(XM,newshape=(row,col,NS))
            cl.plot_image_subplots(dir_save+"fig_"+prefix+"_visibletype_"+self.visible_type +"_generated_samples_given_z_"+str(s)+".pdf", data=sample_set_x_3way, figwidth=6, figheight=6, colormap="gray", aspect="equal", num_col=int(numpy.ceil(numpy.sqrt(NS))), wspace=0.01, hspace=0.001)
        

    def make_Z_matrix(self, Z):
        ZT=numpy.reshape(Z,newshape=(1,Z.shape[0],Z.shape[1]))
        return ZT
        

    def generate_samples_given_x(self, train_set_x_sub=None, dir_save="./", prefix="RBM"):        
        NS=train_set_x_sub.shape[1]
        H,HM=self.sample_h_given_x(train_set_x_sub)
        HM_3way=self.make_Z_matrix(HM)
        self.HM_3way=HM_3way
        cl.plot_image_subplots(dir_save+"fig_"+prefix+"_visibletype_"+self.visible_type +"_given_sample_generate_HM.pdf", data=HM_3way, figwidth=6, figheight=6, colormap="cool", aspect="auto", origin="lower", interpolation="nearest", colorbar=True, num_col=int(numpy.ceil(numpy.sqrt(NS))), wspace=0.1, hspace=0.1)
        # save the Z code
        numpy.savetxt(dir_save+"fig_"+prefix+"_visibletype_"+self.visible_type +"_given_sample_generate_HM.txt", HM.transpose(),fmt="%.4f",delimiter="\t")
        
        # sorted results
        ind=numpy.argsort(HM.sum(axis=1))
        ind=ind[::-1]
        HM_sorted=HM[ind,:]
        HM_3way_sorted=self.make_Z_matrix(HM_sorted)
        cl.plot_image_subplots(dir_save+"fig_"+prefix+"_visibletype_"+self.visible_type +"_given_sample_generate_HM_sorted.pdf", data=HM_3way_sorted, figwidth=6, figheight=6, colormap="cool", aspect="auto", origin="lower", interpolation="nearest", colorbar=True, num_col=int(numpy.ceil(numpy.sqrt(NS))), wspace=0.1, hspace=0.1)
        numpy.savetxt(dir_save+"fig_"+prefix+"_visibletype_"+self.visible_type +"_given_sample_generate_HM_sorted.txt", HM_sorted.transpose(),fmt="%.4f",delimiter="\t")
        return HM,HM_sorted
        
        
        