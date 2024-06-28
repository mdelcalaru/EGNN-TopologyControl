import numpy as np
import cvxpy as cp


class MNF_share_solver():
  def __init__(self, task_config, comm_config, channel, Kopts):
    # Get parameters. Mainly a channel and Network and Task config.
    self.cm = channel
    self.x_task=task_config
    self.x_comm = comm_config
    self.config=np.vstack((task_config,comm_config))
    self.rate, _ = self.cm.predict(self.config)
    self.adjacency = self.cm.adjacency(self.config)
    self.N=self.config.shape[0]
    self.M=task_config.shape[0]
    self.K=self.M*(self.M-1)
    self.Kopts=Kopts
  
  def update_channel(self, task_config, comm_config):
    self.x_task = task_config
    self.x_comm = comm_config
    self.config = np.vstack((task_config,comm_config))
    self.rate, _ = self.cm.predict(self.config)
    self.adjacency = self.cm.adjacency(self.config)

  def solver(self):#,task_config, comm_config):
    #self.update_channel(task_config, comm_config)
    # Define Variables for problem
    rs = cp.Variable(self.N*self.N*self.K,nonneg=True)
    ai = cp.Variable((self.M,self.M))
    Tau = cp.Variable((self.N,self.N),nonneg=True)
    A = cp.Variable((self.N,self.K))   
    mapK=[]
    for i in range(self.M):
        for j in range(i+1,self.M):
            mapK.append([i,j])

    for i in range(1, self.M):
        for j in range(i):
            if i!=j:
                mapK.append([i,j])

    indn=np.arange(self.K)          
    mapK=np.array(mapK)

    indx_AtoZero=np.ones((self.N,self.K),dtype=bool)
    indx_AtoZero[mapK[:,0],self.Kopts]=False
    indx_AtoZero[mapK[:,1],self.Kopts]=False

    ai_mask=np.zeros((self.M,self.M), dtype=bool)
    ai_mask[mapK[self.Kopts,0],mapK[self.Kopts,1]]=True	

    # Define funcion to max
    Ce  = cp.sum(cp.log(ai[ai_mask]))/(self.M*(self.M-1))


    # CONSTRAINS 
    constraints = []
    for k in range(self.K):
      r_Ks=cp.reshape(rs[self.N*self.N*k:self.N*self.N*(k+1)], (self.N, self.N),order='C')  
      constraints +=[A[:,k] == cp.sum(r_Ks,axis=1)-cp.sum(r_Ks,axis=0)]  #Es una matriz NxK
      
     
    mu2_index0=len(constraints)
    constraints +=[cp.sum(cp.multiply(self.adjacency,Tau),axis=1)+self.adjacency@cp.sum(cp.multiply(self.adjacency,Tau),axis=1) <=1]

    B=cp.reshape(rs[0:self.N*self.N], (self.N, self.N),order='C')
    for k in range(1,self.K):
      B+=cp.reshape(rs[self.N*self.N*k:self.N*self.N*(k+1)], (self.N, self.N),order='C')
    
    constraints += [cp.multiply(self.rate,Tau) >= B ]
    
    constraints += [ai[mapK[:,0],mapK[:,1]] <= A[mapK[:,0],indn] ]
    constraints += [-ai[mapK[:,0],mapK[:,1]] <= A[mapK[:,1],indn] ]
    
    constraints += [A[indx_AtoZero] == 0]
    constraints += [cp.diag(ai) == 1]
    constraints += [Tau[np.where(self.adjacency==0)] == 0]

    #constraints += [Tau <= 1]
    constraints += [cp.diag(Tau) == 0]

    # Define problem and solve
    prob = cp.Problem(cp.Maximize(Ce), constraints)
    #prob.solve(solver=cp.MOSEK,verbose=False, mosek_params={'MSK_IPAR_INTPNT_SOLVE_FORM':   'MSK_SOLVE_DUAL'})
    prob.solve(solver=cp.SCS)
    #prob.solve()
    #print("mu_=, %s" %constraints[0].dual_value.reshape((self.N,self.N)))
    #print("la_{1}=, %s" %constraints[1].dual_value)
    #print("mu_{2}=, %s" %constraints[19])
    return Ce.value ,rs.value, ai.value, Tau.value, prob.status
