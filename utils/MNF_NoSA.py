import numpy as np
import random
import math
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
import time
from scipy.special import softmax

class Agent:
    def __init__(self,max_radius = 1,pos=None,seed=None):
      if pos is None:
        self.x =  max_radius * (np.random.uniform() )
        self.y =  max_radius * (np.random.uniform())
      else:
        self.x = pos[0]
        self.y = pos[1]

    def update_position(self, delta_pos):
        self.x += delta_pos[0]
        self.y += delta_pos[1]

class Swarm:
    def __init__(self,num_agents, pos=None,max_radius = 1,seed=None):
        self.num_agents = num_agents
        self.agents = []
        if seed != None:
          np.random.seed(seed)
        for i in range(num_agents):
          if pos is None:
            self.agents.append(Agent(max_radius=max_radius))
          else:
            self.agents.append(Agent(max_radius=max_radius,pos=pos[i]))

    def get_positions(self):
      positions = []
      for agent in self.agents:
        positions.append([agent.x,agent.y])
      return np.array(positions)

    def update_positions_swarm(self,delta_pos):
      for i, agent in enumerate(self.agents):
        agent.update_position(delta_pos[i])
      return

class Scenario:
    def __init__(self, num_task_agents, num_net_agents,alpha_0=0.1,TA_pos=None,NA_pos=None,agent_density = 1, seed_ta=None, seed_na=None,d_max=1):
        self.d_max = d_max
        self.N = num_task_agents  + num_net_agents
        self.max_radius = np.sqrt(self.N / (agent_density))
        self.task_agents = Swarm(num_task_agents,max_radius = self.max_radius,seed=seed_ta,pos=TA_pos)
        self.net_agents = Swarm(num_net_agents, max_radius = self.max_radius,seed=seed_na,pos=NA_pos)
        self.num_task_agents = num_task_agents
        self.num_net_agents = num_net_agents
        self.capacity_matrix = np.zeros((num_task_agents + num_net_agents, num_task_agents + num_net_agents))
        self.get_capacity()
        self.alpha = alpha_0

    def plot_agents(self,axis=None,title=None):
      if axis == None:
        plt.figure(figsize=(8, 6))
        for i, (x, y) in enumerate(zip(self.task_agents.get_positions()[:,0], self.task_agents.get_positions()[:,1])):
          plt.text(x + 0.05, y + 0.05, str(i + 1),color="blue")
        for i, (x, y) in enumerate(zip(self.net_agents.get_positions()[:,0], self.net_agents.get_positions()[:,1])):
          plt.text(x + 0.05, y + 0.05, str(i + 1),color="red")
        plt.scatter(self.task_agents.get_positions()[:,0], self.task_agents.get_positions()[:,1], s=150, color='blue', marker='o', label='Task Agent')
        plt.scatter(self.net_agents.get_positions()[:,0], self.net_agents.get_positions()[:,1], s=150, color='red', marker='^', label='Net Agent')
        circle_radius = self.d_max  # Adjust the radius as needed
        for xi, yi in zip(self.task_agents.get_positions()[:,0], self.task_agents.get_positions()[:,1]):
            circle = plt.Circle((xi, yi), circle_radius, color='blue', linewidth=2,alpha=0.03)
            plt.gca().add_patch(circle)
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        if title==None:
          plt.title('Agent Positions')
        else:
          plt.title(title)
        plt.xlim(0,self.max_radius)
        plt.ylim(0,self.max_radius)
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.show()
      else:
        for i, (x, y) in enumerate(zip(self.task_agents.get_positions()[:,0], self.task_agents.get_positions()[:,1])):
          axis.text(x + 0.05, y + 0.05, str(i + 1),color="blue")
        for i, (x, y) in enumerate(zip(self.net_agents.get_positions()[:,0], self.net_agents.get_positions()[:,1])):
          axis.text(x + 0.05, y + 0.05, str(i + 1),color="red")
        axis.scatter(self.task_agents.get_positions()[:,0], self.task_agents.get_positions()[:,1],s=150,  color='blue', marker='o', label='Task Agent')
        axis.scatter(self.net_agents.get_positions()[:,0], self.net_agents.get_positions()[:,1], s=150, color='red', marker='^', label='Net Agent')
        circle_radius = self.d_max  # Adjust the radius as needed
        for xi, yi in zip(self.task_agents.get_positions()[:,0], self.task_agents.get_positions()[:,1]):
            circle = plt.Circle((xi, yi), circle_radius, color='blue', linewidth=2,alpha=0.03)
            axis.add_patch(circle)

        axis.set_xlabel('X-axis')
        axis.set_ylabel('Y-axis')
        if title==None:
          axis.set_title('Agent Positions')
        else:
          axis.set_title(title)
        axis.set_xlim(0,self.max_radius)
        axis.set_ylim(0,self.max_radius)
        axis.legend(loc='best')
        axis.grid(True, alpha=0.3)


    def plot_commodity_w_rates(self,k,r,axis):
      k = k-1
      positions = np.concatenate([self.task_agents.get_positions(),self.net_agents.get_positions()])
      for i in range(self.N):
        for j in range(self.N):
          x1, y1 = positions[i]
          x2, y2 = positions[j]
          dx = x2 - x1
          dy = y2 - y1
          arrow_width = r[k,i,j]
          if arrow_width > 0.02:
            axis.arrow(x1,y1,dx,dy,head_width=0.06,head_length=0.06,fc='black',ec='black',length_includes_head=True,alpha=arrow_width, lw=0.5)
      for i, (x, y) in enumerate(zip(self.task_agents.get_positions()[:,0], self.task_agents.get_positions()[:,1])):
        axis.text(x + 0.05, y + 0.05, f'{i + 1} / {(np.sum(r,axis=2).T - np.sum(r,axis=1).T)[i,k]:.2f}',color="blue")
      for i, (x, y) in enumerate(zip(self.net_agents.get_positions()[:,0], self.net_agents.get_positions()[:,1])):
        axis.text(x + 0.05, y + 0.05, f'{self.num_task_agents + i + 1} / {(np.sum(r,axis=2).T - np.sum(r,axis=1).T)[self.num_task_agents + i,k]:.2f}',color="red")
      axis.scatter(self.task_agents.get_positions()[:,0], self.task_agents.get_positions()[:,1], s=150, color='blue', marker='o', label='Task Agent')
      axis.scatter(self.net_agents.get_positions()[:,0], self.net_agents.get_positions()[:,1],s=150,  color='red', marker='^', label='Net Agent')
      axis.set_xlabel('X-axis')
      axis.set_ylabel('Y-axis')
      axis.set_xlim(0,self.max_radius)
      axis.set_ylim(0,self.max_radius)
      axis.legend(loc='best')
      axis.grid(True, alpha=0.3)
      return

    def plot_agents_w_sink(self,sink,r,axis):
      sink = sink-1
      if sink != -1:
        axis.scatter(self.task_agents.get_positions()[:sink,0], self.task_agents.get_positions()[:sink,1], s=150, color='blue', marker='o', label='Task Agent')
        axis.scatter(self.task_agents.get_positions()[sink:,0], self.task_agents.get_positions()[sink:,1], s=150, color='blue', marker='o', label='Task Agent')
        axis.scatter(self.task_agents.get_positions()[sink,0], self.task_agents.get_positions()[sink,1], s=150, color='violet', marker='D', label='Sink Agent')
      else:
        axis.scatter(self.task_agents.get_positions()[:,0], self.task_agents.get_positions()[:,1], s=150, color='blue', marker='o', label='Task Agent')
      axis.scatter(self.net_agents.get_positions()[:,0], self.net_agents.get_positions()[:,1],s=150,  color='red', marker='^', label='Net Agent')
      axis.set_xlabel('X-axis')
      axis.set_ylabel('Y-axis')
      axis.set_xlim(0,self.max_radius)
      axis.set_ylim(0,self.max_radius)
      axis.legend(loc='best')
      axis.grid(True, alpha=0.3)
      return



    def plot_agents_w_rates(self,r,commodities,save_path=None):
        nmbr_ax_y = int(np.sqrt(commodities))
        nmbr_ax_x = int(np.ceil(commodities/nmbr_ax_y))
        fig, ax = plt.subplots(nmbr_ax_x,nmbr_ax_y)
        fig.set_size_inches((int(7*nmbr_ax_y),int(7*nmbr_ax_x)))
        positions = np.concatenate([self.task_agents.get_positions(),self.net_agents.get_positions()])
        for axx in range(nmbr_ax_x):
          for axy in range(nmbr_ax_y):
            idx = int(nmbr_ax_y*axx + axy)
            for i in range(self.N):
              for j in range(self.N):
                x1, y1 = positions[i]
                x2, y2 = positions[j]
                dx = x2 - x1
                dy = y2 - y1
                arrow_width = r[idx,i,j]
                if arrow_width > 0.02:
                  if nmbr_ax_y > 1:
                    ax[axx,axy].arrow(x1,y1,dx,dy,head_width=0.06,head_length=0.06,fc='black',ec='black',length_includes_head=True,alpha=arrow_width, lw=0.5)
                  else:
                    ax[axx].arrow(x1,y1,dx,dy,head_width=0.06,head_length=0.06,fc='black',ec='black',length_includes_head=True,alpha=arrow_width, lw=0.5)
            if nmbr_ax_y > 1 :
              for i, (x, y) in enumerate(zip(self.task_agents.get_positions()[:,0], self.task_agents.get_positions()[:,1])):
                ax[axx,axy].text(x + 0.05, y + 0.05, f'{i + 1} / {(np.sum(r,axis=2).T - np.sum(r,axis=1).T)[i,idx]:.2f}',color="blue")
              for i, (x, y) in enumerate(zip(self.net_agents.get_positions()[:,0], self.net_agents.get_positions()[:,1])):
                ax[axx,axy].text(x + 0.05, y + 0.05, f'{self.num_task_agents + i + 1} / {(np.sum(r,axis=2).T - np.sum(r,axis=1).T)[self.num_task_agents + i,idx]:.2f}',color="red")
              ax[axx,axy].scatter(self.task_agents.get_positions()[:,0], self.task_agents.get_positions()[:,1], s=150, color='blue', marker='o', label='Task Agent')
              ax[axx,axy].scatter(self.net_agents.get_positions()[:,0], self.net_agents.get_positions()[:,1],s=150,  color='red', marker='^', label='Net Agent')
              ax[axx,axy].set_xlabel('X-axis')
              ax[axx,axy].set_ylabel('Y-axis')
              ax[axx,axy].set_title(f'Commodity {idx + 1}')
              ax[axx,axy].set_xlim(0,self.max_radius)
              ax[axx,axy].set_ylim(0,self.max_radius)
              ax[axx,axy].legend(loc='best')
              ax[axx,axy].grid(True, alpha=0.3)
            else:
              for i, (x, y) in enumerate(zip(self.task_agents.get_positions()[:,0], self.task_agents.get_positions()[:,1])):
                ax[axx].text(x + 0.05, y + 0.05, f'{i + 1} / {(np.sum(r,axis=2).T - np.sum(r,axis=1).T)[i,idx]:.2f}',color="blue")
              for i, (x, y) in enumerate(zip(self.net_agents.get_positions()[:,0], self.net_agents.get_positions()[:,1])):
                ax[axx].text(x + 0.05, y + 0.05, f'{self.num_task_agents + i + 1} / {(np.sum(r,axis=2).T - np.sum(r,axis=1).T)[self.num_task_agents + i,idx]:.2f}',color="red")
              ax[axx].scatter(self.task_agents.get_positions()[:,0], self.task_agents.get_positions()[:,1],s=150,  color='blue', marker='o', label='Task Agent')
              ax[axx].scatter(self.net_agents.get_positions()[:,0], self.net_agents.get_positions()[:,1],s=150,  color='red', marker='^', label='Net Agent')
              ax[axx].set_xlabel('X-axis')
              ax[axx].set_ylabel('Y-axis')
              ax[axx].set_title(f'Commodity {idx + 1}')
              ax[axx].set_xlim(0,self.max_radius)
              ax[axx].set_ylim(0,self.max_radius)
              ax[axx].legend(loc='best')
              ax[axx].grid(True, alpha=0.3)
        if save_path!=None:
          plt.savefig("case_1_comms.pdf")
        plt.show()

    def get_capacity(self):  
      ta_positions = self.task_agents.get_positions()
      na_positions = self.net_agents.get_positions()
      positions = np.concatenate([ta_positions, na_positions])
      dist = distance_matrix(positions,positions)
      capacity_matrix = np.exp(-(dist/self.d_max)**4)
      np.fill_diagonal(capacity_matrix,0)
      # capacity_matrix[0:self.num_task_agents,0:self.num_task_agents] = 0
      self.capacity_matrix = capacity_matrix
      return capacity_matrix

    def get_grad_capacity(self):
      ta_positions = self.task_agents.get_positions()
      na_positions = self.net_agents.get_positions()
      positions = np.concatenate([ta_positions, na_positions])
      positions_mat = np.tile(positions,self.N).reshape(self.N,self.N,2)
      dist = distance_matrix(positions,positions)
      #exponent_matrix = (4/self.d_max)*np.exp(-(dist/self.d_max)**4).reshape(self.N,self.N,1)
      #grad_matrix = exponent_matrix * (positions_mat - positions_mat.transpose(1,0,2))
      exponent_matrix = (4/10**4)*np.exp(-(dist/10)**4).reshape(self.N,self.N,1)
      grad_matrix = (exponent_matrix * (positions_mat - positions_mat.transpose(1,0,2))**3)
      return grad_matrix

    def update_positions(self,mu):
      grad_mat = self.get_grad_capacity()
      mu_mat = mu.reshape(self.N,self.N,1) + mu.reshape(self.N,self.N,1).transpose(1,0,2)
      delta_pos_mat = grad_mat * mu_mat
      delta_pos_arr = -self.alpha * delta_pos_mat.sum(axis=1)
      self.net_agents.update_positions_swarm(delta_pos_arr[self.num_task_agents:self.N])
      return


class Optimization:
    def __init__(self, num_task_agents, num_net_agents,beta_1=0.1,beta_2=0.1,agent_density = 0.3,l2_penalty=0.01,seed_opt=16,fairness=1,weights=None,custom_comms=False,mask_reles=None,mask_sinks=None,mask_sources=None,commodities=None):
        np.random.seed(seed_opt)
        self.l2_penalty = l2_penalty
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.N = num_task_agents + num_net_agents
        self.num_net_agents = num_net_agents
        self.num_task_agents = num_task_agents
        if not custom_comms:
          self.commodities = num_task_agents
          self.generate_masks()
        else:
          self.commodities = commodities
          self.mask_sinks = mask_sinks
          self.mask_reles = mask_reles
          self.mask_sources = mask_sources
        self.r = np.random.rand(self.commodities,self.N,self.N)
        self.a = np.random.rand(self.N,self.commodities)
        self.lbda = np.random.rand(self.N,self.commodities) + 0.1
        self.mu = np.random.rand(self.N,self.N)
        self.capacity = np.random.rand(self.N,self.N)
        self.f = fairness
        if weights is None:
          self.weights=np.ones((self.N,self.commodities))
        else:
          self.weights=weights

    def generate_masks(self):
      self.mask_sources = np.zeros((self.N,self.commodities))
      self.mask_sinks = np.zeros((self.N,self.commodities))
      self.mask_reles = np.zeros((self.N,self.commodities))
      for i in range(self.N):
        for j in range(self.commodities):
          if i >= self.num_task_agents:
            self.mask_reles[i,j] = 1
          else:
            if i==j:
              self.mask_sinks[i,j] = 1
            else:
              self.mask_sources[i,j] = 1
      return

    def evaluate_primal(self):
      a_sources = self.a[self.mask_sources==1]
      l2_pen = np.sum(self.r**2) * self.l2_penalty
      if self.f == 1:
        primal = np.sum(np.log(a_sources)) - l2_pen
      else:
        primal =  np.sum(a_sources**(1-self.f)/(1-self.f)) - l2_pen
      return primal

    def evaluate_lagrangian(self):
      a_sources = self.a[self.mask_sources==1]
      l2_pen = np.sum(self.r**2) * self.l2_penalty
      c1 = np.sum(self.mu * (np.sum(self.r,axis=0)-self.capacity))
      c2_sources = np.sum((self.lbda * (self.a - (np.sum(self.r,axis=2).T - np.sum(self.r,axis=1).T)))[self.mask_sources==1])
      c2_reles = np.sum((self.lbda * (self.a - (np.sum(self.r,axis=2).T - np.sum(self.r,axis=1).T)))[self.mask_reles==1])
      if self.f == 1:
        lag = np.sum(np.log(a_sources)) - c1 - c2_sources- c2_reles -l2_pen
      else:
        lag = np.sum(a_sources**(1-self.f)/(1-self.f)) - c1 - c2_sources- c2_reles -l2_pen
      return lag

    def update_a(self):
        self.a[self.mask_reles==1] = 0
        self.a[(self.mask_sources==1) & (self.weights==0)]= (np.sum(self.r,axis=2).T - np.sum(self.r,axis=1).T)[(self.mask_sources==1) & (self.weights==0)]
        self.a[(self.mask_sources==1) & (self.weights!=0)] = np.clip(1 / ((self.lbda[(self.mask_sources==1) & (self.weights!=0)])**(1/self.f))  / self.weights[(self.mask_sources==1) & (self.weights!=0)],0,self.N)
        self.a[self.mask_sinks==1] = (np.sum(self.r,axis=2).T - np.sum(self.r,axis=1).T)[self.mask_sinks==1]
        return

    def update_r(self):
        mu_mtrx = np.tile(self.mu,(self.commodities,1,1))
        lbda_mtrx = np.tile(self.lbda, (self.N,1,1))
        r_linear = (-mu_mtrx + lbda_mtrx.transpose((2,1,0)) - lbda_mtrx.transpose((2,0,1)))/ (2*self.l2_penalty)
        self.r = np.clip(r_linear,0, 1)
        return


    def update_lbda(self):
        self.lbda -= self.beta_1* ((np.sum(self.r,axis=2).T - np.sum(self.r,axis=1).T) - self.a)
        self.lbda[self.mask_sources==1] = np.clip(self.lbda[self.mask_sources==1],0 , None)
        self.lbda[self.mask_sinks==1] = 0
        return

    def update_mu(self):
        self.mu -= self.beta_2*(self.capacity-np.sum(self.r,axis=0))
        self.mu = np.clip(self.mu, 0 , None)
        np.fill_diagonal(self.mu,0)
        return

    def plot_stats(self):
      print("Function Values")
      print("Objetive")
      print(self.evaluate_primal())
      print("Lagrangian")
      print(self.evaluate_lagrangian())

      print("Primal Variables")
      print("r")
      print(self.r)
      print("a")
      print(self.a)

      print("Dual Variables")
      print("Lambda")
      print(self.lbda)
      print("Mu")
      print(self.mu)

      print("Constraint slack")
      print("Maximum Commodity Generated/Consumed ")
      print(np.sum(self.r,axis=2).T - np.sum(self.r,axis=1).T)
      print("Capacity of Links ")
      print(self.capacity)
      print(" Link Usage ")
      print(np.sum(self.r,axis=0))

class Simulation:
    def __init__(self, num_task_agents=None, num_net_agents=None,task_agents_init=None,net_agents_init=None,beta_1=0.1,beta_2=0.1,alpha_0=0.1,fairness=1,weights=None,agent_density = 1,seed_ta=None, seed_na = None,seed_opt=16,nmbr_iterations=1000,gradient_ascent_steps=1,l2_penalty=0.01,NA_pos_0=None,TA_pos_0=None,d_max=1,decreasing_alpha=True,plot_partial=False,plot_final=True,opt_func=None,idx_sink=0,custom_comms=False,mask_reles=None,mask_sinks=None,mask_sources=None,commodities=None):

        self.plot_partial = plot_partial
        self.decreasing_alpha = decreasing_alpha
        self.nmbr_iterations = nmbr_iterations
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.alpha_0 = alpha_0
        self.l2_penalty = l2_penalty
        self.opt_func = opt_func
        self.plot_final = plot_final
        self.scenario = Scenario(num_task_agents=num_task_agents, num_net_agents=num_net_agents,alpha_0=alpha_0,TA_pos=TA_pos_0,NA_pos=NA_pos_0,agent_density=agent_density,seed_ta=seed_ta, seed_na=seed_na,d_max=d_max)

        # weights = distance_matrix(sim_fair_1.scenario.task_agents.get_positions(),sim_fair_1.scenario.task_agents.get_positions())**2
        # weights[weights!=0] = 1/weights[weights!=0]
        # weights = softmax(weights,axis=1)
        # #weights[weights<0.05] = 0
        # zeros_array = np.zeros((num_net_agents, weights.shape[1]))
        # weights = np.vstack((weights, zeros_array))

        # weights = distance_matrix(sim_fair_1.scenario.task_agents.get_positions(),sim_fair_1.scenario.task_agents.get_positions())
        # weights[weights<1] = 0
        # weights[weights>1] = 1
        # zeros_array = np.zeros((num_net_agents, weights.shape[1]))
        # weights = np.vstack((weights, zeros_array))

        self.opt = Optimization(num_task_agents, num_net_agents,beta_1,beta_2,alpha_0,l2_penalty,seed_opt,fairness=fairness,weights=weights,custom_comms=custom_comms,mask_reles=mask_reles,mask_sinks=mask_sinks,mask_sources=mask_sources,commodities=commodities)
        self.opt.capacity = self.scenario.get_capacity()
        self.grad_history = []
        self.mu_history = []
        self.r_change_history = []
        self.optimal_mu_history = []
        self.NA_pos_history = []
        self.TA_pos_history = []
        self.gradient_ascent_steps = gradient_ascent_steps
        self.alpha = alpha_0
        self.alpha_0 = alpha_0
        self.obj_hist = []
        self.lag_hist = []
        self.unf_1 = []
        self.unf_2 = []
        self.unf_3 = []
        self.alpha_history = []
        self.iter_time = []
        self.decreasing_constant=0

    def step(self):
        for i in range(self.gradient_ascent_steps):
          self.opt.update_a()
          self.opt.update_r()
          self.opt.update_lbda()
          self.opt.update_mu()
        self.append_records()
        self.scenario.update_positions(self.opt.mu)
        self.opt.capacity = self.scenario.get_capacity()
        self.append_stats()

    def append_records(self):
        self.mu_history.append(self.opt.mu)
        self.NA_pos_history.append(self.scenario.net_agents.get_positions())
        self.TA_pos_history.append(self.scenario.task_agents.get_positions())
        self.alpha_history.append(self.scenario.alpha)
        grad_mat = self.scenario.get_grad_capacity()
        mu_mat = self.opt.mu.reshape(self.opt.N,self.opt.N,1) + self.opt.mu.reshape(self.opt.N,self.opt.N,1).transpose(1,0,2)
        delta_pos_mat = grad_mat * mu_mat
        delta_pos_arr = -self.alpha * delta_pos_mat.sum(axis=1)
        grad = delta_pos_arr[self.opt.num_task_agents:self.opt.N]
        self.grad_history.append(grad)

    def append_stats(self):
        c1 = np.sum((np.sum(self.opt.r,axis=0)-self.opt.capacity))
        c2 = np.sum(((self.opt.a - (np.sum(self.opt.r,axis=2).T - np.sum(self.opt.r,axis=1).T)))[self.opt.mask_sources==1])
        c3 = np.sum(((self.opt.a - (np.sum(self.opt.r,axis=2).T - np.sum(self.opt.r,axis=1).T)))[self.opt.mask_reles==1])
        self.unf_1.append(c1)
        self.unf_2.append(c2)
        self.unf_3.append(c3)
        self.obj_hist.append(self.opt.evaluate_primal())
        self.lag_hist.append(self.opt.evaluate_lagrangian())
        self.r_change_history.append(np.linalg.norm(self.opt.r))

    def plot_stats(self,axis=None):
      if axis is None:
        fig, ax = plt.subplots(1,3)
        fig.set_size_inches(18,6)
        ax[0].plot(self.unf_1)
        ax[1].plot(self.unf_2)
        ax[2].plot(self.unf_3)
        ax[0].set_title(r'Link Capacity: $\sum_{ij} \quad \sum_k r_{ij}^k - c_{ij}$')
        ax[1].set_title(r'Flow Capacity: $\sum_{ik} \quad a_i^k -  \sum_i r_{ij}^k - \sum_i r_{ji}^k$')
        ax[2].set_title(r'Reles Flow: $\sum_{ik} \quad \sum_i r_{ij}^k - \sum_i r_{ji}^k$')
        ax[0].grid(True)
        ax[1].grid(True)
        ax[2].grid(True)
        plt.show()

        plt.plot(self.r_change_history,label="Change in r")
        plt.show()

        # plt.plot(self.obj_hist,label="Objective")
        # plt.plot(self.lag_hist,label="Lagrangian")
        # plt.legend()
        # plt.grid(True)
        # plt.title("Objective vs Lagrangian")
        # plt.show()
      else:
        axis[0].plot(self.unf_1)
        axis[1].plot(self.unf_2)
        axis[2].plot(self.unf_3)
        axis[0].set_title(r'Link Capacity: $\sum_{ij} \quad \sum_k r_{ij}^k - c_{ij}$')
        axis[1].set_title(r'Flow Capacity: $\sum_{ik} \quad a_i^k -  \sum_i r_{ij}^k - \sum_i r_{ji}^k$')
        axis[2].set_title(r'Reles Flow: $\sum_{ik} \quad \sum_i r_{ij}^k - \sum_i r_{ji}^k$')
        axis[0].grid(True)
        axis[1].grid(True)
        axis[2].grid(True)

    def simulate(self):
      #fig,ax = plt.subplots(1,2)
      #fig.set_size_inches(14,6)
      #xTA_ini=self.scenario.task_agents.get_positions()
      #xNA_ini=self.scenario.net_agents.get_positions()
      #self.scenario.plot_agents(axis=ax[0],title="Intial Agent Positions")
      for i in range(self.nmbr_iterations):
        start_time = time.time()
        if self.decreasing_alpha:
          self.scenario.alpha = self.alpha_0 / (1 + i/self.decreasing_constant)
        self.step()
        if i % 50 ==0 and self.plot_partial:
         self.scenario.plot_agents()
        end_time = time.time()
        self.iter_time.append(end_time-start_time)

      #xTA_fin=self.scenario.task_agents.get_positions()
      xNA_fin=self.scenario.net_agents.get_positions()
      #self.scenario.plot_agents(axis=ax[1],title="Final Agent Positions")
      #plt.savefig("case_1_agents_pos.pdf")
      #plt.plot()
      #self.opt.plot_stats()
      #self.plot_stats()
      #if self.plot_final:
        #self.scenario.plot_agents_w_rates(self.opt.r,self.opt.commodities,save_path="case_1_comms.pdf")
      
      return xNA_fin

    def solve_for_all_times(self,full_solve_steps=5000):
        diff_grad = []
        diff_mu = []
        obj = []
        dual = []
        grads = []
        for j in range(0,len(self.NA_pos_history),10):
          NA_pos = self.NA_pos_history[j]
          TA_pos = self.TA_pos_history[j]
          opt_local = Optimization(TA_pos.shape[0], NA_pos.shape[0],TA_pos.shape[0],self.beta_1,self.beta_2,self.alpha_0,self.l2_penalty)
          scen_local = Scenario(TA_pos.shape[0], NA_pos.shape[0],TA_pos=TA_pos,NA_pos=NA_pos)
          opt_local.capacity = scen_local.get_capacity()
          for i in range(full_solve_steps):
              opt_local.update_a()
              opt_local.update_r()
              opt_local.update_lbda()
              opt_local.update_mu()
          # Calculate Optimal Gradient
          grad_mat = scen_local.get_grad_capacity()
          mu_mat = opt_local.mu.reshape(opt_local.N,opt_local.N,1) + opt_local.mu.reshape(opt_local.N,opt_local.N,1).transpose(1,0,2)
          delta_pos_mat = grad_mat * mu_mat
          delta_pos_arr = -self.alpha_history[j] * delta_pos_mat.sum(axis=1)
          grad_local = delta_pos_arr[opt_local.num_task_agents:opt_local.N]
          grads.append(np.linalg.norm(grad_local))
          diff_grad.append(np.linalg.norm(grad_local-self.grad_history[j]))
          diff_mu.append(np.linalg.norm(opt_local.mu-self.mu_history[j]))
          obj.append(opt_local.evaluate_primal())
          dual.append(opt_local.evaluate_lagrangian())
          self.optimal_mu_history.append(opt_local.mu)
        '''
        # PLOTS
        plt.plot(diff_mu)
        plt.grid(True,alpha=0.3)
        plt.title("Difference between Mus")
        plt.show()

        plt.plot(obj)
        plt.plot(dual)
        plt.grid(True,alpha=0.3)
        plt.title("Primal and Dual values")
        #plt.show()

        plt.plot(grads)
        plt.grid(True,alpha=0.3)
        plt.title("Gradient Norm")
        plt.show()

        plt.plot(diff_grad)
        plt.grid(True,alpha=0.3)
        plt.title("Gradient Difference")
        plt.show()
        '''
        return 