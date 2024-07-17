import numpy as np
import cvxpy as cp

class Edge(object):
    """ An undirected, capacity limited edge. """
    def __init__(self, capacity,flows):
        self.capacity = capacity
        self.rates = cp.Variable(flows,nonneg=True)
        self.tau   = cp.Variable(nonneg=True)
        
    # Connects two nodes via the edge.
    def connect(self, out_node, in_node):
        in_node.edge_rates.append(-self.rates)
        out_node.edge_rates.append(self.rates)
        out_node.taus.append(self.tau)
        out_node.neigh_nodes.append(in_node.index)
        #out_node.neigh_taus.append(cp.sum(in_node.taus))
        #in_node.neigh_taus.append(cp.sum(out_node.taus))

    # Returns the edge's internal constraints.
    def constraints(self):
        cst= [0 <= self.capacity*self.tau-cp.sum(self.rates)]
        #cst+=[self.tau<=1]
        return cst
    
class Node(object):
    """ A node with accumulation. """
    def __init__(self, indx,accumulation=[]):
        self.accumulation = accumulation
        self.index=indx
        self.edge_rates = []
        self.taus=[]
        self.neigh_nodes=[]
    # Returns the node's internal constraints.
    def constraints(self):
        #total_rate=0
        #for ra_ed in self.edge_rates:
        #    total_rate+=ra_ed

        cst=[self.accumulation-sum(f for f in self.edge_rates) == 0]
        #cst=[total_rate == self.accumulation]
        #cst+=[1-sum(t for t in self.taus)-sum(nt for nt in self.neigh_taus) >=0]
        
        return cst
    
def MNF_graph_solver(num_task_config, num_comm_config, adj,rate):
    T=num_task_config
    N=num_comm_config+T
    K=T*(T-1)
    Kopts=np.arange(K)

    ak=cp.Variable(K,nonneg=True)
    mapK=[]
    for i in range(T):
        for j in range(i+1,T):
            mapK.append([i,j])

    for i in range(1, T):
        for j in range(i):
            if i!=j:
                mapK.append([i,j])

        
    mapK=np.array(mapK)         

    indx_acu=np.zeros((N,K),dtype=int)
    indx_acu[mapK[:,0],Kopts]=1
    indx_acu[mapK[:,1],Kopts]=-1
    nodes=[]
    for node in range(N):
        nodes.append(Node(node,accumulation=cp.multiply(ak,indx_acu[node,:])))

    edges=[]
    for i in range(N):
        for j in range(N):
            if adj[i,j]:
                edges.append(Edge(rate[i,j], K))
                edges[-1].connect(nodes[i], nodes[j])         

    constraints = []
    for o in nodes + edges:
        constraints += o.constraints()

    for node in nodes:
        neighs_taus=[]
        for nt in node.neigh_nodes:
            neighs_taus.append(cp.sum(nodes[nt].taus))
        
        constraints+=[1-sum(t for t in node.taus)-sum(nt for nt in neighs_taus) >=0]


    logsum=cp.sum(cp.log(ak+1e-6))#/K
    prob = cp.Problem(cp.Maximize(logsum), constraints)
    prob.solve(solver=cp.MOSEK,verbose=False, mosek_params={'MSK_IPAR_INTPNT_SOLVE_FORM':   'MSK_SOLVE_DUAL'} )
    #prob.solve(solver=cp.SCS)#,feastol=1e-24)
    '''edges_tau_value=[]
    edges_rates=[]
    for edge in edges:
        edges_tau_value.append(edge.tau.value)
        edges_rates.append(edge.rates.value)

    nodes_edges_rates=[]
    nodes_taus=[]
    nodes_accumulation=[]
    nodes_neigh_taus=[]

    for node in nodes:   
        nodes_edges_rates.append(node.edge_rates)
        nodes_taus.append(node.taus)
        nodes_accumulation.append(node.accumulation)
        #nodes_neigh_taus.append(node.neigh_taus)
    '''
    return logsum.value, prob.status,# ak.value, edges_tau_value, edges_rates, nodes_edges_rates, nodes_taus, nodes_accumulation, nodes_neigh_taus