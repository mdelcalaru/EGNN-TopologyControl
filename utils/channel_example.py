from matplotlib import cm
from channel_model import expModel, PiecewisePathLossModel
import numpy as np
import matplotlib.pyplot as plt

#cm = expModel(alpha=0.7)
#cm_ind=expModel(indicatrix=True)
cm_ind=PiecewisePathLossModel(print_values=False)
def Capacity(x, y, cm):
    rate, var = cm.predict(np.vstack((x, y)))
    return rate[0,1]

def adjacency(x, y, cm):
    return cm.adjacency(np.vstack((x, y)))


x = np.zeros((80,2))
y = np.array([0,0])
x[:,0]=np.linspace(0, 60, 80)
l=[]
l2=[]
l3=[]

for i in range(x.shape[0]):
    l.append(Capacity(x[i,:], y, cm_ind))
    adj=adjacency(x[i,:], y, cm_ind)

    l2.append(adj[0,1])
    adj_ind=adjacency(x[i,:], y, cm_ind)
    l3.append(adj_ind[0,1])

#major_ticks = np.linspace(0, 1, 3, endpoint=True)
# minor_ticks = np.linspace(0, 1, 9, endpoint=True)

# major_ticksx = np.linspace(0, 20, 5, endpoint=True)
# minor_ticksx = np.linspace(0, 20, 9, endpoint=True)

plt.plot(x[:,0], l, x[:,0], l3)
# plt.yticks(minor_ticks, minor=True)
# plt.yticks(major_ticks, minor=False)
# plt.xticks(minor_ticksx, minor=True)
# plt.xticks(major_ticksx, minor=False)
plt.legend(['Capacity', 'Indicatrix'])
#plt.plot(x[:,0], l,x[:,0], l3)
#plt.axvline(x=14.5)
#plt.axhline(y=0.125)
plt.grid(which='minor', alpha=0.2)
plt.grid(which='major', alpha=0.5)
plt.grid('both')
plt.xlabel('Distance')
plt.ylabel('Value')
plt.show()
