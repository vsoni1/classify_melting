import sys
import os
import simulate_crystal_transitions
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
import glob
import simulate_crystal_transitions

def psi6(points):
    """calculates  psi_6 order parameter of point set

    Parameters
    ----------
    points : N x 2 array of coordinates

    Returns
    -------
    psi6s
        N x 1 array of order parameters associated with each particle
    
    """
    def find_neighbors(ix, triang):
        nix_1 = triang.vertex_neighbor_vertices[0][ix]
        nix_2 = triang.vertex_neighbor_vertices[0][ix+1]
        return triang.vertex_neighbor_vertices[1][nix_1:nix_2]
        
    not_nan_indices = np.unique(np.where(
                        np.isfinite(points))[0])
    psi6s = np.empty(len(points),dtype=np.complex)
    psi6s.fill(np.nan)
    pol6 = np.empty([len(points),2])
    pol6.fill(np.nan)
    not_nan_points = points[not_nan_indices]
    tri = Delaunay(not_nan_points)
    for pi,nni in zip(range(len(not_nan_points)),
                      not_nan_indices):
        neighs = find_neighbors(pi,tri)
        vecs = not_nan_points[neighs]-not_nan_points[pi]
        #angs = np.arctan2(vecs[:,1],vecs[:,0])
        angs = np.arctan(vecs[:,1]/vecs[:,0])
        psi6s[nni]  = np.mean(np.exp(6.0j*angs))
    return psi6s  
    
def generate(path):
    """generates labeled data needed for training and saves to disk 
    in path

    Parameters
    ----------
    path : directory where to save

    """
    
    if not os.path.exists(path+'/lj_fluid_transition/*.gsd'):
        os.makedirs(savepath)  
        for temp in np.linspace(2.5,100,40):
            for i in range(1200):
                simulate_crystal_transitions.run_sim(int(i),int(temp))
        
    data = glob.glob(path+'/lj_fluid_transition/*.gsd')

    T = []
    p6 = []
    H = []

    for ind,item in enumerate(data):
        print(ind)
        t = gsd.hoomd.open(name=item, mode='rb')
        points = t[4].particles.position[:,[0,1]]
        H.append(np.histogram2d(points[:,0],points[:,1], 120)[0].ravel())
        p6.append(
            np.mean(
                np.abs(psi6(points))
            )
        )
        T.append(
            int(item.rsplit('kT')[1].rsplit('_')[0])
        )

    inds = np.arange(48000) 
    np.random.shuffle(inds)    
    
    labels = np.array([[int(p<.66),int(p>.66)] for p in p6])
    data = H[inds]
    labels = labels[inds]
    testdata = data[:4000]
    traindata = data[4000:]
    testlabels = labels[:4000]
    trainlabels = labels[4000:]
    np.savez(path+'/traindata',data=traindata,labels=trainlabels)
    np.savez(path+'/testdata',data=testdata,labels=testlabels)        