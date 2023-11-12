import time
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import KDTree
import sklearn.gaussian_process as GPy

from utils import geometry2

def gp_kernel(poses):
    # Configuration for Gaussian Process
    if len(poses) > 1: r = poses.as_numpy()[:, -1].reshape([-1, 1])
    else: r = poses.as_numpy().squeeze()[-1].reshape([-1, 1])
    return np.hstack([poses.t().reshape([-1, 2]), np.cos(r), np.sin(r)])

class GPPFMap:
    def __init__(self, 
                 map_poses,
                 map_feats,
                 sod_neighbors=10):
        
        # Map info
        self.map_poses = map_poses
        self.gp_map_poses = gp_kernel(self.map_poses)
        self.map_feats = map_feats
        
        # GP
        self.gp_regressor = self.set_up()
        
        # For plot purposes
        self.selectedSamples = []
        
        # Position tree (can be improved with Nigh)
        self.sod_neighbors = sod_neighbors
        self.ref_pose_tree = KDTree(self.gp_map_poses, 
                                    leafsize=self.sod_neighbors)
        
    def set_up(self):
        # Train with some random data from the map
        train_N = min(len(self.map_poses)//100, 10) # Number of training samples
        train_inds = np.random.choice(np.arange(len(self.map_poses)), train_N, replace=False)
        
        # Set the kernel and the GP parameters
        kernel = 1.0 * GPy.kernels.RBF() + 1.0 * GPy.kernels.WhiteKernel()
        gp_regressor = GPy.GaussianProcessRegressor(kernel=kernel, 
                                                    n_restarts_optimizer=10)
        gp_regressor.fit(self.gp_map_poses[train_inds],
                         self.map_feats[train_inds])
        
        # Disable fitting from now
        gp_regressor.optimizer = None
        gp_regressor.n_restarts_optimizer = 0
        
        return gp_regressor
    
    def log_likelihood(self, poses, observation):
        _, gt_ind  = self.ref_pose_tree.query(gp_kernel(poses), 
                                              k=self.sod_neighbors)
        
        log_liks = []
        tims = []
        for i, inds_ in enumerate(gt_ind):
            self.gp_regressor.fit(self.gp_map_poses[inds_], 
                                  self.map_feats[inds_])
            mean, std = self.gp_regressor.predict(gp_kernel(poses[i]), 
                                                  return_std=True)
            log_liks.append(geometry2.log_univariate_pdf(observation, mean, std**2))
        
        # For plot purposes
        self.selectedSamples = self.map_poses[np.unique(gt_ind)]
        
        return np.array(log_liks)

