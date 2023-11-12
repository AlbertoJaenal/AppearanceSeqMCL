import numpy as np
from scipy.stats._multivariate import _PSD,_LOG_2PI
from utils import geometry2

from utils import map_abstraction_utils

################################################################
######### Topological Regions
################################################################
class TopologicalRegion:
    # For the topological OBSERVATION model
    def __init__(self, 
                 pose_mean, 
                 feat_mean, 
                 covariance,
                 verbose=False):
        
        self.verbose = verbose
        
        # Local data
        #self.set_up unnecessary
        self.mean_pose = pose_mean
        self.mean_feat = feat_mean.astype(np.float32)
        
        self.cov_matrix_pose = np.copy(covariance[:3, :3]).astype(np.float32)
        self.cov_matrix_feat = np.copy(covariance[-1, -1]).astype(np.float32)
        
        self.d_size = 1
    
    def mahalannobis(self, observation):    
        return np.sqrt(np.linalg.norm(observation - self.mean_feat)**2 / self.cov_matrix_feat)
    
    def log_likelihood(self, observation):    
        return geometry2.log_univariate_pdf(observation, self.mean_feat, self.cov_matrix_feat)
        
class TopologicalEstimator:
    # Estimator based on particle pose
    def __init__(self, 
                 means, 
                 covariances):
        #self.set_up unnecessary
        self.pose_means = means[0] if len(means) == 1 else means
        self.base = map_abstraction_utils.Multi_GAM_Pose(self.pose_means, covariances)
        
    # Unused odometry and regions.
    def mahalannobis(self, poses):
        return self.base.mahalannobis(poses).T
        
    def log_likelihood(self, poses):
        return self.base.log_likelihood(poses).T