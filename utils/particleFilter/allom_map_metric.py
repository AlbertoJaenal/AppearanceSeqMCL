import numpy as np
from scipy.stats._multivariate import _PSD,_LOG_2PI
from sklearn.decomposition import PCA

from utils import geometry2
from utils import map_abstraction_utils

_COV_REG = 1e-6

################################################################
######### Metric Regions
################################################################
class MetricRegion:
    # For the metric observation model
    def __init__(self, 
                 map_poses,
                 map_feats,
                 weights,
                 num_components=128,
                 verbose=False):
         
        self.verbose = verbose
        self.num_components = num_components
        
        # Set up the data!
        self.set_up(map_poses, map_feats, weights)
        
        # The estimation uncertainty is fixed!! So we can pre-do lot of work!
        # Noise matrix for multi appearance
        noise_matrix = np.eye(len(self.proj_feat_mean_local), dtype=np.float32) + 0.5

        # Sigma_delta_est = sigma_dd - sigma_dp*sigma_pp^{-1}*sigma_pd
        self.feat_estimation_unc = (self.local_covariance[3:, 3:] -\
                                    np.matmul(self.feat_prev_mult, 
                                              self.local_covariance[:3, 3:])).astype(np.float32)
        # Sigma_pose_est = sigma_pp - sigma_pd*sigma_dd^{-1}*sigma_dp
        self.pose_estimation_unc = (self.local_covariance[:3, :3] -\
                                    np.matmul(self.pose_prev_mult, 
                                              self.local_covariance[3:, :3])).astype(np.float32)
        
        # PSD and size for multivariate estimation
        psd = _PSD(self.feat_estimation_unc*noise_matrix + 1e-10 * np.eye(self.num_components), allow_singular=False)
        self.cov_info = (psd.rank, psd.U.astype(np.float32), psd.log_pdet.astype(np.float32))
        del psd
        
        self.d_size = self.local_covariance.shape[0]

    ###############
    # GET ESSENTIAL COMPONENTS: MEAN AND COVARIANCE
    def set_up(self, poses_cluster, feats_cluster, weights):
        # Dilucidate if possible to reduce to the given dimension number. If not, the higher 2-power
        self.num_components = int(2**min(np.log2(self.num_components), 
                                         np.floor(np.log2(len(poses_cluster)-1))))
        
        # Transformation (PCA)
        self.project_method = PCA(n_components=self.num_components, svd_solver='full')
        feats_cluster_proj = self.project_method.fit_transform(feats_cluster).astype(np.float32)
        
        # Obtain the weights
        norm_probs = weights.reshape([-1, 1]) / weights.sum()
        
        # Obtain the local parameters
        self.pose_mean_local = geometry2.weighted_SE2_mean(poses_cluster, 
                                                           norm_probs.squeeze(),
                                                           prev_mean=poses_cluster[weights.argmax()])
        self.proj_feat_mean_local = np.average(feats_cluster_proj, 
                                               weights=norm_probs.squeeze(), 
                                               axis=0).astype(np.float32)
        devs = np.hstack([geometry2.logSE2(self.pose_mean_local / poses_cluster), 
                          feats_cluster_proj - self.proj_feat_mean_local])
        
        # Local data
        self.local_covariance = np.dot((norm_probs * devs).T, devs).astype(np.float32)
        local_feat_covariance_inv = np.linalg.inv(self.local_covariance[3:, 3:]).astype(np.float32)
        local_pose_covariance_inv = np.linalg.inv(self.local_covariance[:3, :3] + _COV_REG * np.eye(3)).astype(np.float32)
        
        # Precompute E for pose
        self.pose_prev_mult = np.matmul(self.local_covariance[:3, 3:], 
                                        local_feat_covariance_inv).astype(np.float32)
        # Precomputed E
        self.feat_prev_mult = np.matmul(self.local_covariance[3:, :3], 
                                        local_pose_covariance_inv).astype(np.float32) 



    ###############
    # REGION DIMENSIONALITY REDUCTION
    def project_query_descriptor(self, descriptor):
        # Transform descriptor
        des = descriptor.reshape([1, -1]) if len(descriptor.shape) == 1 else descriptor
        return self.project_method.transform(des)

    ###############
    # POSE ESTIMATION
    def get_pose_estimation(self, observation):
        # mu_pose_est = mu_p + sigma_pd*sigma_dd^{-1}*(delta_proj_obs)
        poseEstimate = self.pose_mean_local *\
                       geometry2.expSE2(np.matmul(self.pose_prev_mult, 
                                                  self.project_query_descriptor(observation).squeeze() - self.proj_feat_mean_local))
        
        return poseEstimate, self.pose_estimation_unc
        
        
    ###############
    # DELTA MEAN ESTIMATION
    def get_delta_estimation_mean(self, pose):
        # mu_delta_est = mu_d + sigma_dp*sigma_pp^{-1}*(Delta_pose)
        if len(pose) == 1:
            estimation_mean = self.proj_feat_mean_local +\
                              np.matmul(self.feat_prev_mult, 
                                        geometry2.logSE2(self.pose_mean_local / pose).T).astype(np.float32)
        else:
            estimation_mean = self.proj_feat_mean_local.reshape([-1, 1]) +\
                              np.matmul(self.feat_prev_mult,
                                        geometry2.logSE2(self.pose_mean_local / pose).T).astype(np.float32)
        return estimation_mean
    
    ###############
    # MAHALANNOBIS DISTANCE
    # (Private) W
    def _obs_mahalannobis_given_mean(self, observation, means):
        if len(means.shape) == 1:
            return np.sqrt(np.sum(np.square(np.dot(observation - means, self.cov_info[1])), axis=-1))
        else:
            return np.sqrt(np.sum(np.square(np.dot((observation.reshape([-1, 1]) - means).T, self.cov_info[1])), axis=-1))
            
    # (Public) 
    def obs_mahalannobis(self, poses, observation):
        return self._obs_mahalannobis_given_mean(self.project_query_descriptor(observation).reshape([-1]), 
                                                 self.get_delta_estimation_mean(poses))
    ###############
    # LIKELIHOOD ESTIMATION
    # (Private)
    def _log_likelihood_given_mean(self, observation, means):
        return -0.5 * (self.cov_info[0] * _LOG_2PI + self.cov_info[2] + self._obs_mahalannobis_given_mean(observation, means)**2)
    
    # (Public) 
    def obs_log_likelihood(self, poses, observation):
        return self._log_likelihood_given_mean(self.project_query_descriptor(observation).reshape([-1]), 
                                               self.get_delta_estimation_mean(poses))

        
class InRegionEstimator:
    # Estimator based on particle pose
    def __init__(self, 
                 means, 
                 covariances):
        #self.set_up unnecessary
        self.pose_means = means
        self.base = map_abstraction_utils.Multi_GAM_Pose(means, covariances)
        
    # Unused odometry and regions.
    def mahalannobis(self, poses):
        return self.base.mahalannobis(poses).T
        
    def log_likelihood(self, poses):
        return self.base.log_likelihood(poses).T
