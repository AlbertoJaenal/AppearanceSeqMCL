import time
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats._multivariate import _PSD,_LOG_2PI

from utils import geometry2
from utils import map_abstraction_utils

from . import allom_map_topol
from . import allom_map_metric

################################################################
######### ALLOM Map
################################################################

class ALLOMmap:
    def __init__(self, 
                 map_poses,
                 map_feats,
                 global_pose_means, 
                 global_feat_means, 
                 global_covariances, 
                 num_components=128,
                 verbose=False):
         
        self.verbose = verbose
        
        self._REG = 1e-6
        assert len(map_poses) == len(map_feats)
        
        self.numRegions = len(global_pose_means)
        assert self.numRegions == len(global_feat_means) and self.numRegions == len(global_covariances)
        
        pg = global_pose_means if not global_pose_means._single else [global_pose_means]
        self.set_up(map_poses, map_feats, pg, global_feat_means, global_covariances, num_components)
        
        
    ###############
    # SET TOPOLOGICAL AND METRIC REGIONS
    def set_up(self, poses_map, feats_map, poses_mean_g, feats_mean_g, covs_g, num_dims):
        ######
        # TOPOLOGICAL
        self.topologicalRegions = [allom_map_topol.TopologicalRegion(pose_m, feat_m, cov) for (pose_m, feat_m, cov) in zip(poses_mean_g, feats_mean_g, covs_g)]
        
        ######
        # METRIC
        reg_matrix = np.eye(4) * self._REG * [1, 1, 1, 0]
        
        
        likelihood_sample_per_region = np.array(([map_abstraction_utils.Single_GAM_Pose_Appearance(pose_m, 
                                                                                                   feat_m, 
                                                                                                   cov_m + reg_matrix).likelihood(poses_map, feats_map) \
                                                  for (pose_m, feat_m, cov_m) in zip(poses_mean_g, feats_mean_g, covs_g)]))
        
        # All regions are assumed to have the same prior
        assignation, weights = likelihood_sample_per_region.argmax(0), likelihood_sample_per_region.max(0)
        assert len(np.unique(assignation)) == len(poses_mean_g)
        
        
        self.metricRegions = [allom_map_metric.MetricRegion(poses_map[assignation==i], 
                                                            feats_map[assignation==i], 
                                                            weights[assignation==i],
                                                            num_components=num_dims) \
                              for i in range(len(poses_mean_g))]
        # Proportionality DEBUG
        self.descriptor_dim = np.array([metricReg.d_size for metricReg in self.metricRegions])
        
        # Propagation
        self.inRegionEstim = allom_map_metric.InRegionEstimator(geometry2.combine([metReg.pose_mean_local for metReg in self.metricRegions]), 
                                                                [np.copy(metReg.local_covariance[:3, :3]) + np.eye(3) * self._REG for metReg in self.metricRegions])
                                                                
        # Relocalization
        self.topologicalReloc = allom_map_topol.TopologicalEstimator(poses_mean_g, 
                                                                     [np.copy(cov[:3, :3]) + np.eye(3) * self._REG for cov in covs_g])
        
        if self.verbose: 
            print('Generated %d topological and metric regions' % (len(self.topologicalRegions)))
            print('Local descriptor!\n\tDimensions:', self.descriptor_dim)
    
    ################################## 
    
    ##################################
    # VPR
    def vpr_mahalannobis(self, observation):
        return np.array([topolReg.mahalannobis(observation) for topolReg in self.topologicalRegions])
    
    def vpr_log_likelihood(self, observation):
        return np.array([topolReg.log_likelihood(observation) for topolReg in self.topologicalRegions]) 
    
    
    ##################################
    # INITIALIZATION: Pose regression
    def initial_pose_pdf(self, region, observation):
        return self.metricRegions[region].get_pose_estimation(observation)
    
    
    ##################################
    # DELTA likelihood
    def delta_mahalannobis(self, poses, observation):
        # Get delta given p [FOR ALL REGIONS]
        return (np.array([metricReg.obs_mahalannobis(poses, observation) for metricReg in self.metricRegions])).T
    
    def proj_obs_log_likelihood(self, poses, observation):
        return (np.array([metricReg.obs_log_likelihood(poses, observation) for metricReg in self.metricRegions])).T
    
    def delta_log_likelihood_unique(self, poses, regions, observation):
        a = np.zeros([len(poses)])
        for i in np.unique(regions):
            inds_ = np.argwhere(i==regions).reshape([-1])
            a[inds_] = self.metricRegions[i].obs_log_likelihood(poses[inds_.squeeze()], observation)
        return a.T 
    
    
    ##################################
    # In region for the metric GAMs 
    def pose_mahalannobis_metric(self, poses):
        # Get pose mahalannobis  [FOR ALL REGIONS]
        return self.inRegionEstim.mahalannobis(poses)
        
    def pose_log_likelihood_metric(self, poses):
        # Get pose likelihood  [FOR ALL REGIONS]
        return self.inRegionEstim.log_likelihood(poses)
        
    # In region for the topological GAMs
    def pose_mahalannobis_topol(self, poses):
        # Get pose mahalannobis  [FOR ALL REGIONS]
        return self.topologicalReloc.mahalannobis(poses)
        
    def pose_log_likelihood_topol(self, poses):
        # Get pose likelihood  [FOR ALL REGIONS]
        return self.topologicalReloc.log_likelihood(poses)
    
    
    # Extract the params
    def get_params(self, drop=False, metric=True):
        # Dictionary of parameters
        parameter_dict = {
            'mu_pose': [],
            'mu_feat': [], 
            'cov_matrix': []
        }
        if metric:
            for metricReg in self.metricRegions:
                parameter_dict['mu_pose'].append(metricReg.pose_mean_local)
                parameter_dict['mu_feat'].append(metricReg.proj_feat_mean_local)
                parameter_dict['cov_matrix'].append(metricReg.local_covariance)
        else:
            for topolReg in self.topologicalRegions:
                parameter_dict['mu_pose'].append(topolReg.mean_pose)
                parameter_dict['mu_feat'].append(topolReg.mean_feat)
                parameter_dict['cov_matrix'].append(topolReg.cov_matrix)
        
        parameter_dict['mu_pose'] = geometry2.combine(parameter_dict['mu_pose'])
        parameter_dict['mu_feat'] = np.array(parameter_dict['mu_feat'])
        parameter_dict['cov_matrix'] = np.array(parameter_dict['cov_matrix'])
        if len(self.metricRegions) > 1:
            parameter_dict['numpy_mu_pose'] = np.hstack([parameter_dict['mu_pose'].t(), 
                                                         parameter_dict['mu_pose'].R().as_euler('xyz')[:, -1].reshape([-1, 1])])
        else:
        
            parameter_dict['numpy_mu_pose'] = np.hstack([parameter_dict['mu_pose'].t(), 
                                                         parameter_dict['mu_pose'].R().as_euler('xyz')[-1]])
        parameter_dict['N'] = len(self.metricRegions)
        return parameter_dict
