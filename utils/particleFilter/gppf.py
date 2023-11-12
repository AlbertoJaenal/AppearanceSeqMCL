import numpy as np
import time, sys

from utils import geometry2
from .particleFilter import ParticleFilter

class GaussianProcessParticleFilter(ParticleFilter):
    def __init__(self, 
                 particle_number,
                 dense_map,
                 odom_sigma=[0.025, 0.01, 0.015]):
        super().__init__(particle_number, odom_sigma)
        
        self.dense_map = dense_map
        self.random_min = dense_map.map_poses.as_numpy().min(0)
        self.random_max = dense_map.map_poses.as_numpy().max(0)
    
    #######################################
    # Weight the particles
    def weighting(self):
        particle_weights = np.exp(self.dense_map.log_likelihood(self.poses, 
                                                                self.current_observation))
            
        # Update w^k_t
        self.update_weight(particle_weights.squeeze())
        
    # Update the particle weight
    def update_weight(self, update):
        super().update_weight(update)
    
        if np.any(np.isnan(self.likelihoods)):
            raise ValueError