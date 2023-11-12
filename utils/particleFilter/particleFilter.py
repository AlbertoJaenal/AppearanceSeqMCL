import numpy as np
import time, sys

from utils import geometry2

class ParticleFilter:
    def __init__(self, 
                 particle_number,
                 odom_sigma=[0.025, 0.01, 0.015]):
        # DATA
        self.N = particle_number
        
        # NOISE
        self.odom_sigma = odom_sigma
        
        # OUTPUT
        self.result = []
        self.historial = []
        
        # INITTIALIZATION
        self.init_modes = ["perfect", "random"]
        self.random_min, self.random_max = np.array([-1, -1, -np.pi]), np.array([1, 1, np.pi])
    
        # DEBUG, verbose
        self.ate = 0
        self.verbose = False
    
   #######################################
    # Initialization of the particle filter
    def initialization(self, init_mode):
        assert init_mode in self.init_modes
        if init_mode == "perfect":
            ##############
            # Initialize around the current gt
            print('PERFECT INITIALIZATION!')
            assert self.current_gt_pose is not None, ""
            poses = self.current_gt_pose *\
                    geometry2.expSE2(np.random.multivariate_normal(np.zeros([3]),  
                                                                   np.eye(3)*1e-3, 
                                                                   size=self.N).squeeze())
            
        elif init_mode == "random":
            print('RANDOM INITIALIZATION!')
            poses = np.random.random([self.N, 3])
            poses = poses * (self.random_max - self.random_min) + self.random_min
            poses = geometry2.SE2Poses(poses[:, :2], 
                                       geometry2.Rotation2.from_euler('z', poses[:, -1]))
        # Poses
        self.poses = geometry2.combine(poses)
        # Set the likelihoods to the same
        self.likelihoods = np.ones([self.N]) / self.N
        
    #######################################
    # Propagate the particles
    def propagate(self):
        # Propagating poses with noisy odometry
        increments = self.current_odometry * geometry2.expSE2(np.random.normal(scale=self.odom_sigma, 
                                                                               size=(len(self.poses), 3)))
        self.poses = self.poses * increments
    
    #######################################
    # Weight the particles
    def weighting(self):
        weight = np.ones([self.N])
        self.update_weight(weight)
        
    # Update the particle weight
    def update_weight(self, update):
        self.likelihoods *= update
        # Normalize \sum_k w^k_t = 1
        self.likelihoods = self.likelihoods / self.likelihoods.sum()
    
    #######################################
    # Resample the particles
    def resampling(self):
        print('\tResampling step %d' % self.step)
        # RESAMPLE, Compute reinitialization
        c = np.hstack((0, np.cumsum(self.likelihoods)))
        new_poses = []
        
        for ix in range(self.N):
            r = np.random.uniform()
            rand_ix = np.searchsorted(c, r) - 1
            new_poses.append(self.poses[rand_ix])
        
        # Poses
        self.poses = geometry2.combine(new_poses)
        # Set the likelihoods to the same
        self.likelihoods = np.ones([self.N]) / self.N
        
    #######################################
    # Get the mean result
    def get_result(self, near_samples=2):
        # Select some temporary mean
        if len(self.result) == 0: 
            pose_avg = self.poses[self.likelihoods.argmax().squeeze()]
        else:
            pose_avg = self.result[-1][0]
        
        # Get distance between the tempotat
        near_samples_idx = geometry2.metric(pose_avg, self.poses, 1) < near_samples
        prob = self.likelihoods[near_samples_idx].sum()
            
        # Reselect the temporary mean if not enough probability
        if prob < 0.5:
            for i in np.argsort(self.likelihoods)[::-1].reshape([-1]):
                near_samples_idx_ = geometry2.metric(self.poses[i], self.poses, 1) < near_samples
                w_ = self.likelihoods[near_samples_idx_].sum()
                
                if w_ > prob:
                    near_samples_idx = near_samples_idx_
                    prob = w_
                    if prob >= 0.5: break
        
        # Compute weighted mean
        if near_samples_idx.sum() == 1:
            pose_avg = self.poses[np.argwhere(near_samples_idx).squeeze()]
        else:
            pose_avg = geometry2.weighted_SE2_mean(self.poses[near_samples_idx], 
                                                   self.likelihoods[near_samples_idx],
                                                   prev_mean=pose_avg)
        # Get the confidence
        confidence = self.likelihoods[near_samples_idx].sum() / self.likelihoods.sum()

        self.result.append([pose_avg, confidence])
        self.historial.append(np.hstack([self.poses.t(), 
                                         self.likelihoods.reshape([-1, 1])])) 
    
    def show_state(self, current_gt):
        dev = self.result[-1][0] / current_gt
        self.ate += np.linalg.norm(dev.t())**2
        print("Step %d" % self.step,
              "Pose error  %.2f %.2f (m, º)" % (np.linalg.norm(dev.t()),
                                                180/np.pi*np.abs(dev.R().as_euler('xyz')[-1])),
              "ATE %.2f" % np.sqrt(self.ate / self.step))
    
    #######################################
    # Run
    def run(self, 
            odometry, 
            observations, 
            gt_poses=None, 
            init="perfect", 
            step_incr=1,
            step_plot=1):
        # Set data
        self.ate = 0
        self.odometries = odometry
        self.observations = observations
        self.gt_poses = gt_poses
        self.step_incr = step_incr
        assert len(self.odometries) == len(self.observations)
        if self.gt_poses is not None: assert len(self.odometries) == len(self.gt_poses)
        
        # Get data
        self.step = 0
        self.current_observation = self.observations[self.step]
        self.current_gt_pose = self.gt_poses[self.step] if self.gt_poses is not None else None
        # Initialize
        self.initialization(init)
        
        self.step += self.step_incr
        t_init = time.time()
        while self.step < len(self.odometries):
            self.current_observation = self.observations[self.step]
            self.current_odometry = self.odometries[self.step - self.step_incr] / self.odometries[self.step]
            self.current_gt_pose = self.gt_poses[self.step] if self.gt_poses is not None else None
            
            self.propagate()            
            self.weighting()
            self.get_result()
            
            # Resampling
            eff_w = 1 / np.sum(self.likelihoods * self.likelihoods)
            if eff_w < (self.N / 2):
                self.resampling()
            
            if self.verbose and self.step % step_plot == 0 and self.gt_poses is not None: self.show_state(self.current_gt_pose)
                
            self.step += self.step_incr
        t_end = time.time()
        
        print('Done! Time per step: %.3f' % ((t_end - t_init) / (len(self.odometries) / self.step_incr)))