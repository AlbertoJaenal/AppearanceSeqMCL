import numpy as np
from scipy.stats import chi2

from utils import geometry2
from . import particleFilter

class ALLOMParticleFilter(particleFilter.ParticleFilter):
    def __init__(self, 
                 particle_number,
                 allom_map,
                 reinit_window=-1,
                 odom_sigma=[0.025, 0.01, 0.015]):
        super().__init__(particle_number, odom_sigma)
        
        self.allom_map = allom_map
        
        self.topological_lik_hist = []
        self.topological_lik_hist2 = []
        
        self.last_reinit = -1
        self.reinit_thr = np.log(0.9995)
        self.reinit_window = reinit_window
        self.reinit_hist = []
        
        
        self.init_modes = ["perfect", "vpr"]
        self.initial_vpr_thr = 0.8
        
    #######################################
    # Initialization by VPR
    def initialization(self, init_mode):
        assert init_mode in self.init_modes
        # Topological likelihood 
        self.topological_likelihoods = self.topological_likelihoods * 0
        
        if init_mode == "perfect":
            super().initialization(init_mode)
            self.region_of_particles = int(self.allom_map.topologicalPropag.mahalannobis(self.gt).argmin()) *\
                               np.ones([self.N], dtype=np.int)
            
            self.topological_likelihoods[np.arange(self.N), self.region_of_particles] = 1
            return
        else:
            ##############
            # Get VPR
            vpr = np.exp(self.allom_map.vpr_log_likelihood(self.current_observation))

            # Most probable regions and likelihoods (over threshold)
            feasible_regions = np.argwhere(vpr >= vpr.max() * self.initial_vpr_thr).reshape([-1])
            if self.verbose: print('\tREGRESSED INIT! %d REGIONS: ' % len(feasible_regions), feasible_regions)
            feasible_region_likelihoods = np.array(vpr)[feasible_regions]
            feasible_region_likelihoods = feasible_region_likelihoods / feasible_region_likelihoods.sum()

            # Regression
            pose_distributions = [self.allom_map.initial_pose_pdf(feas_reg, self.current_observation) for feas_reg in feasible_regions]

            # Assignation (proportional to the cluster probability)
            particles_per_feasible_region = np.hstack((0, np.cumsum(feasible_region_likelihoods) * self.N)).astype(np.int)
            particles_per_feasible_region = np.diff(particles_per_feasible_region)

            # Initialization
            poses, regions, filled = [], [], 0
            for feas_reg_idx in range(len(feasible_regions)):
                # Initialize the pose distribution for the particles in the region
                cov = (pose_distributions[feas_reg_idx][1] +\
                       pose_distributions[feas_reg_idx][1].T) / 2 + 10e-6 * np.eye(3)
                regressed_poses = pose_distributions[feas_reg_idx][0] *\
                                    geometry2.expSE2(np.random.multivariate_normal(np.zeros([3]), 
                                                                                   cov, 
                                                                                   size=particles_per_feasible_region[feas_reg_idx]).squeeze())
                # Assign the particles to the region
                regions += [feasible_regions[feas_reg_idx]] * particles_per_feasible_region[feas_reg_idx]
                self.topological_likelihoods[filled:filled + particles_per_feasible_region[feas_reg_idx], 
                                             feasible_regions[feas_reg_idx]] = 1
                # Deal with shape 1
                if particles_per_feasible_region[feas_reg_idx] > 1:  poses += regressed_poses
                else:  poses.append(regressed_poses)
                filled += particles_per_feasible_region[feas_reg_idx]

        self.poses = geometry2.combine(poses)
        self.region_of_particles = np.array(regions)
        self.likelihoods = np.ones([len(self.poses)]) / len(self.poses)
    
    #######################################
    # Weight the particles
    # P(p_t | u_t, d_t, p_{t-1}) = Σ_r p(r_t | p_{t-1}, u_t) p(delta_t | p_t, r_t) p(p_t | u_t, p_{t-1})
    def weighting(self, min_weight=0.8):
        # Reinit topological likelihoods
        self.topological_likelihoods = np.zeros([len(self.poses), self.allom_map.numRegions], dtype=np.int)
        
        # Get topological weight: p(r_t | p_{t-1}, u_t)
        # Note that, the particle is ASSIGNED to a region, so its probability of falling there is 1
        region_belief = np.exp(self.allom_map.pose_log_likelihood_metric(self.poses))
        self.region_of_particles = region_belief.argmax(1).reshape([-1])
        self.topological_likelihoods[np.arange(len(self.poses)), self.region_of_particles] = 1
            
        # Get local observation model p(delta_t | p_t, r_t) 
        # Using log_likelihood for stability
        delta_log_likds = self.allom_map.delta_log_likelihood_unique(self.poses, 
                                                                     self.region_of_particles, 
                                                                     self.current_observation)
        # Important for weight compute from log_likelihoods
        if np.any(delta_log_likds < 0):
            delta_log_likds = delta_log_likds - delta_log_likds.min() + abs(delta_log_likds.min())/100
        # Compute Σ_r p(p_t | r_t, u_t, d_t) p(r_t | p_t-1, u_t)
        particle_weights = delta_log_likds * self.topological_likelihoods[np.arange(len(self.poses)), 
                                                                          self.region_of_particles]
        
        # Update w^k_t
        self.update_weight(particle_weights.squeeze())
        
    # Update the particle weight
    def update_weight(self, update):
        super().update_weight(update)
    
        if np.any(np.isnan(self.likelihoods)):
            raise ValueError
            
    #######################################
    # Resample the particles
    def resampling(self):
        # RESAMPLE, Compute reinitialization
        c = np.hstack((0, np.cumsum(self.likelihoods)))
        new_poses, new_regions, topol_liks = [], [], []
        
        for ix in range(len(self.poses)):
            r = np.random.uniform()
            rand_ix = np.searchsorted(c, r) - 1
            new_poses.append(self.poses[rand_ix])
            new_regions.append(self.region_of_particles[rand_ix])
            topol_liks.append(self.topological_likelihoods[rand_ix])
        
        # Poses, regions, and topol_liks
        self.poses = geometry2.combine(new_poses)
        self.region_of_particles = np.array(new_regions)
        self.topological_likelihoods = np.array(topol_liks)
        
        # Set the likelihoods to the same
        self.likelihoods = np.ones([len(self.poses)]) / len(self.poses)

    #######################################
    # Get the mean result
    def get_result(self, near_samples=2):
        # Get cluster probabilities
        region_weight = np.zeros([self.allom_map.numRegions])
        for i in np.unique(self.region_of_particles):
            region_weight[i] += self.likelihoods[np.argwhere(self.region_of_particles==i)].sum()
        region_weight = region_weight / self.likelihoods.sum()
        
        # Topological
        top_region_weight = self.topological_likelihoods.sum(0) / self.topological_likelihoods.sum()
        
        super().get_result(near_samples=near_samples)
        
        # Append region information
        if True: self.result[-1].append(region_weight) # Probabilistic
        else: self.result[-1].append(top_region_weight) # 1/0
        self.historial[-1] = (np.hstack([self.historial[-1],
                                         self.region_of_particles[:, None]]))
        
        if self.reinit_window > 0 and (self.step - self.last_reinit) >= self.reinit_window:
            # Check if do reinit
            self.compute_reloc_likelihood(self.result[-1][0])

            #print(self.step, np.mean(self.topological_lik_hist[-self.reinit_window:]), self.reinit_thr)
            if np.sum(self.topological_lik_hist[-self.reinit_window:]) > self.reinit_thr:
                if self.verbose: print('\tReinitialization step %d' % self.step)
                self.initialization('vpr')
                self.reinit_hist.append(self.step)
                self.last_reinit = self.step
        
    def compute_reloc_likelihood(self, pose_avg):
        # Mahalannobis between estimate and regions
        region_mds = np.square(self.allom_map.pose_mahalannobis_topol(pose_avg))
        # Minimum chi^2
        if False:
            self.topological_lik_hist.append(np.log(np.clip(chi2.cdf(np.clip(region_mds, 0, 50), 3), 0.02, 1.).min()))
        else:
            self.topological_lik_hist2.append(region_mds[self.result[-1][-1].argmax()])
            self.topological_lik_hist.append(np.log(np.clip(chi2.cdf(np.clip(self.topological_lik_hist2[-self.reinit_window:], 0, 50), 3), 0.02, 1.).min()))
        
    def show_state(self, current_gt):
        dev = self.result[-1][0] / current_gt
        self.ate += np.linalg.norm(dev.t())**2
        selected_region = self.allom_map.topologicalRegions[self.result[-1][-1].argmax()]
        mah_dist = geometry2.multivariate_mahalannobis(current_gt, 
                                                       selected_region.mean_pose, 
                                                       selected_region.cov_matrix_pose)
        print("Step %d" % self.step,
              "Pose error  %.2f m, %.2f º" % (np.linalg.norm(dev.t()),
                                                180/np.pi*np.abs(dev.R().as_euler('xyz')[-1])),
              "ATE %.2f" % np.sqrt(self.ate / self.step),
              "|| Mah. dist. %.2f (region %d)" % (mah_dist, self.result[-1][-1].argmax()))
        
    def run(self, odometry, observations, 
            gt_poses=None, init="perfect", 
            step_incr=1, step_plot=1):
        # Initialize the topological likelihood of each particle
        self.topological_likelihoods = np.zeros([self.N, self.allom_map.numRegions], dtype=np.int)
        
        super().run(odometry, observations, 
                    gt_poses=gt_poses, init=init, 
                    step_incr=step_incr, step_plot=step_plot)
            