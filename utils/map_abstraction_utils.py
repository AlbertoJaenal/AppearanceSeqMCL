import numpy as np
from scipy.spatial.transform import Rotation
from scipy.stats._multivariate import _PSD,_LOG_2PI
from sklearn.cluster import KMeans

from . import geometry2
from .DB_index import compute_db_index


##############################################################################################
#########################              INITIALIZATION                #########################
##############################################################################################

def initialize(init_mode, number_of_clusters, poses, feats, repetitions=3):
    if init_mode == 'random':
        # Get the indexes
        indexes = np.sort(np.random.choice(np.arange(len(poses)),  number_of_clusters, replace=False))
        covariances = np.zeros([len(indexes), 4, 4])
        for cluster_n, ind_ in enumerate(indexes):
            covariances[cluster_n, :] = np.eye(4) * [0.5, 0.5, np.pi/18, 0.5]
        
    elif init_mode == 'kmeans':
        eval_dict = []
        # Using cosine and sine to codify the pose for the initialization
        new_poses = np.hstack([poses.as_numpy()[:, :2], 
                               np.cos(poses.as_numpy()[:, -1].reshape([-1, 1])), 
                               np.sin(poses.as_numpy()[:, -1].reshape([-1, 1]))])
        # Select the intialization with lower DB index
        for try_n in range(repetitions):
            kmeans = KMeans(n_clusters=number_of_clusters, tol=1e-5, max_iter=10000, algorithm='full', n_init=50).fit(new_poses)
            centers = np.hstack([kmeans.cluster_centers_[:, 0].reshape([-1, 1]),
                                 kmeans.cluster_centers_[:, 1].reshape([-1, 1]),
                                 np.arctan2(kmeans.cluster_centers_[:, 2], 
                                            kmeans.cluster_centers_[:, 3]).reshape([-1, 1])])
            
            db_index, indexes = compute_db_index(poses, kmeans.labels_, centers, compute_indexes=True)
            eval_dict.append((db_index, indexes, np.copy(kmeans.labels_)))
            
            
        print('DBs:', [x[0] for x in eval_dict])
        indexes = eval_dict[int(np.argmin([x[0] for x in eval_dict]))][1]
        labels = eval_dict[int(np.argmin([x[0] for x in eval_dict]))][2]
        
        # Pre-compute the covariances
        covariances = np.zeros([len(indexes), 4, 4])
        for cluster_n, ind_ in enumerate(indexes):
            covariances[cluster_n, :3, :3] = np.cov(geometry2.logSE2(poses[labels==cluster_n] / poses[ind_]).T)
            covariances[cluster_n,  3,  3] = np.mean(np.linalg.norm(feats[labels==cluster_n] - feats[ind_], axis=-1) ** 2)
        
    else:
        raise ValueError('{} not a known init method'.format(init_mode))
    
    return indexes, covariances, labels
    

####################################################################################################
#########################              2D POSE + APPEARANCE                #########################
####################################################################################################
    
def maximization(data, per_cluster_probs, previous_mean, dev=False):
    pose_data, feat_data = data
    norm_probs = per_cluster_probs.reshape([-1, 1]) / np.sum(per_cluster_probs)
    
    muPos = geometry2.weighted_SE2_mean(pose_data, norm_probs, prev_mean=previous_mean[0])
    muApp = np.average(feat_data, weights=norm_probs.squeeze(), axis=0)
    
    pose_dev = geometry2.logSE2(muPos / pose_data) # From mu to poses
    feat_dev = np.linalg.norm(feat_data - muApp, axis=1, keepdims=True)
    
    cov = np.identity(4)
    cov[:3, :3] = np.dot((norm_probs * pose_dev).T, pose_dev)
    cov[3,   3] = np.dot((norm_probs * feat_dev).T, feat_dev)
    return (muPos, muApp), cov
    
class pose_feat_distribution:
    # Distribution parameters
    def __init__(self, mean, cov):
        self.pose_mean, self.feat_mean = mean
        self.cov = cov

    # PDF evaluation 
    def pdf(self, data):
        pose_data, feat_data = data
        
        # Concatenating
        pose_dev = geometry2.logSE2(self.pose_mean / pose_data) # From mu to poses
        total_dev = np.hstack([pose_dev,
                               np.linalg.norm(feat_data - self.feat_mean, axis=1, keepdims=True)])
        
        psd = _PSD(self.cov, allow_singular=False)
        rankp, prec_Up, log_det_covp = psd.rank, psd.U, psd.log_pdet
        total_maha = np.sum(np.square(np.dot(total_dev, prec_Up)), axis=-1)
        
        return np.exp(-0.5 * (rankp * _LOG_2PI + log_det_covp + total_maha))
        

####################################################################################################
#########################                   EVALUATION                     #########################
####################################################################################################

####################################################################################################
######### Class to simultaneously handle the appearance term of different GAMs (for VPR purposes)
####################################################################################################
class Multi_GAM_Appearance():
    def __init__(self, ref_desc, ref_desc_var):
        self.inv_vars = 1 / (ref_desc_var)
        self.log_k = _LOG_2PI + np.log(ref_desc_var)
        self.ref_desc = ref_desc
        
    def mahalannobis(self, query_desc):
        if len(query_desc.shape) == 1: query_desc = query_desc.reshape([1, -1])
        return np.sqrt(((query_desc[:, None, :] - self.ref_desc)**2).sum(-1) * self.inv_vars)
        
    def log_likelihood(self, query_desc):
        return -0.5 * (self.log_k + self.mahalannobis(query_desc)**2)
    
    def likelihood(self, query_desc):
        return np.exp(self.log_likelihood(query_desc))
####################################################################################################
######### Class to simultaneously handle the pose term of different GAMs (for topological purposes)
####################################################################################################
class Multi_GAM_Pose():
    def __init__(self, 
                 pose_means, 
                 pose_covs):
        # Mean information
        self.pose_means = pose_means
        
        # Covariance information
        self.cov_infos = []
        for ref_cov in pose_covs:
            psd = _PSD(ref_cov, allow_singular=False)
            # rankp, prec_Up, log_det_covp
            self.cov_infos.append((psd.rank, 
                                   psd.U, 
                                   psd.log_pdet))
       
        self.ranks = np.stack([x[0] for x in self.cov_infos])
        self.Us = np.stack([x[1] for x in self.cov_infos])
        self.log_pdets = np.stack([x[2] for x in self.cov_infos])
    
    # Mahalannobis distance between a query (or queries) and the pose distributions of the GAM
    def mahalannobis(self, query_pose):
        if len(self.pose_means) == 1:
            if len(query_pose) == 1:
                probs = np.array([np.sum(np.square(np.dot(geometry2.logSE2(query_pose / self.pose_means), self.Us[0])))])
            else:
                probs = np.zeros([len(self.pose_means), len(query_pose)])
                probs[0, :] = np.square(np.dot(geometry2.logSE2(query_pose / self.pose_means), self.Us[0])).sum(1)
        else:
            if len(query_pose) == 1:
                probs = np.array([np.sum(np.square(np.dot(dev, u))) for dev, u in \
                                  zip(geometry2.logSE2(query_pose / self.pose_means), self.Us)])
            else:
                probs = np.zeros([len(self.pose_means), len(query_pose)])
                for i, (sb, u) in enumerate(zip(self.pose_means, self.Us)):
                    probs[i, :] = np.square(np.dot(geometry2.logSE2(query_pose / sb), u)).sum(1)
        return np.sqrt(probs)
    
    # Log pdf between a query (or queries) and the pose distributions of the GAM
    def log_likelihood(self, query_pose):
        if len(query_pose) == 1:
            return -0.5 * (self.ranks * _LOG_2PI + self.log_pdets + self.mahalannobis(query_pose)**2)
        else:
            return -0.5 * (self.ranks.reshape([-1, 1]) * _LOG_2PI + self.log_pdets.reshape([-1, 1]) + self.mahalannobis(query_pose)**2)


###############################################################################################
######### Class to handle the pose and appearance term of a single GAM
###############################################################################################
class Single_GAM_Pose_Appearance():
    def __init__(self, 
                 pose_mean, 
                 feat_mean, 
                 covariance):
        self.pose_mean = pose_mean
        self.feat_mean = feat_mean
        
        psd = _PSD(covariance, allow_singular=False)
        self.cov_info = (psd.rank, psd.U, psd.log_pdet)
        
    def mahalannobis(self, query_poses, query_descs):
        dev = np.hstack([geometry2.logSE2(self.pose_mean / query_poses),
                         np.linalg.norm(query_descs - self.feat_mean, axis=-1, keepdims=True)])
        return np.sqrt(np.sum(np.square(np.dot(dev, self.cov_info[1])), axis=-1))
    
    def log_likelihood(self, query_poses, query_descs):
        return -0.5 * (self.cov_info[0] * _LOG_2PI + self.cov_info[2] + self.mahalannobis(query_poses, query_descs)**2)
    
    def likelihood(self, query_poses, query_descs):
        return np.exp(self.log_likelihood(query_poses, query_descs))
