import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

from . import geometry2


def compute_db_index(poses, labels, cluster_centers, w=1, compute_indexes=False):
    number_of_clusters = len(cluster_centers)
    
    indexes_, centroids, cluster_radius = [], [], []
    for cluster_n in range(number_of_clusters):
        centroid = geometry2.SE2Poses(cluster_centers[cluster_n][:2],
                                     geometry2.Rotation2.from_euler('xyz', cluster_centers[cluster_n] * [0, 0, 1]))
        centroids.append(centroid)
        cluster_radius.append(np.sqrt(np.mean([geometry2.metric(centroid, pose, w=w)**2 for pose in poses[cluster_n==labels]])))
        
    if compute_indexes:
        for centroid in centroids:
            indexes_.append(int(np.argmin([geometry2.metric(centroid, pose, w=w) for pose in poses])))
     
    db_matrix = np.zeros([number_of_clusters, number_of_clusters])
    for i in range(number_of_clusters):
        for j in range(i+1, number_of_clusters):
            db_matrix[i, j] = (cluster_radius[i] + cluster_radius[j]) / np.array(geometry2.metric(centroids[j], centroids[i], w=w))
            db_matrix[j, i] = db_matrix[i, j]
    
    return db_matrix.max(0).mean(), indexes_

def compute_db_index_feats(feats, labels, cluster_centers, w=1, compute_indexes=False):
    number_of_clusters = len(cluster_centers)
    
    indexes_, centroids, cluster_radius = [], [], []
    for cluster_n in range(number_of_clusters):
        centroid = cluster_centers[cluster_n]
        
        centroids.append(centroid)
        cluster_radius.append(np.sqrt(np.mean(np.linalg.norm(feats[cluster_n==labels] - centroid, axis=-1)**2)))
        
    if compute_indexes:
        for centroid in centroids:
            indexes_.append(int(np.argmin(np.linalg.norm(feats - centroid, axis=-1))))
     
    db_matrix = np.zeros([number_of_clusters, number_of_clusters])
    for i in range(number_of_clusters):
        for j in range(i+1, number_of_clusters):
            db_matrix[i, j] = (cluster_radius[i] + cluster_radius[j]) / np.linalg.norm(centroids[j] - centroids[i])
            db_matrix[j, i] = db_matrix[i, j]
    
    return db_matrix.max(0).mean(), indexes_



def try_clusters(full_poses_, full_feats_, min_c, max_c, step_c, repetitions, wd=1, title='Unknown dataset', return_indexes=False):
    ks, infs = [], []
    print('Using SE2 and w=1 for combined distance!')
    
    for k in np.arange(min_c, max_c, step_c):
        
        db_i, db_if, ch_i = [], [], []
        for i in range(repetitions):            
            kmeans = KMeans(n_clusters=k, 
                            tol=1e-5, max_iter=10000, algorithm='full', n_init=50).fit(full_poses_.as_numpy())

            db_ind, indexes = compute_db_index(full_poses_, kmeans.labels_, kmeans.cluster_centers_, w=wd, compute_indexes=True)
            feat_centers = full_feats_[indexes]
            db_indf, _ = compute_db_index_feats(full_feats_, kmeans.labels_, feat_centers, w=wd, compute_indexes=return_indexes)

            # Davies-Bouldin Index
            db_i.append(db_ind)
            db_if.append(db_indf)

        print(k, np.mean(db_i), np.mean(db_if))

        ks.append(k)
        infs.append([np.mean(db_i), np.mean(db_if)])

    
    fig = plt.figure(figsize=(18, 6.5))
    fig.suptitle(title, fontsize=50)
    fig.tight_layout(rect=[0.05, 0.05, 0.95, 0.95])
    plt.tick_params(axis='x', labelsize=19)
    plt.tick_params(axis='y', labelsize=19)
    plt.xlabel("Number of clusters", fontsize=28)
    plt.ylabel("DB Index", fontsize=28)
    plt.plot(ks, [x[0] for x in infs], label="poses")
    plt.plot(ks, [x[1] for x in infs], label="feats")
    plt.legend()
    plt.show()