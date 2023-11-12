import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.transforms as transforms

from . import geometry2

def plot_ellipse(mean_p, cov_mat, color='b', alpha=0.3):
    cov00, cov11 = cov_mat[0, 0], cov_mat[1, 1]
    pearson = np.clip(cov_mat[0, 1] / np.sqrt(cov00 * cov11),
                      -1, 1)

    ell_radius_x, ell_radius_y = np.sqrt(1 + pearson), np.sqrt(1 - pearson)

    ell_path = patches.Ellipse((0, 0), 
                               width=ell_radius_x * 2, 
                               height=ell_radius_y * 2,
                               alpha=alpha, 
                               linewidth=2, 
                               fill=True, 
                               color=color)
    ell_tr = transforms.Affine2D() \
                       .rotate_deg(45) \
                       .scale(np.sqrt(cov00) * 2, np.sqrt(cov11) * 2) \
                       .translate(mean_p[0], mean_p[1])
    return ell_path, ell_tr



def generate_cov(mean, cov, N=1000):
    n = np.random.multivariate_normal([0, 0, 0], cov, N)
    poses = mean * geometry2.expSE2(n)
    d = poses.as_numpy() - mean.as_numpy()
    return np.dot(d.T, d) / N



def plot_map(model_params, 
             pose_data=None, 
             figure_handle=None,
             title=None, 
             plot_text=False,
             weights=None):
    colors = plt.cm.rainbow(np.linspace(0, 1, model_params['N'] // 2 + 1))
    colors = np.concatenate([colors, colors, colors])

    if figure_handle is not None:
        fig, ax = figure_handle
    else:
        fig, ax = plt.subplots(figsize=(14, 8))
        
    if title is not None:
        plt.title(title, fontsize=40)

    if pose_data is not None:
        plt.scatter(pose_data.t()[:, 0], pose_data.t()[:, 1], c='k', s=2, alpha=0.3)
    
    nn = np.copy(colors)[:model_params['N']]
    
    if weights is not None:
        assert len(weights) == model_params['N']
    
    w_seg = 0.3
    for iseg in range(model_params['N']):
        if weights is not None:
            if weights[iseg] == 0: continue
            w_seg = 0.05 + 0.35 * weights[iseg]
        
        if not model_params['mu_pose']._single:
            mean, color = model_params['mu_pose'][iseg], colors[iseg]
        else:
            mean, color = model_params['mu_pose'], colors[iseg]
        
        # Re-calculate the matrix for positions
        aux_cov = generate_cov(mean, model_params['cov_matrix'][iseg][:3, :3], N=10000)[:2, :2]
        arrow_size = np.sqrt(aux_cov[0, 0] + aux_cov[1, 1])
        
        ell_path, ell_tr = plot_ellipse(mean.t(), aux_cov, color=color, alpha=w_seg)
                 
        ell_path.set_transform(ell_tr + ax.transData)
        ax.add_patch(ell_path)
        if plot_text: 
            plt.text(mean.t()[0] + arrow_size/10 * np.sin(mean.R().as_rotvec().squeeze()), 
                     mean.t()[1] + arrow_size/10 * np.cos(mean.R().as_rotvec().squeeze()), 
                     'Cl. %d' % iseg,
                     color=color * [0.5, 0.5, 0.5, 1],
                     fontsize=15)
        
        ax.arrow(mean.t()[0], 
                 mean.t()[1], 
                 arrow_size * np.cos(mean.R().as_rotvec().squeeze()), 
                 arrow_size * np.sin(mean.R().as_rotvec().squeeze()), 
                 color=color, width=arrow_size / 20, alpha=w_seg)
    if figure_handle is None:
        plt.axis('equal')
        plt.legend(fontsize=20)
        plt.xlabel("x (m)", fontsize=30)
        plt.ylabel("y (m)", fontsize=30)

