import numpy as np
import json, h5py
from . import transformations as tr
from . import geometry2

def interpret_pose(line, do_3d):
    gg = np.zeros([6]) if do_3d else np.zeros([3])
    if len(line) == 12:
        # Three lines matrix format
        pos = np.array(line).reshape([3,4])
        if do_3d:
            gg[:3] = np.copy(pos[:, 3])
            gg[3:] = list(tr.euler_from_matrix(pos[:3, :3]))
        else:
            gg[:2]= np.copy(pos[:, 3])[:2]
            gg[-1] = tr.euler_from_matrix(pos[:3, :3])[-1]
    elif len(line) == 6:
        # xyz-rpy format
        if do_3d:
            gg[:] = line[:]
        else:
            gg[:2] = line[:2]
            gg[-1] = line[-1]
    elif len(line) == 3:
        # xy-yaw format
        gg[:2] = line[:2]
        gg[-1] = line[-1]
    else:
        raise ValueError('Not supported %d length' % len(p))
    return gg

def load_sequence(sequence_data_dict, feat_suffix='', do_3d=False, noise=None, verbose=True):
    sequence_dict = {}
    with open(sequence_data_dict, 'r') as f:
        j = json.load(f)
    
    #########
    # Poses #
    #########
    poses = np.array([interpret_pose(p, do_3d) for p in j['poses']])
    poses = geometry2.SE2Poses(poses[:, :2], 
                               geometry2.Rotation2.from_euler('z', poses[:, -1]))
    sequence_dict['poses'] = poses
    
    ###############
    # Descriptors #
    ###############
    if feat_suffix != '':
        with h5py.File(sequence_data_dict.replace('.json', feat_suffix), 'r') as f:
            feats = np.copy(f['features']).astype(np.float32)
    else:
        feats = []
    sequence_dict['feats'] = feats
        
    ############
    # Odometry #
    ############
    if 'odom_poses' in j.keys():
        odoms_seq = np.array([interpret_pose(p, do_3d) for p in j['odom_poses']])
        odoms_seq = geometry2.SE2Poses(odoms_seq[:, :2], 
                                       geometry2.Rotation2.from_euler('z', odoms_seq[:, -1]))
        
        # Begin odometry from the origin
        final_odoms = [poses[0]]
        for i in range(len(odoms_seq) - 1):
            final_odoms.append(final_odoms[-1] * (odoms_seq[i] / odoms_seq[i+1]))

        # Estimate noise
        rem = np.array([geometry2.logSE2(poses[i] / poses[i+1]) for i in range(len(poses) - 1)]) -\
              np.array([geometry2.logSE2(final_odoms[i] / final_odoms[i+1]) for i in range(len(final_odoms) - 1)])

        sequence_dict['odoms'] = geometry2.combine(final_odoms)
        sequence_dict['noise'] = np.cov(rem.T)
    else:
        if noise is not None and not do_3d:
            # Create odometry from noisy gt
            if verbose: print('No odom found, but found noise. Creating 2d odometry...')
            
            # Create noied increments
            noisy_increments = np.random.multivariate_normal(np.zeros([3]), 
                                                             noise, 
                                                             len(sequence_dict['poses'])-1)
            # Create the odometry 
            final_odoms = [sequence_dict['poses'][0]]
            for incr_i in range(len(noisy_increments)):
                odom_incr = (sequence_dict['poses'][incr_i] / sequence_dict['poses'][incr_i+1]) *\
                             geometry2.expSE2(noisy_increments[incr_i])
                final_odoms.append(final_odoms[-1] * odom_incr)
                
            sequence_dict['odoms'] = geometry2.combine(final_odoms)
            sequence_dict['noise'] = noise
        else:
            # Void odometry 
            sequence_dict['odoms'] = []
            sequence_dict['noise'] = None
            
    if verbose: print('Loaded ' + ' '.join(["%s: %d" % (k, len(sequence_dict[k])) for k in sequence_dict.keys() if k is not 'noise']))
    return sequence_dict