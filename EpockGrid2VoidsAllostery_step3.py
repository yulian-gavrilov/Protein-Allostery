import numpy as np
import pandas as pd
from copy import copy, deepcopy
from sklearn.metrics import jaccard_score
import time
import matplotlib.pyplot as plt
from scipy.spatial import distance as scidist
import MDAnalysis as mda
from EpockGrid2VoidsAllostery_util import *
import argparse
import fnmatch
import os


'''
EpockGrid2VoidsAllostery. Step 3.
Author: Yulian Gavrilov
yulian.gavrilov@bpc.lu.se
yulian.gavrilov@gmail.com

Python code for the analysis of allostric communiction in proteins based on
the dynamics of the internal protein void_clusters.

See readme.txt for the further details.
'''

################################################################################

############## Parser ###############
def parse():

    parser = argparse.ArgumentParser( description = '' )

    # Mandatory Inputs
    parser.add_argument( '--system', type=str, required=True, help='system (protein) name.')
    parser.add_argument( '--input', type=str, required=True, help='Path to the step2 output folder (used as an input here)')
    parser.add_argument( '--out', type=str, required=True, help='Path to the output directory (will be created if not existant).')

    args = parser.parse_args()

    return args.system, args.input, args.out


########################################
system, inf, outf = parse()

print('\nInput options:')
print(f'--system {system} --in {inf} --out {outf}')
mkdir(outf)
################################################################################

#system="dibC_wt"
#u1 = mda.Universe(f'./md3us_{system}_fitCav_dt100_prot_cut_nolig_fr0.pdb',f'./md3us_{system}_fitCav_dt100_prot_cut_nolig.xtc')
#nframes=len(u1.trajectory)

atom_clusters_frames_file = fnmatch.filter(os.listdir(f'{inf}'), '*_atom_clusters_frames*.npy')
#print(atom_clusters_frames_file[0])

#with open(f'{inf}/{system}_atom_clusters_frames{nframes}.npy', 'rb') as f:
with open(f'{inf}/{atom_clusters_frames_file[0]}', 'rb') as f:
    atom_clusters_frames = np.load(f,allow_pickle=True)
nframes=len(atom_clusters_frames)#28498

with open(f'{inf}/{system}_atom_res_dict.npy', 'rb') as f:
    atom_res_dict = np.load(f,allow_pickle=True).item()
#atom_res_dict=atom_res_dict[0:nframes]

# show all accumulated clusters (in the last frame):
frame_ndx=nframes-1
for i in range(0, len(atom_clusters_frames[frame_ndx].clusters)):
    print (atom_clusters_frames[frame_ndx].clusters[i])

################################################################################

######################### MAIN FLOW ############################################

# update atom_clusters_frames clusterID values:

#atom_clusters_frames_reserved = deepcopy(atom_clusters_frames)
all_cluster_storage_out = [atom_clusters_frames[0]]
print("atom_clusters_frames is updating now ...")
# after this loop clusterID values in atom_clusters_frames will be changed.
# the indices of all atoms in the clusters from frame X are compared
# to the indices of all atoms in the cluster from the next frame X+1
# if there is an intersection with the old cluster (from the previous frame X)
# the same clusterID is given to a new cluster in the frame X+1
# if there is an intersection with the several clusters,
# the indexID of a new cluster is decided based on max jaccard indexes between old and new clusters
# Otherwise, if there is no intersection, the cluster in frame X+1 keeps its original clusterID.
# Important! Cluster forming atoms are not updated at this step. no merges/splits.
for i in range(1,nframes):

    if i % 1000 == 0:
        print("step: ",i)

    if i == 1:
        cluster_storage_out_temp, _ = get_jaccard_updated_clusters(atom_clusters_frames, atom_clusters_frames[0], i) # relative to all stored clusters
        all_cluster_storage_out.append(cluster_storage_out_temp)
    else:
        cluster_storage_out, _ = get_jaccard_updated_clusters(atom_clusters_frames, cluster_storage_out_temp, i) # relative to all stored clusters
        cluster_storage_out_temp = cluster_storage_out
        all_cluster_storage_out.append(cluster_storage_out_temp)

print ("atom_clusters_frames is updated now...")

###########################
# get split and merge events
# the comparison between the clusters (using jaccard index) is done only between two adjacent cav_frames
# In the end of the loop we get two variables: all_merges,all_splits
# that will be used to construct N matrix together with
# the last version of cluster_storage_out from the previous loop
# it contains all the clusters across the time frames. Merge/split events were not considered
# for the clusters inside these variable. Thus, the clusters are presented in a form as they appear first (a some time frame).

all_splits = []
all_merges = []
all_cluster_split_merge = [atom_clusters_frames[0]]
print("getting split/merge events now ...")

for i in range(1,nframes):

    if i % 1000 == 0:
        print("step: ",i)

    if i == 1:
        cluster_split_merge, JF, Jrows, Jcols  = get_jaccard_merge_split_clusters(atom_clusters_frames, atom_clusters_frames[0], i)

        current_splits, current_merges = get_current_splits_and_merges(JF,Jrows,Jcols)
        all_splits.append(current_splits)
        all_merges.append(current_merges)
        all_cluster_split_merge.append(cluster_split_merge)

    else:
        cluster_split_merge, JF, Jrows, Jcols  = get_jaccard_merge_split_clusters(atom_clusters_frames, atom_clusters_frames[i-1], i)

        current_splits, current_merges = get_current_splits_and_merges(JF,Jrows,Jcols)
        all_splits.append(current_splits)
        all_merges.append(current_merges)
        all_cluster_split_merge.append(cluster_split_merge)

###############
print ("got all split/merge events")

# If the clusters share the same ID within one time frame, they are merged
# It (sam ID) can happen if the intersect the most with the same cluster from some previous frames
# This merging is not used to count split/merges events across the time frames
# In fact it is not necessary and it does not affect cluster volume calculations
# (which would sum the volume of the clusters with the same ID at one frame anyway)
# Tt is not necessary and doe not affect the further calculations but can be useful in the end during the analysis
# of the clusters at the selected frames:
print ("merging clusters with the same IDs within one frame")
merge_clusters_witin_frame(atom_clusters_frames,nframes)

# get split/mege N matrix (see the algorithm description elsewhere):
print ("making N matrix and N dict ...")
NMS, NMS_dict, all_clusters = get_N_matrix(cluster_storage_out,all_merges,all_splits)

# store the N matrix in a form of a sorted dictionary:
# keys - a pair of splitting/merging clusters, values - number of split/merge events.
# Sorted based on values.
sort_NMS_dict = sorted(NMS_dict.items(), key=lambda x: x[1], reverse=True)

print ("done with N matrix, dict")
# get cluster forming residues

# substitute atom indices in the clusters with the residue indices:
print ("run get_atom_clusters_frames_with_res")
atom_clusters_frames_with_res = get_atom_clusters_frames_with_res(atom_clusters_frames,atom_res_dict)
print ("done with get_atom_clusters_frames_with_res")

################################################################################
# save here:
# atom_clusters_frames (after reindexing)

print("saving now...")

with open(f'{outf}/{system}_atom_clusters_frames{nframes}_reindex.npy', 'wb') as f:
    np.save(f, atom_clusters_frames)
# atom_clusters_frames_with_res
with open(f'{outf}/{system}_atom_clusters_frames{nframes}_reindex_resindex.npy', 'wb') as f:
    np.save(f, atom_clusters_frames_with_res)
# NMS
with open(f'{outf}/{system}_Nmatrix_frames{nframes}.npy', 'wb') as f:
    np.save(f, NMS)
# sort_NMS_dict
with open(f'{outf}/{system}_Nmatrix_dict_frames{nframes}.npy', 'wb') as f:
    np.save(f, sort_NMS_dict)
# cluster_storage_out
with open(f'{outf}/{system}_all_clusters_storage_frames{nframes}.npy', 'wb') as f:
    np.save(f, cluster_storage_out)

print("saved...")
#
################################################################################
