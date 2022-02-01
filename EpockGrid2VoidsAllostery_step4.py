import numpy as np
import pandas as pd
from copy import copy, deepcopy
from sklearn.metrics import jaccard_score
import time
import matplotlib.pyplot as plt
from scipy.spatial import distance as scidist
import MDAnalysis as mda
from EpockGrid2VoidsAllostery_util import *
from scipy.ndimage import gaussian_filter1d
import argparse
import fnmatch
import os

'''
EpockGrid2VoidsAllostery. Step 4.
Author: Yulian Gavrilov
yulian.gavrilov@bpc.lu.se
yulian.gavrilov@gmail.com

Python code for the analysis of allostric communiction in proteins based on
the dynamics of the internal protein void_clusters.

See readme.txt for the further details.
'''

###############################################################################
############## Parser ###############
def parse():

    parser = argparse.ArgumentParser( description = '' )

    # Mandatory Inputs
    parser.add_argument( '--system', type=str, required=True, help='system (protein) name.')
    parser.add_argument( '--input', type=str, required=True, help='Path to the step3 output folder (used as an input here)')
    parser.add_argument( '--out', type=str, required=True, help='Path to the output directory (will be created if not existant).')

    args = parser.parse_args()

    return args.system, args.input, args.out


########################################
system, inf, outf = parse()

print('\nInput options:')
print(f'--system {system} --in {inf} --out {outf}')
mkdir(outf)

# system="dibC_wt"
# u1 = mda.Universe(f'./md3us_{system}_fitCav_dt100_prot_cut_nolig_fr0.pdb', f'./md3us_{system}_fitCav_dt100_prot_cut_nolig.xtc')
# nframes=len(u1.trajectory)

################################################################################

################################################################################
# load here:
# atom_clusters_frames (after reindexing)
atom_clusters_frames_file = fnmatch.filter(os.listdir(f'{inf}'), '*_atom_clusters_frames*_reindex.npy')

#with open(f'{inf}/{system}_atom_clusters_frames{nframes}_reindex.npy', 'rb') as f:
with open(f'{inf}/{atom_clusters_frames_file[0]}', 'rb') as f:
    atom_clusters_frames = np.load(f, allow_pickle=True)
nframes=len(atom_clusters_frames)#28498
# atom_clusters_frames_with_res
with open(f'{inf}/{system}_atom_clusters_frames{nframes}_reindex_resindex.npy', 'rb') as f:
    atom_clusters_frames_with_res = np.load(f, allow_pickle=True)
# NMS
with open(f'{inf}/{system}_Nmatrix_frames{nframes}.npy', 'rb') as f:
    NMS = np.load(f, allow_pickle=True)
# sort_NMS_dict
with open(f'{inf}/{system}_Nmatrix_dict_frames{nframes}.npy', 'rb') as f:
    sort_NMS_dict = np.load(f, allow_pickle=True)#.item()

with open(f'{inf}/{system}_all_clusters_storage_frames{nframes}.npy', 'rb') as f:
    cluster_storage_out = np.load(f, allow_pickle=True).item()
#
################################################################################
#f = open(f"{system}_step4.log", "w")


# show all accumulated clusters (in the last frame)
# after reindexing

print("\nShow all clusters (in the last frame) after reindexing")
frame_ndx=nframes-1
for i in range(0, len(atom_clusters_frames[frame_ndx].clusters)):
    print (atom_clusters_frames[frame_ndx].clusters[i])
print("\n\n")
####
print("\nShow all accumulated clusters (in the last frame) after reindexing")
for i in range(0,len(cluster_storage_out.clusters)):
    print (cluster_storage_out.clusters[i].clusterID, end = " ")
print("\n",i)

# print N matrix elements with max number of splits/merges
# exclude cases when a cluster merges with itself
print("N matrix elements with max number of splits/merges")
temp=['','']
temp[0] = ""; temp[1] = ""
for i in sort_NMS_dict:
    cl0 = i[0].split("-")[0]
    cl1 = i[0].split("-")[1]

    if float(i[1]) > 500 and cl0 != cl1:
        if cl0!=temp[1] or cl1!=temp[0]:
            print(i[0], i[1])
    temp[0] = cl0; temp[1] = cl1

print("\n")

####

# show all clusters (in the last frame)
# after reindexing
# after substituting atom indices with residues' indices
print("show all clusters (in the last frame) after reindexing; after substituting atom indices with residues' indices")
frame_ndx=nframes-1
for i in range(0, len(atom_clusters_frames_with_res[frame_ndx].clusters)):
    print (atom_clusters_frames_with_res[frame_ndx].clusters[i])
print("\n\n")
####

# # print all clusters' IDs
#
# for i in range(0,len(cluster_storage_out.clusters)):
#     print (cluster_storage_out.clusters[i].clusterID, end = " ")
# print("\n",i)


####
# GET CLUSTERS' VOLUME

volume_in_frames_0_5 = get_cluster_volume("0_5",atom_clusters_frames)
volume_in_frames_0_1 = get_cluster_volume("0_1",atom_clusters_frames)
volume_in_frames_0_2 = get_cluster_volume("0_2",atom_clusters_frames)
volume_in_frames_0_6 = get_cluster_volume("0_6",atom_clusters_frames)
volume_in_frames_0_10 = get_cluster_volume("0_10",atom_clusters_frames)
volume_in_frames_4_4 = get_cluster_volume("4_4",atom_clusters_frames)

#f.close()
####
# PLOT CLUSTERS' VOLUME

#https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.gaussian_filter1d.html


sigma = 3 # gaussian_filter1d

fig, ax = plt.subplots(figsize=(6,4),dpi=100)

colors=['forestgreen','lime','blue','cornflowerblue']

ax.plot(gaussian_filter1d(volume_in_frames_0_5,sigma),color='g',linestyle='-')
ax.plot(gaussian_filter1d(volume_in_frames_0_1,sigma),color='b',linestyle='-')
ax.plot(gaussian_filter1d(volume_in_frames_0_2,sigma),color='y',linestyle='-')
ax.plot(gaussian_filter1d(volume_in_frames_0_6,sigma),color='m',linestyle='-')
ax.plot(gaussian_filter1d(volume_in_frames_0_10,sigma),color='cyan',linestyle='-')
ax.plot(gaussian_filter1d(volume_in_frames_4_4,sigma),color='orange',linestyle='-')

plt.legend(['cluster 0_5','cluster 0_1','cluster 0_2','cluster 0_6','cluster 0_10','cluster 4_4'],
loc='upper right',prop={"size":10},ncol=3, bbox_to_anchor=(1, -0.15))
#plt.legend(['cluster 0_0','cluster 0_4','cluster 0_6','cluster 0_10','cluster 4_4'],loc='upper right',prop={"size":10})
#plt.legend(['cluster 0_1'],loc='upper right',prop={"size":10})
plt.ylabel('Volume (number of probe spheres)',fontsize=10)
plt.xlabel('frame',fontsize=10);

plt.title( 'cort_wt. Change in volume of the internal voids clusters', fontsize = 10);

#plt.ylim([0, 300])

#plt.xticks(bins_arr[::1]-10/2,bins_arr[::1],fontsize=7, rotation=45)
#plt.yticks(range(0,21,5),fontsize=15);

plt.savefig(f'{outf}/void_clusters_{system}_dt100_{nframes}fr_clusters.pdf', bbox_inches = "tight");
