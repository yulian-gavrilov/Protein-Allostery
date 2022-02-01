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

'''
EpockGrid2VoidsAllostery. Step 2.
Author: Yulian Gavrilov
yulian.gavrilov@bpc.lu.se
yulian.gavrilov@gmail.com

Python code for the analysis of allostric communiction in proteins based on
the dynamics of the internal protein void_clusters.

See readme.txt for the further details.
'''

#####################################
############## Parser ###############
def parse():

    parser = argparse.ArgumentParser( description = '' )

    # Mandatory Inputs
    parser.add_argument( '--system', type=str, required=True, help='system (protein) name.')
    parser.add_argument( '--pdb', type=str, required=True, help='Path to the protein pdb file.')
    parser.add_argument( '--xtc', type=str, required=True, help='Path to the protein xtc file.')
    parser.add_argument( '--grid', type=str,required=True, help='Path to the processed (step1) Epock cavity trajectory (*.npy).')
    parser.add_argument( '--out', type=str, required=True, help='Path to the output directory (will be created if not existant).')

    args = parser.parse_args()

    return args.system, args.pdb, args.xtc, args.grid, args.out


########################################

system, protpdb, protxtc, cavgrid, outf = parse()

print('\nInput options:')
print(f'--system {system} --pdb {protpdb} --xtc {protxtc} --grid {cavgrid} --out {outf}')
mkdir(outf)
######### INPUTS #################

#system="dibC_wt"
#u1 = mda.Universe(f'./md3us_{system}_fitCav_dt100_prot_cut_nolig_fr0.pdb', f'./md3us_{system}_fitCav_dt100_prot_cut_nolig.xtc')
u1 = mda.Universe(f'{protpdb}', f'{protxtc}')
nframes=len(u1.trajectory)

# with open(f'./{system}_unique_cav_{nframes}fr.npy', 'rb') as f:
#     cav_frames_in = np.load(f,allow_pickle=True)
with open(f'{cavgrid}', 'rb') as f:
    cav_frames_in = np.load(f,allow_pickle=True)

################## MAIN FLOW: create atom_clusters_frames #######################


atom_clusters_frames = np.array([])
time_tot=0

for i in range(0,nframes):

    start_time = time.time()

    # find all grid points pairs with Epock based cutoff
    # further it will be the basis of splitting free space into separate
    # internal voids (clusters):
    cav_pairs  = grid_point_pairs(cav_frames_in, frame=i, grid_cutoff = 0.5)
    # grid cavities pairs (cav_pairs) are merged into individual clusters separated by the distance > grid_cutoff:
    grid_clusters = get_clusters(cav_pairs)
    # find the indices of the closest atoms to the each grid point:
    gridPoints2atoms_ndx = closest_atom_to_grid_point(cav_frames_in, u1, frame=i)

    # redefine the clusters in terms of the closest atoms. I.e. substitute grid indices with atom indices.
    # create a dictionary:
    # key - atom, values - number of times this atom appears (i.e. number of corresponding grid points):
    # atom_clusters, atom_ngrids_dict = grid2atom_clusters_and_atom_dict(grid_clusters, gridPoints2atoms_ndx)
    atom_clusters = py_get_atom_clusters(grid_clusters,gridPoints2atoms_ndx)
    #atom_clusters = get_atom_clusters(grid_clusters,gridPoints2atoms_ndx)
    atom_ngrids_dict = get_atom_dict(atom_clusters)
    
    # function name tells for itself:
    grid_cluters_unique_atoms, atom_clusters_grid = get_unique_atoms_in_grid_clusters_and_number_of_grid_point_in_clusters(atom_clusters)
    # redefine the clusters using the same algorithm that was used to define grid points- based clusters:
    unique_atoms_clusters = get_clusters(grid_cluters_unique_atoms)

    # function name tells for itself:
    grids_per_atom_in_clusters = get_numb_of_grids_per_atom_in_cluster(unique_atoms_clusters,atom_ngrids_dict)

    # for each atom based cluster create Atom_cluster object that includes:
    # cluster forming atoms; total number of grid points in the cluster, initial cluster ID in th format: X_Y
    # where X - time frame number; Y - cluster number within this frame:
    unique_atoms_clusters_obj = np.array([])
    for clusterID in range (0,len(unique_atoms_clusters)):
        temp_atm_cluster = Atom_cluster(unique_atoms_clusters[clusterID],grids_per_atom_in_clusters[clusterID],str(i)+"_"+str(clusterID))
        unique_atoms_clusters_obj = np.append(unique_atoms_clusters_obj, temp_atm_cluster)
        clusterID+=1

    unique_atoms_clusters=[]
    grids_per_atom_in_clusters=[]
    # collect all clusters of a frame into one object:
    atom_clusters = Atom_clusters()
    atom_clusters.append_clusters_all(unique_atoms_clusters_obj)
    gridPoints2atoms_ndx = []
    # collect all Atom_clusters objects into one array:
    atom_clusters_frames = np.append(atom_clusters_frames, atom_clusters)

    time_temp = time.time() - start_time
    print("frame {} --- {:6.2f} seconds ---".format(i,time_temp))
    time_tot+=time_temp

print("total time: --- {:6.2f} seconds ---".format(time_tot))


#################################################################################

# create a dictionary. keys - residue indices, values - corresponding residue atoms indices:
atom_res_dict = get_atom_res_dict(u1,atom_clusters_frames)

# save the collections required on the next step:
with open(f'{outf}/{system}_atom_clusters_frames{nframes}.npy', 'wb') as f:
    np.save(f, atom_clusters_frames)

with open(f'{outf}/{system}_atom_res_dict.npy', 'wb') as f:
    np.save(f, atom_res_dict)


#################################################################################
