import MDAnalysis as mda
import numpy as np
import pandas as pd
import time
import argparse
import bz2
from EpockGrid2VoidsAllostery_util import mkdir
from EpockGrid2VoidsAllostery_util import get_epock_points_to_keep

'''
EpockGrid2VoidsAllostery. Step 1.
Author: Yulian Gavrilov
yulian.gavrilov@bpc.lu.se
yulian.gavrilov@gmail.com

Python code for the analysis of allostric communiction in proteins based on
the dynamics of the internal protein void_clusters.

See readme.txt for the further details.
'''

### step 1 ####


############## Parser ###############
def parse():

    parser = argparse.ArgumentParser( description = '' )

    # Mandatory Inputs
    parser.add_argument( '--system', type=str, required=True, help='system (protein) name.')
    parser.add_argument( '--cavpdb', type=str, required=True, help='Path to the Epock cav pdb file.')
    parser.add_argument( '--cavxtc', type=str, required=True, help='Path to the Epock cav xtc file.')
    parser.add_argument( '--protpdb', type=str, required=True, help='Path to the protein pdb file.')
    parser.add_argument( '--protxtc', type=str, required=True, help='Path to the protein xtc file.')
    parser.add_argument( '--edtsurf', type=str, required=True, help='Path to the modified EDISurf ply.bz2 output.')
    parser.add_argument( '--out', type=str, required=True, help='Path to the output directory (will be created if not existant).')
    args = parser.parse_args()

    return args.system, args.cavpdb, args.cavxtc, args.protpdb, args.protxtc, args.edtsurf, args.out

########################################

system, cavpdb, cavxtc, protpdb, protxtc, edtsurf, outf = parse()

print('\nInput options:')
print(f'--system {system} --cavpdb {cavpdb} --cavxtc {cavxtc} --protpdb {protpdb} --protxtc {protxtc} --edtsurf {edtsurf} --out {outf}')
mkdir(outf)

u1 = mda.Universe(f'{cavpdb}', f'{cavxtc}')

nframes=len(u1.trajectory)

########
'''
cav_frames=[]
# non unique entries correspond to unrequired non-empty space grid points
for i in range (0,len(u1.trajectory)):
    temp = np.unique(u1.trajectory[i].positions,axis=0)
    cav_frames.append(temp)

print ("selected unique grid points in all frames...")

with open(f'{outf}/{system}_unique_cav_{nframes}fr.npy', 'wb') as f:
    np.save(f, cav_frames)

#######

with open(f'{outf}/{system}_unique_cav_{nframes}fr.npy', 'rb') as f:
    cav_frames = np.load(f,allow_pickle=True)
'''
#######

u2 = mda.Universe(f'{protpdb}', f'{protxtc}')
nframes=len(u2.trajectory)

######################################################

sphere_radius=3.5
f = bz2.open(f"{edtsurf}", "rt")

#pdb_connolly_current = []
pdb_connolly_current=np.empty((0, 3), float)
epock_points_in_frames = []

start_total_time = time.time()


frame = 0
while True:

    line_raw = f.readline()
    line = line_raw.split(" ")

    if "frame" in line and line[1] == '0\n':
        next

    elif "frame" in line and line[1] != '0\n':
        #print (frame, len(pdb_connolly_current))

        # do all the stuff here
        #####
        start_time = time.time()

        cav_frame = np.unique(u1.trajectory[frame].positions,axis=0)
        #print(f"unique frame {frame}")

        u2.trajectory[frame]
        protein_center = u2.select_atoms("resid 0-263").centroid()

        epock_points_to_keep = get_epock_points_to_keep(protein_center,cav_frame,pdb_connolly_current,sphere_radius)
        epock_points_in_frames.append(epock_points_to_keep)

        print(f"frame {frame} --- %s seconds ---" % (time.time() - start_time))
        ####

        # do in the end of iteration:
        #pdb_connolly_current = []
        pdb_connolly_current=np.empty((0, 3), float)
        frame+=1
        next

    elif not line_raw:
        #print (frame, len(pdb_connolly_current))

        # do all the stuff here for the last frame:
        #####
        start_time = time.time()

        cav_frame = np.unique(u1.trajectory[frame].positions,axis=0)
        #print(f"unique frame {frame}")

        u2.trajectory[frame]
        protein_center = u2.select_atoms("resid 0-263").centroid()

        epock_points_to_keep = get_epock_points_to_keep(protein_center,cav_frame,pdb_connolly_current,sphere_radius)
        epock_points_in_frames.append(epock_points_to_keep)

        print(f"frame {frame} --- %s seconds ---" % (time.time() - start_time))
        ####

        break

    else:
        for i in range(len(line)):
            line[i]=float(line[i])
        #pdb_connolly_current.append(line)
        pdb_connolly_current = np.append(pdb_connolly_current, np.array([line]), axis=0)
        #print (pdb_connolly_current)

f.close()
print("--- total %s seconds ---" % (time.time() - start_total_time))


#####################################################

with open(f'{outf}/{system}_grid_points_probe_{sphere_radius}A_{nframes}fr.npy', 'wb') as f:
    np.save(f,epock_points_in_frames)
