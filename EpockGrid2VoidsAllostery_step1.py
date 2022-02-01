import MDAnalysis as mda
import numpy as np
import argparse
from EpockGrid2VoidsAllostery_util import mkdir

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
    parser.add_argument( '--pdb', type=str, required=True, help='Path to the Epock cav pdb file.')
    parser.add_argument( '--xtc', type=str, required=True, help='Path to the Epock cav xtc file.')
    parser.add_argument( '--out', type=str, required=True, help='Path to the output directory (will be created if not existant).')
    args = parser.parse_args()

    return args.system, args.pdb, args.xtc, args.out

########################################

system, cavpdb, cavxtc, outf = parse()

print('\nInput options:')
print(f'--system {system} --pdb {cavpdb} --xtc {cavxtc} --out {outf}')
mkdir(outf)
#system='dibC_wt'
#u1 = mda.Universe('./cav.pdb', './cav.xtc')
u1 = mda.Universe(f'{cavpdb}', f'{cavxtc}')

nframes=len(u1.trajectory)

cav_frames=[]
# non unique entries correspond to unrequired non-empty space grid points
for i in range (0,len(u1.trajectory)):
    temp = np.unique(u1.trajectory[i].positions,axis=0)
    cav_frames.append(temp)


with open(f'{outf}/{system}_unique_cav_{nframes}fr.npy', 'wb') as f:
    np.save(f, cav_frames)
