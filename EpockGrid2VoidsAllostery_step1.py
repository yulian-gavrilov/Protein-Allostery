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

cav_frames=[]
# non unique entries correspond to unrequired non-empty space grid points
for i in range (0,len(u1.trajectory)):
    temp = np.unique(u1.trajectory[i].positions,axis=0)
    cav_frames.append(temp)


# with open(f'{outf}/{system}_unique_cav_{nframes}fr.npy', 'wb') as f:
#     np.save(f, cav_frames)

u2 = mda.Universe(f'{protpdb}', f'{protxtc}')
nframes=len(u2.trajectory)

######################################################

sphere_radius=3.5
df = pd.read_csv(f"{edtsurf}", compression='bz2',header=None, sep="\s+",engine='python')
df.reset_index(level=0, inplace=True)

pdb_connolly_all = []

end_frame_ndx = df[df["index"].str.contains("frame")].index

i=1
for j in end_frame_ndx[1:]:
    pdb_connolly_all.append(df[i:j].astype(float).values)
    i = j + 1
pdb_connolly_all.append(df[j+1:-1].astype(float).values)

#####################################################

epock_points_in_frames = []
u2.trajectory[frame]
protein_center = u2.select_atoms("resid 0-263").centroid()
#protein_center = u2.select_atoms("protein").centroid()

start_total_time = time.time()

for frame in range(0,nframes):
    start_time = time.time()
    epock_points_to_keep = get_epock_points_to_keep(protein_center,cav_frames[frame],pdb_connolly_all[frame],sphere_radius)


    epock_points_in_frames.append(epock_points_to_keep)
    #print(f"frame {frame} --- %s seconds ---" % (time.time() - start_time))

print("--- total %s seconds ---" % (time.time() - start_total_time))

#####################################################

with open(f'{outf}/{system}_grid_points_probe_{sphere_radius}A_{nframes}fr.npy', 'wb') as f:
    np.save(f,epock_points_in_frames)
