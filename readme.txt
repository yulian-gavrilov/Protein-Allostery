
EpockGrid2VoidsAllostery
2021-2022. Lund. Sweden.
Author: Yulian Gavrilov

Scientific supervisors: Mikael Akke, Pär Söderhjelm
Biophysical Chemistry, Center for Molecular Protein Science, Department of Chemistry, Lund University, P.O. Box 124, SE-22100
Lund, Sweden
Contact:
yulian.gavrilov@bpc.lu.se
yulian.gavrilov@gmail.com


Python code for the analysis of allosteric communication in proteins based on
the dynamics of the internal protein void clusters.

Required input:
1. protein structure (pdb format) and its molecular dynamics (MD) trajectory (xtc format)
2. coordinates of the internal void probes (pdb and xtc files) obtained with Epock software
for the protein of interest based on the protein MD trajectory.
To obtain the input, it is suggested:
- to run Epock analysis for the whole protein: Free Space Detection Sphere should embed all protein atoms.
- to avoid volume contribution form outside of a protein
include (keep) the water molecules in the input trajectory
(and even to slightly increase water radius using Epock's radii.txt file)

Dependencies:
1. MDAnalysis: to deal with the binary xtc trajectories, atom indices, etc.
2. Scipy spatial distance: very efficient way to measure pairwise distances between two independent sets of coordinates.
3. Numpy, Pands, Matplotlib, etc. (see all the imports in EpockGrid2VoidsAllostery_util.py).

Links:
This software is an attempt to apply previously developed algorithm
for the analysis of the internal protein voids (and not the external pockets).
For more details regarding the algorithm see:
La Sala et al. Allosteric Communication Networks in Proteins Revealed through
Pocket Crosstalk Analysis. ACS Cent. Sci. 2017, 3, 949−96. DOI: http://pubs.acs.org/journal/acscii
Epock software: https://epock.bitbucket.io/index.html

Implementation.
In case of the large input trajectories: long MD runs and large Epock grid trajectories
it can take a long time to run all the steps.
Accordingly the software is split into four independent scripts that cover four
independent steps of the data preparation and analysis:

Step 1.
Conversion of Epock output trajectory of the internal voids in xtc format.
For the compatibility with other software (like VMD) this xtc file contains the
same number of probes (grid points) at each time frame. Even if across the timeframes some
grid points overlap with the protein atoms and do not represent the empty space anymore.
Instead of the real coordinates, the coordinates of the center of the Free Space Detection Sphere (see Epock manual)
are assigned to the probes (grid points) that overlap with the protein atoms (also with water molecules, etc.).
So, on the step 1, all non-empty space grid points are removed from all time frames and
the result is saved as a python list in the npy container file.

Step 2.
Using Epock grid spacing cutoff (~ 0.5 Å) separate internal voids are identified
and defined in terms of the closest (forming) protein atom clusters for each time frame.

Step 3.
- N matrix accounting for the split and merge events of the voids (clusters) is calculated
across the input time frames.
- each cluster gets a unique clusterID that is preserved across the timeframes based on the jaccard index
- the clusters are also redefined in terms of the indices of the forming protein residues;
See La Sala et al. for the algorithm details.

Step 4 (under the development).
Analysis of the obtained data:
- Analysis of the allosteric communion between the clusters. Finding the cluster pairs with the highest number of split/merge events.
- Analysis of the individual cluster volume across the timeframes.
- Further analysis of volume dynamics, allostery, cluster and residue persistence etc. Yet to be developed.
