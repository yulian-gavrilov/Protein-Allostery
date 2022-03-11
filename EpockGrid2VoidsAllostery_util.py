import numpy as np
import pandas as pd
import re
import os
from copy import copy, deepcopy
from sklearn.metrics import jaccard_score
import time
import matplotlib.pyplot as plt
from scipy.spatial import distance as scidist
import MDAnalysis as mda
from scipy.ndimage import gaussian_filter1d
import networkx
from networkx.algorithms.components.connected import connected_components
from function_call import py_get_atom_clusters

'''
EpockGrid2VoidsAllostery. Util function.
Author: Yulian Gavrilov
yulian.gavrilov@bpc.lu.se
yulian.gavrilov@gmail.com

Python code for the analysis of allostric communiction in proteins based on
the dynamics of the internal protein void_clusters.

See readme.txt for the further details.
'''

###UTIL#####

def mkdir( directory ):
    if not os.path.isdir(directory):
        os.mkdir( directory )

#######CLASSES####################3

class Atom_cluster():

    def __init__(self,atoms,grid_points,clusterID):
        self.atoms = atoms
        self.grid_points = grid_points
        self.clusterID = clusterID
        self.total_grid_points = sum(self.grid_points)

#     def merge_cluster(self,cluster):
#         self.atoms += cluster.atoms
#         self.grid_points += cluster.grid_points
#         self.total_grid_points = sum(self.grid_points)


    def merge_cluster(self,cluster):
        for i, j in zip(cluster.atoms, cluster.grid_points):
            if i in self.atoms:
                #next
                self.grid_points += [j]
            else:
                self.atoms += [i]
                self.grid_points += [j]

        self.total_grid_points = sum(self.grid_points)

#     def make_unique(self):
#         self.atoms = list(set(self.atoms))
#         self.grid_points = list(set(self.grid_points)) # !!!
#         self.total_grid_points = sum(self.grid_points)

    def __str__(self):
        return f"Cluster ID: {self.clusterID}\nCluster atoms: {self.atoms}\nTotal grid points: {self.total_grid_points}"


class Atom_clusters():


    def __init__(self):

        self.clusters = np.array([])

    def append_clusters_all(self,unique_atoms_clusters_obj):
        for cluster in unique_atoms_clusters_obj:
            self.clusters = np.append(self.clusters,cluster)

    def append_clusters(self,cluster):
        self.clusters = np.append(self.clusters,cluster)

    def delete_cluster(self,cluster):
        self.clusters = self.clusters[self.clusters!=cluster]

    def delete_clusters(self,clusters):
        self.clusters = [ele for ele in self.clusters if ele not in clusters]

    def __str__(self):
        clusters_show = ''
        for cluster in self.clusters:
            clusters_show += '\n'+cluster.__str__()
        return 'The frame has:' + clusters_show

##############################################################################
#################### FUNCTIONS TO FILTER EPOCK POINTS #########################

def get_epock_points_to_keep(protein_center,cav_frame,pdb_connolly,sphere_radius):

    epock_points_to_keep = np.array([[0,0,0]])

    # distance between all epock and connolly points:
    epock_pnts_connolly_pnts_dist = scidist.cdist(cav_frame,pdb_connolly)
    # closets connolly point to each epock point:
    closest_connolly_pnts_ndx = np.argmin(epock_pnts_connolly_pnts_dist,axis=1)
    # closest connolly points' coordinates:
    closest_connolly_pnt_coord = pdb_connolly[closest_connolly_pnts_ndx]
    # for each epock grid point: distance between the closest connolly point and the protein center:
    center_to_closest_connolly_point_dist = scidist.cdist([protein_center],closest_connolly_pnt_coord)
    # for each epock grid point: distance between the epock grid point point and the protein center
    center_to_epock_points_dist = scidist.cdist([protein_center],cav_frame)
    # keep only the epock points for which the distance to the protein center < the distance to the closest connolly pont minus (connolly) sphere radius (3.5Å)
    dist_diff = center_to_epock_points_dist-(center_to_closest_connolly_point_dist-sphere_radius)
    epock_points_to_keep_ndx = np.argwhere((dist_diff < 0))[:,1]
    # save the coordinates of the chosen epock points
    epock_points_to_keep = cav_frame[epock_points_to_keep_ndx]

    return epock_points_to_keep


##############################################################################
#################### FUNCTIONS TO GET atom_cluster_frames ###################

def grid_point_pairs(cav_frames_in, frame=0, grid_cutoff = 0.5):

    all_dist = scidist.cdist(cav_frames_in[frame], cav_frames_in[frame], 'euclidean')
    all_dist = np.triu(all_dist,1)
    all_pairs = np.argwhere((all_dist <= grid_cutoff) & (all_dist > 0))
    all_pairs+=1

    return all_pairs

def get_clusters(TwoDlist):

    #https://stackoverflow.com/questions/4842613/merge-lists-that-share-common-elements

    l = deepcopy(TwoDlist)

    def to_graph(l):
        G = networkx.Graph()
        for part in l:
            # each sublist is a bunch of nodes
            G.add_nodes_from(part)
            # it also imlies a number of edges:
            G.add_edges_from(to_edges(part))
        return G

    def to_edges(l):
        """
          treat `l` as a Graph and returns it's edges
          to_edges(['a','b','c','d']) -> [(a,b), (b,c),(c,d)]
        """
        it = iter(l)
        last = next(it)

        for current in it:
            yield last, current
            last = current

    G = to_graph(l)
    res = list(connected_components(G))
    resultlist = [list(ele) for ele in res]

    return resultlist

# def get_coord_of_grid_point_in_cluster(a_grid_point_pdb_index,test_all_cav,frame=0):

#     # based on the indices of grid clusters one can find the corresponding coordinates of each grid point in each cluster

#     # a_grid_point = grid_clusters[:][2][0] # a grid point # 0 from the cluster # 2
#     # test_all_cav = Cav_frames()

#     for a_grid_point in range(0,test_all_cav.frames[frame].entries.size): # first frame (i=0), all grid points
#         if (test_all_cav.frames[frame].entries[a_grid_point].index == a_grid_point_pdb_index):
#         # if a grid point from a cluster == one of the grid points in the frame
#         # (find a point and extract its coordinates)
#             return frame, (test_all_cav.frames[frame].entries[a_grid_point].index-1) # -1: return python, not pdb index!!

def closest_atom_to_grid_point(cav_frames_in, u1, frame=0):

    all_pdb_ndx = u1.atoms.indices+1
    all_dist = scidist.cdist(cav_frames_in[frame], u1.trajectory[frame].positions, 'euclidean')

    min_dist_all_grid_points_to_atoms_ndx = np.array([],dtype = int)

    for gridp in range (0,np.size(all_dist,0)):
        temp = np.where(all_dist[gridp] == np.amin(all_dist[gridp]))

        min_dist_all_grid_points_to_atoms_ndx = np.append(min_dist_all_grid_points_to_atoms_ndx, int(all_pdb_ndx[temp][0]))

    return min_dist_all_grid_points_to_atoms_ndx


def get_atom_clusters(grid_clusters,gridPoints2atoms_ndx):

    atom_clusters = [[0 for x in row] for row in grid_clusters]

    for i in range(0,len(grid_clusters)):
        for j in range(0,len(grid_clusters[i])):
            atom_clusters[i][j] = gridPoints2atoms_ndx[grid_clusters[i][j]-1]
    return atom_clusters

def get_atom_dict(atom_clusters):

    all_atoms = [j for sub in atom_clusters for j in sub]
    atom_ngrids_dict = {k:all_atoms.count(k) for k in all_atoms} #my_dict = {i:MyList.count(i) for i in MyList}

    # print(atom_ngrids_dict)
    # print(atom_clusters)

    return atom_ngrids_dict

## olde, not used:
# def grid2atom_clusters_and_atom_dict(grid_clusters, gridPoints2atoms_ndx):
#
#     atom_clusters = deepcopy(grid_clusters)
#     my_dict={}
#     for i in range(0,len(grid_clusters)):
#
#         for j in range(0,len(grid_clusters[i])):
#             atom_clusters[i][j] = gridPoints2atoms_ndx[grid_clusters[i][j]-1]
#
#             #print (atom_clusters[i][j], end = " ")
#
#         #print ("")
#
#     all_atoms = [j for sub in atom_clusters for j in sub]
#     atom_ngrids_dict = {k:all_atoms.count(k) for k in all_atoms} #my_dict = {i:MyList.count(i) for i in MyList}
#
#     # print(atom_ngrids_dict)
#     # print(atom_clusters)
#
#     return atom_clusters, atom_ngrids_dict

def get_unique_atoms_in_grid_clusters_and_number_of_grid_point_in_clusters(atom_clusters):
    grid_cluters_unique_atoms = []
    atom_clusters_grid = np.array([],dtype=int)

    for i in range(0,len(atom_clusters)):
        atom_clusters_grid = np.append(atom_clusters_grid, int(np.size(atom_clusters[:][i])))
        myset = set(atom_clusters[:][i])
        #print (list(myset))
        grid_cluters_unique_atoms.append(list(myset))

    return grid_cluters_unique_atoms, atom_clusters_grid

def get_numb_of_grids_per_atom_in_cluster(unique_atoms_clusters,atom_ngrids_dict):
    grids_per_atom_in_clusters = []
    for cluster in unique_atoms_clusters:
        grids_per_atom = [atom_ngrids_dict[atom] for atom in cluster]
        grids_per_atom_in_clusters.append(grids_per_atom)

    return grids_per_atom_in_clusters

def jaccard(list1, list2):
    intersection = len(list(set(list1).intersection(list2)))
    union = (len(list1) + len(list2)) - intersection
    return float(intersection) / union


def get_atom_res_dict(u1,atom_clusters_frames):
    atom_clusters_frames[0].clusters[0].atoms

    atom_res_keys = []
    atom_res_values = []

    current_atom_ndx=0

    #atom_ndx = u1.atoms[current_atom_ndx].index+1
    for i in range(0,len(u1.atoms)):
        atom_res_keys.append(u1.atoms[i].index+1)
        atom_res_values.append(u1.atoms[i].resid)

    atom_res_dict = dict(zip(atom_res_keys, atom_res_values))
    #print(atom_res_dict)

    return atom_res_dict

def merge_clusters_witin_frame(atom_clusters_frames,nframes):

    for frame_ndx in range(0,nframes):
        clusters_index_to_delete = []
        for i in range(0, len(atom_clusters_frames[frame_ndx].clusters)):
                for j in range(i+1, len(atom_clusters_frames[frame_ndx].clusters)):
                    if atom_clusters_frames[frame_ndx].clusters[i].clusterID == atom_clusters_frames[frame_ndx].clusters[j].clusterID:
                        atom_clusters_frames[frame_ndx].clusters[i].merge_cluster(atom_clusters_frames[frame_ndx].clusters[j])
                        clusters_index_to_delete.append(atom_clusters_frames[frame_ndx].clusters[j])

        atom_clusters_frames[frame_ndx].delete_clusters(clusters_index_to_delete)

    return atom_clusters_frames

def get_atom_clusters_frames_with_res(atom_clusters_frames,atom_res_dict):

    atom_clusters_frames_with_res = deepcopy(atom_clusters_frames)

    for i in range(0, len(atom_clusters_frames_with_res)):
        for j in range(0, len(atom_clusters_frames_with_res[i].clusters)):
            for k in range(0, len(atom_clusters_frames_with_res[i].clusters[j].atoms)):
                atom_clusters_frames_with_res[i].clusters[j].atoms[k] = atom_res_dict[atom_clusters_frames[i].clusters[j].atoms[k]]
                #print (atom_clusters_frames[i].clusters[j].atoms[k], atom_clusters_frames_with_res[i].clusters[j].atoms[k])
    return atom_clusters_frames_with_res

###########################################################################################################

##########################################FUNCTIONS TO GET MERGES, SPLITS and N matrix ###############################################

def get_jaccard_updated_clusters_old(atom_clusters_frames, cluster_storage, compare_frame_ndx = 1):

    # !!! also updates atom_clusters_frames


    cluster_storage_out = deepcopy(cluster_storage)

    JM = np.zeros(shape=(len(atom_clusters_frames[compare_frame_ndx].clusters),len(cluster_storage.clusters)))  # jaccard matrix
    max_Jaccard = np.array([])

    for i in range (0,len(atom_clusters_frames[compare_frame_ndx].clusters)):
        for j in range (0,len(cluster_storage.clusters)):
                temp = jaccard(cluster_storage.clusters[j].atoms, atom_clusters_frames[compare_frame_ndx].clusters[i].atoms)
                JM[i][j] = temp
        max_Jaccard_i = np.where(JM[i] == np.amax(JM[i]))[0][0]
        max_Jaccard = np.append(max_Jaccard,max_Jaccard_i)

        if JM[i,max_Jaccard_i] == 0:
            cluster_storage_out.append_clusters(atom_clusters_frames[compare_frame_ndx].clusters[i])
        else:
            #cluster_storage_out.clusters[max_Jaccard_i].merge_cluster(atom_clusters_frames[compare_frame_ndx].clusters[i])

            atom_clusters_frames[compare_frame_ndx].clusters[i].clusterID = cluster_storage_out.clusters[max_Jaccard_i].clusterID

    return cluster_storage_out, JM

##############################################

def get_jaccard_updated_clusters(atom_clusters_frame, cluster_storage):

    # !!! also updates atom_clusters_frames


    cluster_storage_out = deepcopy(cluster_storage)

    JM = np.zeros(shape=(len(atom_clusters_frame.clusters),len(cluster_storage.clusters)))  # jaccard matrix
    max_Jaccard = np.array([])

    for i in range (0,len(atom_clusters_frame.clusters)):
        for j in range (0,len(cluster_storage.clusters)):
                temp = jaccard(cluster_storage.clusters[j].atoms, atom_clusters_frame.clusters[i].atoms)
                JM[i][j] = temp
        max_Jaccard_i = np.where(JM[i] == np.amax(JM[i]))[0][0]
        max_Jaccard = np.append(max_Jaccard,max_Jaccard_i)

        if JM[i,max_Jaccard_i] == 0:
            cluster_storage_out.append_clusters(atom_clusters_frame.clusters[i])
        else:
            #cluster_storage_out.clusters[max_Jaccard_i].merge_cluster(atom_clusters_frame.clusters[i])

            atom_clusters_frame.clusters[i].clusterID = cluster_storage_out.clusters[max_Jaccard_i].clusterID

    return cluster_storage_out, JM

################################################################################################

def get_jaccard_merge_split_clusters_old(atom_clusters_frames, cluster_storage, compare_frame_ndx = 1):

    cluster_storage_out = deepcopy(cluster_storage)


    JM = np.zeros(shape=(len(atom_clusters_frames[compare_frame_ndx].clusters),len(cluster_storage.clusters)))  # jaccard matrix
    max_Jaccard = np.array([])
    Jrows = np.array([])
    Jcols = np.array([])
    for i in range (0,len(atom_clusters_frames[compare_frame_ndx].clusters)):
        Jrows = np.append(Jrows,atom_clusters_frames[compare_frame_ndx].clusters[i].clusterID)
        for j in range (0,len(cluster_storage.clusters)):
                temp = jaccard(cluster_storage.clusters[j].atoms, atom_clusters_frames[compare_frame_ndx].clusters[i].atoms)
                JM[i][j] = temp
                if i == 0:
                    Jcols = np.append(Jcols,cluster_storage.clusters[j].clusterID)

        max_Jaccard_i = np.where(JM[i] == np.amax(JM[i]))[0][0]
        max_Jaccard = np.append(max_Jaccard,max_Jaccard_i)

        if JM[i,max_Jaccard_i] == 0:
            cluster_storage_out.append_clusters(atom_clusters_frames[compare_frame_ndx].clusters[i])
        else:
            cluster_storage_out.clusters[max_Jaccard_i].merge_cluster(atom_clusters_frames[compare_frame_ndx].clusters[i])

    return cluster_storage_out, JM, Jrows, Jcols

################################

def get_jaccard_merge_split_clusters(atom_clusters_frame, cluster_storage):

    cluster_storage_out = deepcopy(cluster_storage)


    JM = np.zeros(shape=(len(atom_clusters_frame.clusters),len(cluster_storage.clusters)))  # jaccard matrix
    max_Jaccard = np.array([])
    Jrows = np.array([])
    Jcols = np.array([])
    for i in range (0,len(atom_clusters_frame.clusters)):
        Jrows = np.append(Jrows,atom_clusters_frame.clusters[i].clusterID)
        for j in range (0,len(cluster_storage.clusters)):
                temp = jaccard(cluster_storage.clusters[j].atoms, atom_clusters_frame.clusters[i].atoms)
                JM[i][j] = temp
                if i == 0:
                    Jcols = np.append(Jcols,cluster_storage.clusters[j].clusterID)

        max_Jaccard_i = np.where(JM[i] == np.amax(JM[i]))[0][0]
        max_Jaccard = np.append(max_Jaccard,max_Jaccard_i)

        if JM[i,max_Jaccard_i] == 0:
            cluster_storage_out.append_clusters(atom_clusters_frame.clusters[i])
        else:
            cluster_storage_out.clusters[max_Jaccard_i].merge_cluster(atom_clusters_frame.clusters[i])

    return cluster_storage_out, JM, Jrows, Jcols


################################################################################################

def get_current_splits_and_merges(JF,Jrows,Jcols):

    current_merges=[]
    current_splits=[]

    for i in range(0,len(Jcols)): # all columns: all new with one old, splitting
        #JF[:,i] # all new vs one old
        #print("cluster",Jcols[i],"splits to  ",Jrows[JF[:,i]>0])
        for k in Jrows[JF[:,i]>0]:
            #print (Jcols[i],k)
            current_splits.append([Jcols[i],k])
    #print("")
    for j in range(0,len(Jrows)): # all rows: all old with one new, merging
        #JF[j,:] # all old vs one new
        #print("cluster",Jrows[j],"merges with",Jcols[JF[j,:]>0])
        for k in Jcols[JF[j,:]>0]:
            #print (Jrows[j],k)
            current_merges.append([Jrows[j],k])

    #print (current_splits)
    #print (current_merges)

    return current_splits, current_merges


def get_N_matrix(cluster_storage_out,all_merges,all_splits):
    # collect all cluster IDs:
    all_clusters = []
    clusters1 = []
    clusters2 = []
    NMS_linear = []

    for i in range (0,len(cluster_storage_out.clusters)):
        all_clusters.append(cluster_storage_out.clusters[i].clusterID)

    # make the matrix of merges:
    NM = np.zeros((len(all_clusters),len(all_clusters)))
    # make the matrix of splits:
    NS = np.zeros((len(all_clusters),len(all_clusters)))
    # make the matrix of max(merges,splits):
    NMS = np.zeros((len(all_clusters),len(all_clusters)))

    all_clusters_dict = dict(zip(all_clusters, range(len(all_clusters))))

    # print (all_clusters_dict)
    # for i in range(0,len(all_merges)):
    #     for j in range(0,len(all_merges[i])):
    #             print(all_merges[i][j], end = " ")
    #     print("")

    for frame in range(0,len(all_merges)):
        for merge_pair in range(0,len(all_merges[frame])):
            row = all_clusters_dict[all_merges[frame][merge_pair][0]] # in the end always 0 or 1 (pairs)
            col = all_clusters_dict[all_merges[frame][merge_pair][1]] # in the end always 0 or 1 (pairs)

            all_merges[frame][merge_pair][0], all_merges[frame][merge_pair][1]

            NM[row,col] +=1

    for frame in range(0,len(all_splits)):
        for split_pair in range(0,len(all_splits[frame])):
            row = all_clusters_dict[all_splits[frame][split_pair][0]] # in the end always 0 or 1 (pairs)
            col = all_clusters_dict[all_splits[frame][split_pair][1]] # in the end always 0 or 1 (pairs)
            NS[row,col] +=1


    for i in range(0,len(NM)):
        for j in range(0,len(NM[i])):
            if NM[i][j]>NS[i][j]:
                NMS[i][j] = NM[i][j]
            else:
                NMS[i][j] = NS[i][j]

    for i in range(0,len(NMS)):
        for j in range(0,len(NMS[i])):
            cluster1 = list(all_clusters_dict.keys())[list(all_clusters_dict.values()).index(i)]
            cluster2 = list(all_clusters_dict.keys())[list(all_clusters_dict.values()).index(j)]
            if NMS[i][j] > 0:
                clusters1.append(str(cluster1)+"-"+str(cluster2))
                #clusters2.append(cluster2)
                NMS_linear.append(NMS[i][j])
                #print (cluster1, cluster2, NMS[i][j])

        #print ("")

    NMS_dict = dict(zip(clusters1,NMS_linear))


    return NMS, NMS_dict, all_clusters


#################################################################################################
##### ANALYSIS OF THE OUTPUT ########
#################################################################################################


def get_cluster_volume(cluster_of_choice,atom_clusters_frames):
    # Print the volume of a cluster of choice as a function of time:
    #cluster_of_choice = "0_1"
    running_sum = []
    volume_in_frames = []
    for i in range(0, len(atom_clusters_frames)):
        for j in range(0, len(atom_clusters_frames[i].clusters)):
            if atom_clusters_frames[i].clusters[j].clusterID == cluster_of_choice:
                #print (atom_clusters_frames[i].clusters[j].atoms, end = " ")
                #print (atom_clusters_frames[i].clusters[j].clusterID, end = " ")
                #print (atom_clusters_frames[i].clusters[j].total_grid_points, end = " ")
                running_sum.append(atom_clusters_frames[i].clusters[j].total_grid_points)
        temp = sum(running_sum)
        volume_in_frames.append(temp)
        running_sum = []
        #print ("")

    return volume_in_frames

def get_max_split_merge(sort_NMS_dict, split_merge_cutoff = 500):

    # print N matrix elements with max number of splits/merges
    # exclude cases when a cluster meges with itself
    #print("N matrix elements with max number of splits/merges")

    split_merge_count = {}

    temp=['','']
    temp[0] = ""; temp[1] = ""

    sumcl=0

    #cl_of_int = "0_5"
    for i in sort_NMS_dict:
        cl0 = i[0].split("-")[0]
        cl1 = i[0].split("-")[1]

        if float(i[1]) > split_merge_cutoff and cl0 != cl1:
            if cl0!=temp[1] or cl1!=temp[0]:
                #print(i[0], i[1])
                split_merge_count[i[0]] = i[1]

#             if cl0 == cl_of_int or cl1 == cl_of_int:
#                 sumcl+=float(i[1])
            temp[0] = cl0; temp[1] = cl1

#    print("")
#     print(sumcl)
    return split_merge_count

def get_all_split_merge_for_a_cluster(sort_NMS_dict, cl_of_int, split_merge_cutoff = 0):

    # print N matrix elements with max number of splits/merges
    # exclude cases when a cluster meges with itself
    #print("N matrix elements with max number of splits/merges")


    temp=['','']
    temp[0] = ""; temp[1] = ""

    sumcl=0

    for i in sort_NMS_dict:
        cl0 = i[0].split("-")[0]
        cl1 = i[0].split("-")[1]

        if (cl0 == cl_of_int or cl1 == cl_of_int) and cl0 != cl1 and float(i[1]) > split_merge_cutoff:
            #print(i[0], i[1])
            if cl0!=temp[1] or cl1!=temp[0]:
                sumcl+=float(i[1])

            temp[0] = cl0; temp[1] = cl1

#    print("")
#     print(sumcl)
    return sumcl

def print_cluster_volume_and_contacts(atom_clusters_frames,sort_NMS_dict,cl_of_int,volume_cutoff = 2,numb_contacts_cutoff = 100):
    print("Volume (units: number of 1.125 Å cubes)")
    temp=['','']
    temp[0] = ""; temp[1] = ""

    # cl_of_int = "0_5"
    # volume_cutoff = 2 # min protein volume (in grid points) to be considered
    # numb_contacts_cutoff = 100 # minumum number of split/merge events to be considered

    volume_in_frames_cl_of_int=get_cluster_volume(cl_of_int,atom_clusters_frames)
    print("Input cluster:", cl_of_int,": ",np.round(np.mean(volume_in_frames_cl_of_int),2),"±",np.round(np.std(volume_in_frames_cl_of_int),2))
    print("Clusters in contact with the input cluster:")

    for i in sort_NMS_dict:
        cl0 = i[0].split("-")[0]
        cl1 = i[0].split("-")[1]

        if float(i[1]) > numb_contacts_cutoff and cl0 != cl1 and (cl0 == cl_of_int or cl1 == cl_of_int):
            if cl0!=temp[1] or cl1!=temp[0]:
                #print(i[0], i[1])
                if cl0 == cl_of_int:
                    temp = get_cluster_volume(cl1,atom_clusters_frames)
                    if np.mean(temp) > volume_cutoff:
                        print(cl1,": ", np.round(np.mean(temp),2),"±",np.round(np.std(temp),2), end = " ")
                        print("numb of contacts: ",i[0], i[1])
                else:
                    temp = get_cluster_volume(cl0,atom_clusters_frames)
                    if np.mean(temp) > volume_cutoff:
                        print(cl0,": ", np.round(np.mean(temp),2),"±",np.round(np.std(temp),2), end = " ")
                        print("numb of contacts: ",i[0], i[1])

            temp[0] = cl0; temp[1] = cl1

    print("")

def get_cluster_persistency(atom_clusters_frames,cluster_storage_out,nframes,persistency_cutoff_percent=10):

    # get the cluster persistency (number of frames you can find the cluster in)

    cluster_persistency = []
    all_clusterIDs = []
    for i in range(0,len(cluster_storage_out.clusters)):
        all_clusterIDs.append(cluster_storage_out.clusters[i].clusterID)
    #print (len(all_clusterIDs))

    clusters_persistency_dict = dict(zip(all_clusterIDs,np.zeros(len(all_clusterIDs),dtype=int)))

    for frame_ndx in range(0, len(atom_clusters_frames)):
        for i in range(0, len(atom_clusters_frames[frame_ndx].clusters)):
            temp_ID = atom_clusters_frames[frame_ndx].clusters[i].clusterID
            temp_Vol = atom_clusters_frames[frame_ndx].clusters[i].total_grid_points
            clusters_persistency_dict[temp_ID] += 1


    clusters_persistency_dict_sorted = sorted(clusters_persistency_dict.items(), key=lambda x: x[1], reverse=True)

    #persistency_cutoff_percent=10
    all_persistency_clIDs = {}
    all_persistency_clIDs_percent = {}

    persistency_cutoff=nframes*persistency_cutoff_percent/100
    for cluster in clusters_persistency_dict_sorted:
        if cluster[1] >= persistency_cutoff:
            temp_percent = int(round(cluster[1]/nframes*100,0))
            #cluster_persistency.append(f"{cluster} ({temp_percent} %)")
            cluster_persistency.append(cluster)
            #cluster_persistency.append(temp_percent)
            all_persistency_clIDs.update({cluster})

            #all_persistency_clIDs_percent.update({cluster})
            all_persistency_clIDs_percent[cluster[0]] = temp_percent

    print("")
#     print(all_persistency_clIDs)
#     print(all_persistency_clIDs_percent)
    return all_persistency_clIDs, all_persistency_clIDs_percent, cluster_persistency


def get_cluster_group_contacts(atom_clusters_frames,sort_NMS_dict,all_persistency_clIDs_percent):

    # get the number of frames with split/merge events (contacts) between the clusters
    # difference from get_max_split_merge: takes all_persistency_clIDs_percent as an input
    # i.e. consideres only the persistant clusters.
    # the persistancy is defined with persistency_cutoff_percent in get_print_cluster_persistency()

    numb_of_selec_contacts = []
    temp=['','']
    temp[0] = ""; temp[1] = ""

    selected_contacts_keys = []
    selected_contacts_velues = []

    for i in sort_NMS_dict:
        cl0 = i[0].split("-")[0]
        cl1 = i[0].split("-")[1]

        if cl0 != cl1 and cl0 in all_persistency_clIDs_percent.keys() and cl1 in all_persistency_clIDs_percent.keys():
            if cl0!=temp[1] or cl1!=temp[0]:
                numb_of_selec_contacts.append([i[0], i[1]])
                selected_contacts_keys.append(i[0])
                selected_contacts_velues.append(i[1])

            temp[0] = cl0; temp[1] = cl1

    selected_contacts_dict = dict(zip(selected_contacts_keys,selected_contacts_velues))

    #print("")
    return numb_of_selec_contacts, selected_contacts_dict


def get_residues_in_clusters_count(atom_clusters_frames_with_res, nframes, resid = [5,6]):

    # count the occurence of the input residue(s) in all the clusters

    if type(resid) != list:
        resid = [resid]

    res_in_frame_clusters = {}

    for frame_ndx in range(0, nframes):
        for cluster in atom_clusters_frames_with_res[frame_ndx].clusters:
            if any(x in resid for x in cluster.atoms):

                if cluster.clusterID in res_in_frame_clusters:
                    res_in_frame_clusters[cluster.clusterID]+=1
                else:
                    res_in_frame_clusters[cluster.clusterID]=1

    return res_in_frame_clusters

def get_residues_in_frames_count(atom_clusters_frames_with_res, nframes, resid = [5,6]):

    # count number of frames the input residue(s) are present

    if type(resid) != list:
        resid = [resid]

    res_in_frames = 0
    for frame_ndx in range(0, nframes):
        for cluster in atom_clusters_frames_with_res[frame_ndx].clusters:
            if any(x in resid for x in cluster.atoms):
                res_in_frames+=1
                break

    return res_in_frames

def get_res_persistency_in_cluster(atom_clusters_frames_with_res, all_persistency_clIDs, nframes, aclusterID, first_res = 1, last_res = 265):

    # all_persistency_clIDs - number of frames the clusters exist. Used to normalize the residue persistency in a cluster

    res_in_cluster_dict = {}

    for resid in range(first_res,last_res):
        for frame_ndx in range(0, nframes):
            for cluster in atom_clusters_frames_with_res[frame_ndx].clusters:
                if cluster.clusterID == aclusterID and resid in cluster.atoms:

                    if resid in res_in_cluster_dict:
                        res_in_cluster_dict[resid]+=1
                    else:
                        res_in_cluster_dict[resid]=1

    res_in_cluster_percent_dict = {}
    res_in_cluster_abs_percent_dict = {}

    for key in res_in_cluster_dict:
        res_in_cluster_percent_dict[key] = round(res_in_cluster_dict[key]*100/all_persistency_clIDs[aclusterID],2)
        res_in_cluster_abs_percent_dict[key] = round(res_in_cluster_dict[key]*100/nframes,2)
    #     print(res_in_cluster_dict[key],"or", round(res_in_cluster_dict[key]*100/all_persistency_clIDs[aclusterID],2), "%",
    #          "or",round(res_in_cluster_dict[key]*100/nframes,2), "abs % ")

    return res_in_cluster_dict, res_in_cluster_percent_dict, res_in_cluster_abs_percent_dict

def make_pymol_script(all_persistency_clIDs_percent,
                     atom_clusters_frames_with_res,
                     system,nframes,selected_contacts_dict,
                     persistency_cutoff_percent = 10,
                     sphere_radius_scaler = 3,
                     radius_correction = 1500):

    # all_persistency_clIDs_percent
    # atom_clusters_frames_with_res
    # system
    # nframes

    # print("")
    cluster_groups_pml = []
    spheres_strings_pml = []
    cylinders_strings_pml = []

    #sphere_radius_scaler = 3
    cluster_groups_pml.append("bg_color white\n")
    cluster_groups_pml.append("color gray70, all")
    cluster_groups_pml.append("\n\n")

    # spheres
    #https://doc.instantreality.org/tools/color_calculator/
    red = [0.949, 0.152, 0.094]
    orange = [0.921, 0.380, 0.156]
    pink = [0.929, 0.549, 0.564]
    light_blue = [0.549, 0.929, 0.921]
    #blue = [0.247, 0.325, 0.850]
    dark_blue = [0.035, 0.023, 0.815]

    spheres_strings_pml.append("")
    spheres_strings_pml.append("from pymol.cgo import *")
    spheres_strings_pml.append("from pymol import cmd")
    spheres_strings_pml.append("")

    for i in all_persistency_clIDs_percent.keys():
        cl_frame = int(i.split("_")[0])
        cl_index = int(i.split("_")[1])
        #print (cl_frame,cl_index)
        for j in atom_clusters_frames_with_res[cl_frame].clusters:
    #         print (j.clusterID)
    #         print("###")
            if j.clusterID == i:
                cluster_groups_pml.append("select resi")
                cluster_groups_pml.append(" ")
                for res in set(j.atoms):
                    cluster_groups_pml.append(res)
                    cluster_groups_pml.append("+")
                del cluster_groups_pml[-1]
                cluster_groups_pml.append("\n")
                cluster_groups_pml.append("set_name sele, ")
                cluster_groups_pml.append("c")
                cluster_groups_pml.append(i)
                cluster_groups_pml.append("\n")

                sphere_radius = len(set(j.atoms))/sphere_radius_scaler
                spheres_strings_pml.append(f"sphere_c{i} = cmd.centerofmass(\"c{i}\")")
                spheres_strings_pml.append(f"sphere_c{i}.append({sphere_radius})")

                if  all_persistency_clIDs_percent[i] in range (80,100+1):
                    #print (all_persistency_clIDs_percent[i], "red")
                    spheres_strings_pml.append(f"color_c{i} = {red}#red")
                elif all_persistency_clIDs_percent[i] in range (60,80):
                    spheres_strings_pml.append(f"color_c{i} = {orange}#orange")
                elif all_persistency_clIDs_percent[i] in range (40,60):
                    spheres_strings_pml.append(f"color_c{i} = {pink}#pink")
                elif all_persistency_clIDs_percent[i] in range (20,40):
                    spheres_strings_pml.append(f"color_c{i} = {light_blue}#light_blue")
                elif all_persistency_clIDs_percent[i] in range (0,20):
                    spheres_strings_pml.append(f"color_c{i} = {dark_blue}#dark blue")

    #             print(f"sphere_{i} = cmd.centerofmass(\"{i}\")")
    #             print("")

    spheres_strings_pml.append("")
    spheres_strings_pml.append("spherelist = [ \\")

    for i in all_persistency_clIDs_percent.keys():
        spheres_strings_pml.append(f"   COLOR,    color_c{i}[0],color_c{i}[1],color_c{i}[2],\\")
        spheres_strings_pml.append(f"   SPHERE,   sphere_c{i}[0],sphere_c{i}[1],sphere_c{i}[2],sphere_c{i}[3],\\")

    spheres_strings_pml.append("]")
    spheres_strings_pml.append("")

    spheres_strings_pml.append("cmd.load_cgo(spherelist, 'all_spheres',   1)")

    # cylinders
    cylinders_strings_pml.append("")
    #cylinders_strings_pml.append("radius=1")
    #radius_correction = 1500
    for pair in selected_contacts_dict.keys():

        radius = float(selected_contacts_dict[pair])/radius_correction
        cylinders_strings_pml.append(f"radius={radius}")

        cl0 = pair.split("-")[0]
        cl1 = pair.split("-")[1]
        cylinders_strings_pml.append(f"cylinder_{cl0}_{cl1} = [ 9.0, sphere_c{cl0}[0],sphere_c{cl0}[1],sphere_c{cl0}[2], sphere_c{cl1}[0],sphere_c{cl1}[1],sphere_c{cl1}[2],radius,0,0,0,0,0,0 ]")
        cylinders_strings_pml.append(f"cmd.load_cgo(cylinder_{cl0}_{cl1},\"c{cl0}_{cl1}\")")
    # put all the the pml file:

    with open(f'for_pymol_{system}_{nframes}fr_persistency{persistency_cutoff_percent}.pml', 'w') as f:

        for string0 in cluster_groups_pml:
            print(string0, end="", file=f)

        for string1 in spheres_strings_pml:
            print(string1, file=f)

        for string2 in cylinders_strings_pml:
            print(string2, file=f)
