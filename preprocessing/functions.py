import trimesh
import os
import sys
import numpy as np
from numpy.matlib import repmat
from numpy.linalg import norm
import shutil
import pymesh
from Bio.PDB import * 
import Bio.PDB
from Bio.PDB import * 
from rdkit import Chem
import warnings
warnings.filterwarnings("ignore")
from IPython.utils import io
from sklearn.neighbors import KDTree
from scipy.spatial import distance
import random
from subprocess import Popen, PIPE
from rdkit.Chem import AllChem, Draw, Descriptors, rdMolTransforms
import rdkit.Chem as Chem
import rdkit.Chem.rdMolDescriptors as rdMolDescriptors
import rdkit.Chem.EState as EState
import rdkit.Chem.rdPartialCharges as rdPartialCharges
import rdkit.Chem.rdChemReactions as rdRxns
import copy
att_dtype = np.float32

import networkx as nx

# basename = '4zzd'
# target_filename = basename+'_protein.pdb' 
# ligand_filename = basename+'_ligand.mol2'
BASE_PATH = '/home/dell4/king/202112_graphDrug/'

PDB2PQR_BIN = 'pdb2pqr30'
APBS_BIN = '/home/dell4/softwares/APBS-3.4.1.Linux/bin/apbs'  
MULTIVALUE_BIN = '/home/dell4/softwares/APBS-3.4.1.Linux/share/apbs/tools/bin/multivalue' 

msms_bin = 'msms'    
eps = 1.0e-6
tempdir=BASE_PATH+'mark3/temp/'

# chemistry.py: Chemical parameters for MaSIF.
# Pablo Gainza - LPDI STI EPFL 2018-2019
# Released under an Apache License 2.0

# radii for atoms in explicit case.
radii = {}
radii["N"] = "1.540000"
radii["N"] = "1.540000"
radii["O"] = "1.400000"
radii["C"] = "1.740000"
radii["H"] = "1.200000"
radii["S"] = "1.800000"
radii["P"] = "1.800000"
radii["Z"] = "1.39"
radii["X"] = "0.770000"  ## Radii of CB or CA in disembodied case.
# This  polar hydrogen's names correspond to that of the program Reduce. 
polarHydrogens = {}
polarHydrogens["ALA"] = ["H"]
polarHydrogens["GLY"] = ["H"]
polarHydrogens["SER"] = ["H", "HG"]
polarHydrogens["THR"] = ["H", "HG1"]
polarHydrogens["LEU"] = ["H"]
polarHydrogens["ILE"] = ["H"]
polarHydrogens["VAL"] = ["H"]
polarHydrogens["ASN"] = ["H", "HD21", "HD22"]
polarHydrogens["GLN"] = ["H", "HE21", "HE22"]
polarHydrogens["ARG"] = ["H", "HH11", "HH12", "HH21", "HH22", "HE"]
polarHydrogens["HIS"] = ["H", "HD1", "HE2"]
polarHydrogens["TRP"] = ["H", "HE1"]
polarHydrogens["PHE"] = ["H"]
polarHydrogens["TYR"] = ["H", "HH"]
polarHydrogens["GLU"] = ["H"]
polarHydrogens["ASP"] = ["H"]
polarHydrogens["LYS"] = ["H", "HZ1", "HZ2", "HZ3"]
polarHydrogens["PRO"] = []
polarHydrogens["CYS"] = ["H"]
polarHydrogens["MET"] = ["H"]

hbond_std_dev = np.pi / 3

# Dictionary from an acceptor atom to its directly bonded atom on which to
# compute the angle.
acceptorAngleAtom = {}
acceptorAngleAtom["O"] = "C"
acceptorAngleAtom["O1"] = "C"
acceptorAngleAtom["O2"] = "C"
acceptorAngleAtom["OXT"] = "C"
# Dictionary from acceptor atom to a third atom on which to compute the plane.
acceptorPlaneAtom = {}
acceptorPlaneAtom["O"] = "CA"
# Dictionary from an H atom to its donor atom.
donorAtom = {}
donorAtom["H"] = "N"
# Hydrogen bond information.
# ARG
# ARG NHX
# Angle: NH1, HH1X, point and NH2, HH2X, point 180 degrees.
# radii from HH: radii[H]
# ARG NE
# Angle: ~ 120 NE, HE, point, 180 degrees
donorAtom["HH11"] = "NH1"
donorAtom["HH12"] = "NH1"
donorAtom["HH21"] = "NH2"
donorAtom["HH22"] = "NH2"
donorAtom["HE"] = "NE"

# ASN
# Angle ND2,HD2X: 180
# Plane: CG,ND2,OD1
# Angle CG-OD1-X: 120
donorAtom["HD21"] = "ND2"
donorAtom["HD22"] = "ND2"
# ASN Acceptor
acceptorAngleAtom["OD1"] = "CG"
acceptorPlaneAtom["OD1"] = "CB"

# ASP
# Plane: CB-CG-OD1
# Angle CG-ODX-point: 120
acceptorAngleAtom["OD2"] = "CG"
acceptorPlaneAtom["OD2"] = "CB"

# GLU
# PLANE: CD-OE1-OE2
# ANGLE: CD-OEX: 120
# GLN
# PLANE: CD-OE1-NE2
# Angle NE2,HE2X: 180
# ANGLE: CD-OE1: 120
donorAtom["HE21"] = "NE2"
donorAtom["HE22"] = "NE2"
acceptorAngleAtom["OE1"] = "CD"
acceptorAngleAtom["OE2"] = "CD"
acceptorPlaneAtom["OE1"] = "CG"
acceptorPlaneAtom["OE2"] = "CG"

# HIS Acceptors: ND1, NE2
# Plane ND1-CE1-NE2
# Angle: ND1-CE1 : 125.5
# Angle: NE2-CE1 : 125.5
acceptorAngleAtom["ND1"] = "CE1"
acceptorAngleAtom["NE2"] = "CE1"
acceptorPlaneAtom["ND1"] = "NE2"
acceptorPlaneAtom["NE2"] = "ND1"

# HIS Donors: ND1, NE2
# Angle ND1-HD1 : 180
# Angle NE2-HE2 : 180
donorAtom["HD1"] = "ND1"
donorAtom["HE2"] = "NE2"

# TRP Donor: NE1-HE1
# Angle NE1-HE1 : 180
donorAtom["HE1"] = "NE1"

# LYS Donor NZ-HZX
# Angle NZ-HZX : 180
donorAtom["HZ1"] = "NZ"
donorAtom["HZ2"] = "NZ"
donorAtom["HZ3"] = "NZ"

# TYR acceptor OH
# Plane: CE1-CZ-OH
# Angle: CZ-OH 120
acceptorAngleAtom["OH"] = "CZ"
acceptorPlaneAtom["OH"] = "CE1"

# TYR donor: OH-HH
# Angle: OH-HH 180
donorAtom["HH"] = "OH"
acceptorPlaneAtom["OH"] = "CE1"

# SER acceptor:
# Angle CB-OG-X: 120
acceptorAngleAtom["OG"] = "CB"

# SER donor:
# Angle: OG-HG-X: 180
donorAtom["HG"] = "OG"

# THR acceptor:
# Angle: CB-OG1-X: 120
acceptorAngleAtom["OG1"] = "CB"

# THR donor:
# Angle: OG1-HG1-X: 180
donorAtom["HG1"] = "OG1"

kd_scale = {}
kd_scale["ILE"] = 4.5
kd_scale["VAL"] = 4.2
kd_scale["LEU"] = 3.8
kd_scale["PHE"] = 2.8
kd_scale["CYS"] = 2.5
kd_scale["MET"] = 1.9
kd_scale["ALA"] = 1.8
kd_scale["GLY"] = -0.4
kd_scale["THR"] = -0.7
kd_scale["SER"] = -0.8
kd_scale["TRP"] = -0.9
kd_scale["TYR"] = -1.3
kd_scale["PRO"] = -1.6
kd_scale["HIS"] = -3.2
kd_scale["GLU"] = -3.5
kd_scale["GLN"] = -3.5
kd_scale["ASP"] = -3.5
kd_scale["ASN"] = -3.5
kd_scale["LYS"] = -3.9
kd_scale["ARG"] = -4.5
masif_opts={}
masif_opts["use_hbond"] = True
masif_opts["use_hphob"] = True
masif_opts["use_apbs"] = True
masif_opts["compute_iface"] = True
# Mesh resolution. Everything gets very slow if it is lower than 1.0
masif_opts["mesh_res"] = 1.0
masif_opts["feature_interpolation"] = True

def computeAPBS(vertices, pdb_file, tmp_file_base):
    """
        Calls APBS, pdb2pqr, and multivalue and returns the charges per vertex
    """    
    print(pdb_file)
    #print(tmp_file_base)
    pdb2pqr = PDB2PQR_BIN +" --ff=PARSE --whitespace --noopt --apbs-input %s %s %s"
    make_pqr = pdb2pqr % (tmp_file_base+'.in', pdb_file, tmp_file_base+'.pqr') 
    os.system(make_pqr)
    
    apbs = APBS_BIN + " %s"
    make_apbs = apbs % (tmp_file_base+".in") 
    os.system(make_apbs)
    
    vertfile = open(tmp_file_base + ".csv", "w")
    for vert in vertices:
        vertfile.write("{},{},{}\n".format(vert[0], vert[1], vert[2]))
        #print(vert)
    vertfile.close()
    
    multivalue = MULTIVALUE_BIN + " %s %s %s"
    make_multivalue = multivalue % (tmp_file_base+".csv", tmp_file_base+".pqr.dx", tmp_file_base+"_out.csv") 
    print(tmp_file_base+".csv")
    print(tmp_file_base+".dx")
    os.system(make_multivalue)
    print('Created ouput using multivalue')
    # Read the charge file
    chargefile = open(tmp_file_base + "_out.csv")
    charges = np.array([0.0] * len(vertices))
    for ix, line in enumerate(chargefile.readlines()):
        charges[ix] = float(line.split(",")[3])
    print(tmp_file_base)
    os.system("rm " + tmp_file_base + "*")
    os.system("rm io.mc")
    print('Deleted temp files')
    return 

def computeCharges(pdb_filename, vertices, names):
    parser = PDBParser(QUIET=True)
    struct = parser.get_structure(pdb_filename, pdb_filename + ".pdb")
    residues = {}
    for res in struct.get_residues():
        chain_id = res.get_parent().get_id()
        if chain_id == "":
            chain_id = " "
        residues[(chain_id, res.get_id())] = res

    #atoms = Selection.unfold_entities(struct, "A")
    atoms = struct.get_atoms()
    satisfied_CO, satisfied_HN = computeSatisfied_CO_HN(atoms)

    charge = np.array([0.0] * len(vertices))
    # Go over every vertex
    for ix, name in enumerate(names):
        fields = name.split("_")
        chain_id = fields[0]
        if chain_id == "":
            chain_id = " "
        if fields[2] == "x":
            fields[2] = " "
        res_id = (" ", int(fields[1]), fields[2])
        aa = fields[3]
        atom_name = fields[4]
        # Ignore atom if it is BB and it is already satisfied.
        if atom_name == "H" and res_id in satisfied_HN:
            continue
        if atom_name == "O" and res_id in satisfied_CO:
            continue
        # Compute the charge of the vertex
        charge[ix] = computeChargeHelper(
            atom_name, residues[(chain_id, res_id)], vertices[ix]
        )

    return charge

# Compute the charge of a vertex in a residue.
def computeChargeHelper(atom_name, res, v):
    res_type = res.get_resname()
    # Check if it is a polar hydrogen.
    if isPolarHydrogen(atom_name, res):
        donor_atom_name = donorAtom[atom_name]
        a = res[donor_atom_name].get_coord()  # N/O
        b = res[atom_name].get_coord()  # H
        # Donor-H is always 180.0 degrees, = pi
        angle_deviation = computeAngleDeviation(a, b, v, np.pi)
        angle_penalty = computeAnglePenalty(angle_deviation)
        return 1.0 * angle_penalty
    # Check if it is an acceptor oxygen or nitrogen
    elif isAcceptorAtom(atom_name, res):
        acceptor_atom = res[atom_name]
        b = acceptor_atom.get_coord()
        a = res[acceptorAngleAtom[atom_name]].get_coord()
        # 120 degress for acceptor
        angle_deviation = computeAngleDeviation(a, b, v, 2 * np.pi / 3)
        # TODO: This should not be 120 for all atoms, i.e. for HIS it should be
        #       ~125.0
        angle_penalty = computeAnglePenalty(angle_deviation)
        plane_penalty = 1.0
        if atom_name in acceptorPlaneAtom:
            try:
                d = res[acceptorPlaneAtom[atom_name]].get_coord()
            except:
                return 0.0
            plane_deviation = computePlaneDeviation(d, a, b, v)
            plane_penalty = computeAnglePenalty(plane_deviation)
        return -1.0 * angle_penalty * plane_penalty
        # Compute the
    return 0.0


# Compute the absolute value of the deviation from theta
def computeAngleDeviation(a, b, c, theta):
    return abs(calc_angle(Vector(a), Vector(b), Vector(c)) - theta)


# Compute the angle deviation from a plane
def computePlaneDeviation(a, b, c, d):
    dih = calc_dihedral(Vector(a), Vector(b), Vector(c), Vector(d))
    dev1 = abs(dih)
    dev2 = np.pi - abs(dih)
    return min(dev1, dev2)


# angle_deviation from ideal value. TODO: do a more data-based solution
def computeAnglePenalty(angle_deviation):
    # Standard deviation: hbond_std_dev
    return max(0.0, 1.0 - (angle_deviation / (hbond_std_dev)) ** 2)


def isPolarHydrogen(atom_name, res):
    if atom_name in polarHydrogens[res.get_resname()]:
        return True
    else:
        return False


def isAcceptorAtom(atom_name, res):
    if atom_name.startswith("O"):
        return True
    else:
        if res.get_resname() == "HIS":
            if atom_name == "ND1" and "HD1" not in res:
                return True
            if atom_name == "NE2" and "HE2" not in res:
                return True
    return False


# Compute the list of backbone C=O:H-N that are satisfied. These will be ignored.
def computeSatisfied_CO_HN(atoms):
    ns = NeighborSearch(atoms)
    satisfied_CO = set()
    satisfied_HN = set()
    for atom1 in atoms:
        res1 = atom1.get_parent()
        if atom1.get_id() == "O":
            neigh_atoms = ns.search(atom1.get_coord(), 2.5, level="A")
            for atom2 in neigh_atoms:
                if atom2.get_id() == "H":
                    res2 = atom2.get_parent()
                    # Ensure they belong to different residues.
                    if res2.get_id() != res1.get_id():
                        # Compute the angle N-H:O, ideal value is 180 (but in
                        # helices it is typically 160) 180 +-30 = pi
                        angle_N_H_O_dev = computeAngleDeviation(
                            res2["N"].get_coord(),
                            atom2.get_coord(),
                            atom1.get_coord(),
                            np.pi,
                        )
                        # Compute angle H:O=C, ideal value is ~160 +- 20 = 8*pi/9
                        angle_H_O_C_dev = computeAngleDeviation(
                            atom2.get_coord(),
                            atom1.get_coord(),
                            res1["C"].get_coord(),
                            8 * np.pi / 9,
                        )
                        ## Allowed deviations: 30 degrees (pi/6) and 20 degrees
                        #       (pi/9)
                        if (
                            angle_N_H_O_dev - np.pi / 6 < 0
                            and angle_H_O_C_dev - np.pi / 9 < 0.0
                        ):
                            satisfied_CO.add(res1.get_id())
                            satisfied_HN.add(res2.get_id())
    return satisfied_CO, satisfied_HN


# Compute the charge of a new mesh, based on the charge of an old mesh.
# Use the top vertex in distance, for now (later this should be smoothed over 3
# or 4 vertices)
def assignChargesToNewMesh(new_vertices, old_vertices, old_charges, seeder_opts):
    dataset = old_vertices
    testset = new_vertices
    new_charges = np.zeros(len(new_vertices))
    if seeder_opts["feature_interpolation"]:
        num_inter = 4  # Number of interpolation features
        # Assign k old vertices to each new vertex.
        kdt = KDTree(dataset)
        dists, result = kdt.query(testset, k=num_inter)
        # Square the distances (as in the original pyflann)
        dists = np.square(dists)
        # The size of result is the same as new_vertices
        for vi_new in range(len(result)):
            vi_old = result[vi_new]
            dist_old = dists[vi_new]
            # If one vertex is right on top, ignore the rest.
            if dist_old[0] == 0.0:
                new_charges[vi_new] = old_charges[vi_old[0]]
                continue

            total_dist = np.sum(1 / dist_old)
            for i in range(num_inter):
                new_charges[vi_new] += (
                    old_charges[vi_old[i]] * (1 / dist_old[i]) / total_dist
                )
    else:
        # Assign k old vertices to each new vertex.
        kdt = KDTree(dataset)
        dists, result = kdt.query(testset)
        new_charges = old_charges[result]
    return new_charges

def compute_normal(vertex, face):
    
    """
    compute_normal - compute the normal of a triangulation
    vertex: 3xn matrix of vertices
    face: 3xm matrix of face indices.
    
      normal,normalf = compute_normal(vertex,face)
    
      normal(i,:) is the normal at vertex i.
      normalf(j,:) is the normal at face j.
    
    Copyright (c) 2004 Gabriel Peyr
    Converted to Python by Pablo Gainza LPDI EPFL 2017  
    """

    vertex = vertex.T
    face = face.T
    nface = np.size(face, 1)
    nvert = np.size(vertex, 1)
    normal = np.zeros((3, nvert))
    # unit normals to the faces
    normalf = crossp(
        vertex[:, face[1, :]] - vertex[:, face[0, :]],
        vertex[:, face[2, :]] - vertex[:, face[0, :]],
    )
    sum_squares = np.sum(normalf ** 2, 0)
    d = np.sqrt(sum_squares)
    d[d < eps] = 1
    normalf = normalf / repmat(d, 3, 1)
    # unit normal to the vertex
    normal = np.zeros((3, nvert))
    for i in np.arange(0, nface):
        f = face[:, i]
        for j in np.arange(3):
            normal[:, f[j]] = normal[:, f[j]] + normalf[:, i]

    # normalize
    d = np.sqrt(np.sum(normal ** 2, 0))
    d[d < eps] = 1
    normal = normal / repmat(d, 3, 1)
    # enforce that the normal are outward
    vertex_means = np.mean(vertex, 0)
    v = vertex - repmat(vertex_means, 3, 1)
    s = np.sum(np.multiply(v, normal), 1)
    if np.sum(s > 0) < np.sum(s < 0):
        # flip
        normal = -normal
        normalf = -normalf
    return normal.T

def crossp(x, y):
    # x and y are (m,3) dimensional
    z = np.zeros((x.shape))
    z[0, :] = np.multiply(x[1, :], y[2, :]) - np.multiply(x[2, :], y[1, :])
    z[1, :] = np.multiply(x[2, :], y[0, :]) - np.multiply(x[0, :], y[2, :])
    z[2, :] = np.multiply(x[0, :], y[1, :]) - np.multiply(x[1, :], y[0, :])
    return z

def computeHydrophobicity(names):
    hp = np.zeros(len(names))
    for ix, name in enumerate(names):
        aa = name.split("_")[3]
        hp[ix] = kd_scale[aa]
    return hp

def read_msms(file_root):
    # read the surface from the msms output. MSMS outputs two files: {file_root}.vert and {file_root}.face
    
    vertfile = open(file_root + ".vert")
    meshdata = (vertfile.read().rstrip()).split("\n")
    vertfile.close()

    # Read number of vertices.
    count = {}
    header = meshdata[2].split()
    count["vertices"] = int(header[0])
    ## Data Structures
    vertices = np.zeros((count["vertices"], 3))
    normalv = np.zeros((count["vertices"], 3))
    atom_id = [""] * count["vertices"]
    res_id = [""] * count["vertices"]
    for i in range(3, len(meshdata)):
        fields = meshdata[i].split()
        vi = i - 3
        vertices[vi][0] = float(fields[0])
        vertices[vi][1] = float(fields[1])
        vertices[vi][2] = float(fields[2])
        normalv[vi][0] = float(fields[3])
        normalv[vi][1] = float(fields[4])
        normalv[vi][2] = float(fields[5])
        atom_id[vi] = fields[7]
        res_id[vi] = fields[9]
        count["vertices"] -= 1

    # Read faces.
    facefile = open(file_root + ".face")
    meshdata = (facefile.read().rstrip()).split("\n")
    facefile.close()

    # Read number of vertices.
    header = meshdata[2].split()
    count["faces"] = int(header[0])
    faces = np.zeros((count["faces"], 3), dtype=int)
    normalf = np.zeros((count["faces"], 3))

    for i in range(3, len(meshdata)):
        fi = i - 3
        fields = meshdata[i].split()
        faces[fi][0] = int(fields[0]) - 1
        faces[fi][1] = int(fields[1]) - 1
        faces[fi][2] = int(fields[2]) - 1
        count["faces"] -= 1

    assert count["vertices"] == 0
    assert count["faces"] == 0

    return vertices, faces, normalv, res_id

def output_pdb_as_xyzrn(pdbfilename, xyzrnfilename):
    """
        pdbfilename: input pdb filename
        xyzrnfilename: output in xyzrn format.
    """
    parser = PDBParser()
    struct = parser.get_structure(pdbfilename, pdbfilename)
    outfile = open(xyzrnfilename, "w")
    for atom in struct.get_atoms():
        name = atom.get_name()
        residue = atom.get_parent()
        # Ignore hetatms.
        if residue.get_id()[0] != " ":
            continue
        resname = residue.get_resname()
        reskey = residue.get_id()[1]
        chain = residue.get_parent().get_id()
        atomtype = name[0]

        color = "Green"
        coords = None
        if atomtype in radii and resname in polarHydrogens:
            if atomtype == "O":
                color = "Red"
            if atomtype == "N":
                color = "Blue"
            if atomtype == "H":
                if name in polarHydrogens[resname]:
                    color = "Blue"  # Polar hydrogens
            coords = "{:.06f} {:.06f} {:.06f}".format(
                atom.get_coord()[0], atom.get_coord()[1], atom.get_coord()[2]
            )
            insertion = "x"
            if residue.get_id()[2] != " ":
                insertion = residue.get_id()[2]
            full_id = "{}_{:d}_{}_{}_{}_{}".format(
                chain, residue.get_id()[1], insertion, resname, name, color
            )
        if coords is not None:
            outfile.write(coords + " " + radii[atomtype] + " 1 " + full_id + "\n")

def computeMSMS(pdb_file,  protonate=True, one_cavity=None):
    randnum = random.randint(1,10000000)
    file_base = tempdir
    out_xyzrn = file_base+".xyzrn"

    if protonate:        
        output_pdb_as_xyzrn(pdb_file, out_xyzrn)
    else:
        print("Error - pdb2xyzrn is deprecated.")
        sys.exit(1)
    # Now run MSMS on xyzrn file
    FNULL = open(os.devnull, 'w')
    if one_cavity is not None:
        args = [msms_bin, "-density", "3.0", "-hdensity", "3.0", "-probe", "1.5",\
                "-one_cavity", str(1), str(one_cavity),\
                "-if",out_xyzrn,"-of",file_base, "-af", file_base]
    else:
        args = [msms_bin, "-density", "3.0", "-hdensity", "3.0", "-probe",\
                    "1.5", "-all_components", "-if",out_xyzrn,"-of",file_base, "-af", file_base]
    #print msms_bin+" "+`args`
    p2 = Popen(args, stdout=PIPE, stderr=PIPE)
    stdout, stderr = p2.communicate()

    vertices, faces, normals, names = read_msms(file_base)
    areas = {}
    ses_file = open(file_base+".area")
    next(ses_file) # ignore header line
    for line in ses_file:
        fields = line.split()
        areas[fields[3]] = fields[1]
    print('FILEBASE:', file_base)
    os.system("rm " + file_base + "*")
    return vertices, faces, normals, names, areas

def fix_mesh(mesh, resolution, detail="normal"):
    bbox_min, bbox_max = mesh.bbox;
    diag_len = norm(bbox_max - bbox_min);
    if detail == "normal":
        target_len = diag_len * 5e-3;
    elif detail == "high":
        target_len = diag_len * 2.5e-3;
    elif detail == "low":
        target_len = diag_len * 1e-2;
    
    target_len = resolution
    #print("Target resolution: {} mm".format(target_len));
    # PGC 2017: Remove duplicated vertices first
    mesh, _ = pymesh.remove_duplicated_vertices(mesh, 0.001)


    count = 0;
    print("Removing degenerated triangles")
    mesh, __ = pymesh.remove_degenerated_triangles(mesh, 100);
    mesh, __ = pymesh.split_long_edges(mesh, target_len);
    num_vertices = mesh.num_vertices;
    while True:
        mesh, __ = pymesh.collapse_short_edges(mesh, 1e-6);
        mesh, __ = pymesh.collapse_short_edges(mesh, target_len,
                preserve_feature=True);
        mesh, __ = pymesh.remove_obtuse_triangles(mesh, 150.0, 100);
        if mesh.num_vertices == num_vertices:
            break;

        num_vertices = mesh.num_vertices;
        #print("#v: {}".format(num_vertices));
        count += 1;
        if count > 10: break;

    mesh = pymesh.resolve_self_intersection(mesh);
    mesh, __ = pymesh.remove_duplicated_faces(mesh);
    #mesh = pymesh.compute_outer_hull(mesh);

    ############ Added by Oscar Mendez Lucio ##############
    mesh = pymesh.compute_outer_hull(mesh, all_layers=True);
    num_nodes = [i.num_nodes for i in mesh]
    mesh = mesh[np.argmax(num_nodes)]
    ############################################################

    mesh, __ = pymesh.remove_duplicated_faces(mesh);
    mesh, __ = pymesh.remove_obtuse_triangles(mesh, 179.0, 5);
    mesh, __ = pymesh.remove_isolated_vertices(mesh);
    mesh, _ = pymesh.remove_duplicated_vertices(mesh, 0.001)

    ############ Added by Oscar Mendez Lucio ##############
    mesh = pymesh.separate_mesh(mesh)
    num_nodes = [i.num_nodes for i in mesh]
    mesh = mesh[np.argmax(num_nodes)]
    ############################################################

    return mesh

def save_ply(
    filename,
    vertices,
    faces=[],
    normals=None,
    charges=None,
    vertex_cb=None,
    hbond=None,
    hphob=None,
    iface=None,
    si=None,
    normalize_charges=False,
):
    """ Save vertices, mesh in ply format.
        vertices: coordinates of vertices
        faces: mesh
    """
    mesh = pymesh.form_mesh(vertices, faces)
    if normals is not None:
        n1 = normals[:, 0]
        n2 = normals[:, 1]
        n3 = normals[:, 2]
        mesh.add_attribute("vertex_nx")
        mesh.set_attribute("vertex_nx", n1)
        mesh.add_attribute("vertex_ny")
        mesh.set_attribute("vertex_ny", n2)
        mesh.add_attribute("vertex_nz")
        mesh.set_attribute("vertex_nz", n3)
    if charges is not None:
        mesh.add_attribute("charge")
        if normalize_charges:
            charges = charges / 10
        mesh.set_attribute("charge", charges)
    if hbond is not None:
        mesh.add_attribute("hbond")
        mesh.set_attribute("hbond", hbond)
    if vertex_cb is not None:
        mesh.add_attribute("vertex_cb")
        mesh.set_attribute("vertex_cb", vertex_cb)
    if hphob is not None:
        mesh.add_attribute("vertex_hphob")
        mesh.set_attribute("vertex_hphob", hphob)
    if iface is not None:
        mesh.add_attribute("vertex_iface")
        mesh.set_attribute("vertex_iface", iface)
    if si is not None:
        mesh.add_attribute("vertex_si")
        mesh.set_attribute("vertex_si", si)

    pymesh.save_mesh(
        filename, mesh, *mesh.get_attribute_names(), use_float=True, ascii=True
    )

# Functions for processing ligands
def oneHotVector(val, lst):
    '''Converts a value to a one-hot vector based on options in lst'''
    if val not in lst:
        val = lst[-1]
    return map(lambda x: x == val, lst)

def mol_to_nx(mol):
    G = nx. Graph()
        
    # Globals.
    G.graph["features"] = np.array([None], dtype = np.float32)
    atomCoords = mol.GetConformer().GetPositions()

    for i, atom in enumerate(mol.GetAtoms()):
        if atom.GetAtomicNum() == 1: continue
        G.add_node(atom.GetIdx(),
                   pos = atomCoords[i],
                   x=np.array(list(oneHotVector(atom.GetAtomicNum(),
                                                [4, 5, 6, 7, 8, 9, 12, 14, 15, 16, 17, 23, 26, 27, 29, 30, 33, 34, 35, 44, 45, 51, 53, 75, 76, 77, 78, 80])), 
                                     dtype = np.float32))
        
    for bond in mol.GetBonds():
        if mol.GetAtomWithIdx(bond.GetBeginAtomIdx()).GetAtomicNum() == 1: continue
        if mol.GetAtomWithIdx(bond.GetEndAtomIdx()).GetAtomicNum() == 1: continue
        isConjugated = 1 if bond.GetIsConjugated() and not bond.GetIsAromatic() else 0
        G.add_edge(bond.GetBeginAtomIdx(),
                   bond.GetEndAtomIdx(),
                   edge_attr=np.array(list(oneHotVector(bond.GetBondTypeAsDouble(), 
                                                  [1.0, 1.5, 2.0, 3.0, 99.0]))+[isConjugated], 
                                     dtype = np.float32))
    return G


def get_bonds(mol_list, bidirectional=True):
    atom_counter=0
    bonds = []
    dist = []
    for m in mol_list:
        for x in m.GetBonds():
            conf = m.GetConformer()
            bonds.extend([(x.GetBeginAtomIdx()+atom_counter, x.GetEndAtomIdx()+atom_counter)])
            dist.extend([rdMolTransforms.GetBondLength(conf,x.GetBeginAtomIdx(), x.GetEndAtomIdx())])
            if bidirectional:
                bonds.extend([(x.GetEndAtomIdx()+atom_counter, x.GetBeginAtomIdx()+atom_counter)])
                dist.extend([rdMolTransforms.GetBondLength(conf,x.GetEndAtomIdx(), x.GetBeginAtomIdx())])
        atom_counter += m.GetNumAtoms()
    return bonds, dist


def get_angles(mol_list, bidirectional=True):
    atom_counter = 0
    bendList = []
    angleList = []
    for m in mol_list:
        bendSmarts = '*~*~*'
        bendQuery = Chem.MolFromSmarts(bendSmarts)
        matches = m.GetSubstructMatches(bendQuery)
        conf = m.GetConformer()
        for match in matches:
            idx0 = match[0]
            idx1 = match[1]
            idx2 = match[2]
            bendList.append((idx0+atom_counter, idx1+atom_counter, idx2+atom_counter))
            angleList.append(rdMolTransforms.GetAngleRad(conf, idx0, idx1, idx2))
            if bidirectional:
                bendList.append((idx2+atom_counter, idx1+atom_counter, idx0+atom_counter))
                angleList.append(rdMolTransforms.GetAngleRad(conf, idx2, idx1, idx0))
        atom_counter += m.GetNumAtoms()
    return bendList, angleList


def get_torsions(mol_list, bidirectional=True):
    atom_counter=0
    torsionList = []
    dihedralList = []
    for m in mol_list:
        torsionSmarts = '[!$(*#*)&!D1]~[!$(*#*)&!D1]'
        torsionQuery = Chem.MolFromSmarts(torsionSmarts)
        matches = m.GetSubstructMatches(torsionQuery)
        conf = m.GetConformer()
        for match in matches:
            idx2 = match[0]
            idx3 = match[1]
            bond = m.GetBondBetweenAtoms(idx2, idx3)
            jAtom = m.GetAtomWithIdx(idx2)
            kAtom = m.GetAtomWithIdx(idx3)
            if (((jAtom.GetHybridization() != Chem.HybridizationType.SP2)
                and (jAtom.GetHybridization() != Chem.HybridizationType.SP3))
                or ((kAtom.GetHybridization() != Chem.HybridizationType.SP2)
                and (kAtom.GetHybridization() != Chem.HybridizationType.SP3))):
                continue
            for b1 in jAtom.GetBonds():
                if (b1.GetIdx() == bond.GetIdx()):
                    continue
                idx1 = b1.GetOtherAtomIdx(idx2)
                for b2 in kAtom.GetBonds():
                    if ((b2.GetIdx() == bond.GetIdx())
                        or (b2.GetIdx() == b1.GetIdx())):
                        continue
                    idx4 = b2.GetOtherAtomIdx(idx3)
                    # skip 3-membered rings
                    if (idx4 == idx1):
                        continue
                    torsionList.append((idx1+atom_counter, idx2+atom_counter, idx3+atom_counter, idx4+atom_counter))
                    dihedralList.append(rdMolTransforms.GetDihedralRad(conf, idx1, idx2, idx3, idx4))
                    if bidirectional:
                        torsionList.append((idx4+atom_counter, idx3+atom_counter, idx2+atom_counter, idx1+atom_counter))
                        dihedralList.append(rdMolTransforms.GetDihedralRad(conf, idx4, idx3, idx2, idx1))
        atom_counter += m.GetNumAtoms()
    return torsionList, dihedralList


def mol_with_atom_index( mol ):
    atoms = mol.GetNumAtoms()
    for idx in range( atoms ):
        mol.GetAtomWithIdx( idx ).SetProp( 'molAtomMapNumber', str( mol.GetAtomWithIdx( idx ).GetIdx() ) )
    return mol

def atomenvironments(mol, radius=3):
    envs = []
    for a in mol.GetAtoms():
        idx = a.GetIdx()
        env = Chem.FindAtomEnvironmentOfRadiusN(mol, radius, idx)
        amap = {}
        submol=Chem.PathToSubmol(mol, env, atomMap=amap)
        if amap.get(idx) is not None:
            envs.append(Chem.MolToSmarts(submol))
    return envs

def compute_inp_surface(target_filename, ligand_filename, dist_threshold=10):
    sufix = '_'+str(dist_threshold+5)+'A.pdb'
    out_filename = os.path.splitext(target_filename)[0]
    new_infile= '/home/dell4/king/202112_graphDrug/mark3/'+target_filename.split('/')[-1].split('.')[0]+sufix
    #print('new_infile:'+new_infile)
   
    
    # Get atom coordinates
    mol = Chem.MolFromMol2File(ligand_filename, sanitize=False, cleanupSubstructures=False)
    g = mol_to_nx(mol)
    atomCoords = np.array([g.nodes[i]['pos'].tolist() for i in g.nodes])
    
    # Read protein and select aminino acids in the binding pocket
    parser = Bio.PDB.PDBParser(QUIET=True) # QUIET=True avoids comments on errors in the pdb.

    structures = parser.get_structure('target', target_filename)
    structure = structures[0] # 'structures' may contain several proteins in this case only one.

    atoms  = Bio.PDB.Selection.unfold_entities(structure, 'A')
    ns = Bio.PDB.NeighborSearch(atoms)
    
    close_residues= []
    for a in atomCoords:  
        close_residues.extend(ns.search(a, dist_threshold+5, level='R'))
    close_residues = Bio.PDB.Selection.uniqueify(close_residues)

    class SelectNeighbors(Select):
        def accept_residue(self, residue):
            if residue in close_residues:
                if all(a in [i.get_name() for i in residue.get_unpacked_list()] for a in ['N', 'CA', 'C', 'O']) or residue.resname=='HOH':
                    return True
                else:
                    return False
            else:
                return False

    pdbio = PDBIO()
    pdbio.set_structure(structure)
    pdbio.save(new_infile, SelectNeighbors())
    
    # Identify closes atom to the ligand
    structures = parser.get_structure('target', new_infile)
    structure = structures[0] # 'structures' may contain several proteins in this case only one.
    #print(len(structures))
    atoms  = Bio.PDB.Selection.unfold_entities(structure, 'A')

    #dist = [distance.euclidean(atomCoords.mean(axis=0), a.get_coord()) for a in atoms]
    #atom_idx = np.argmin(dist)
    #dist = [[distance.euclidean(ac, a.get_coord()) for ac in atomCoords] for a in atoms]
    #atom_idx = np.argsort(np.min(dist, axis=1))[0]
    
    # Compute MSMS of surface w/hydrogens, 
    try:
        dist = [distance.euclidean(atomCoords.mean(axis=0), a.get_coord()) for a in atoms]
        atom_idx = np.argmin(dist)
        vertices1, faces1, normals1, names1, areas1 = computeMSMS(new_infile,\
                                                                  protonate=True, one_cavity=atom_idx)
        
        # Find the distance between every vertex in binding site surface and each atom in the ligand.
        kdt = KDTree(atomCoords)
        d, r = kdt.query(vertices1)
        assert(len(d) == len(vertices1))
        iface_v = np.where(d <= dist_threshold)[0]
        faces_to_keep = [idx for idx, face in enumerate(faces1) if all(v in iface_v  for v in face)] 
    
        # Compute "charged" vertices
        if masif_opts['use_hbond']:
            vertex_hbond = computeCharges(out_filename, vertices1, names1)    
    
        # For each surface residue, assign the hydrophobicity of its amino acid. 
        if masif_opts['use_hphob']:
            vertex_hphobicity = computeHydrophobicity(names1)    
        
        # If protonate = false, recompute MSMS of surface, but without hydrogens (set radius of hydrogens to 0).
        vertices2 = vertices1
        faces2 = faces1
    
        # Fix the mesh.
        mesh = pymesh.form_mesh(vertices2, faces2)
        mesh = pymesh.submesh(mesh, faces_to_keep, 0)
        with io.capture_output() as captured:
            regular_mesh = fix_mesh(mesh, masif_opts['mesh_res'])
        
    except:
        try:
            dist = [[distance.euclidean(ac, a.get_coord()) for ac in atomCoords] for a in atoms]
            atom_idx = np.argsort(np.min(dist, axis=1))[0]
            vertices1, faces1, normals1, names1, areas1 = computeMSMS(new_infile,\
                                                                      protonate=True, one_cavity=atom_idx)

            # Find the distance between every vertex in binding site surface and each atom in the ligand.
            kdt = KDTree(atomCoords)
            d, r = kdt.query(vertices1)
            assert(len(d) == len(vertices1))
            iface_v = np.where(d <= dist_threshold)[0]
            faces_to_keep = [idx for idx, face in enumerate(faces1) if all(v in iface_v  for v in face)] 
    
            # Compute "charged" vertices
            if masif_opts['use_hbond']:
                vertex_hbond = computeCharges(out_filename, vertices1, names1)    
    
            # For each surface residue, assign the hydrophobicity of its amino acid. 
            if masif_opts['use_hphob']:
                vertex_hphobicity = computeHydrophobicity(names1)    
        
            # If protonate = false, recompute MSMS of surface, but without hydrogens (set radius of hydrogens to 0).
            vertices2 = vertices1
            faces2 = faces1
    
            # Fix the mesh.
            mesh = pymesh.form_mesh(vertices2, faces2)
            mesh = pymesh.submesh(mesh, faces_to_keep, 0)
            with io.capture_output() as captured:
                regular_mesh = fix_mesh(mesh, masif_opts['mesh_res'])
                
        except:
            vertices1, faces1, normals1, names1, areas1 = computeMSMS(new_infile,\
                                                                      protonate=True, one_cavity=None)

            # Find the distance between every vertex in binding site surface and each atom in the ligand.
            kdt = KDTree(atomCoords)
            d, r = kdt.query(vertices1)
            assert(len(d) == len(vertices1))
            iface_v = np.where(d <= dist_threshold)[0]
            faces_to_keep = [idx for idx, face in enumerate(faces1) if all(v in iface_v  for v in face)] 
    
            # Compute "charged" vertices
            if masif_opts['use_hbond']:
                vertex_hbond = computeCharges(out_filename, vertices1, names1)    
    
            # For each surface residue, assign the hydrophobicity of its amino acid. 
            if masif_opts['use_hphob']:
                vertex_hphobicity = computeHydrophobicity(names1)    
        
            # If protonate = false, recompute MSMS of surface, but without hydrogens (set radius of hydrogens to 0).
            vertices2 = vertices1
            faces2 = faces1
    
            # Fix the mesh.
            mesh = pymesh.form_mesh(vertices2, faces2)
            mesh = pymesh.submesh(mesh, faces_to_keep, 0)
            with io.capture_output() as captured:
                regular_mesh = fix_mesh(mesh, masif_opts['mesh_res'])
        
    # Compute the normals
    vertex_normal = compute_normal(regular_mesh.vertices, regular_mesh.faces)
    # Assign charges on new vertices based on charges of old vertices (nearest
    # neighbor)
    
    if masif_opts['use_hbond']:
        vertex_hbond = assignChargesToNewMesh(regular_mesh.vertices, vertices1,\
            vertex_hbond, masif_opts)
    
    if masif_opts['use_hphob']:
        vertex_hphobicity = assignChargesToNewMesh(regular_mesh.vertices, vertices1,\
            vertex_hphobicity, masif_opts)
    
    if masif_opts['use_apbs']:
        vertex_charges = computeAPBS(regular_mesh.vertices, new_infile, new_infile.split('.p')[0]+"_temp")
        
    # Compute the principal curvature components for the shape index. 
    regular_mesh.add_attribute("vertex_mean_curvature")
    H = regular_mesh.get_attribute("vertex_mean_curvature")
    regular_mesh.add_attribute("vertex_gaussian_curvature")
    K = regular_mesh.get_attribute("vertex_gaussian_curvature")
    elem = np.square(H) - K
    # In some cases this equation is less than zero, likely due to the method that computes the mean and gaussian curvature.
    # set to an epsilon.
    elem[elem<0] = 1e-8
    k1 = H + np.sqrt(elem)
    k2 = H - np.sqrt(elem)
    # Compute the shape index 
    si = (k1+k2)/(k1-k2)
    si = np.arctan(si)*(2/np.pi)
    
    # Convert to ply and save.
    save_ply(out_filename+".ply", regular_mesh.vertices,\
             regular_mesh.faces, normals=vertex_normal, charges=vertex_charges,\
             normalize_charges=True, hbond=vertex_hbond, hphob=vertex_hphobicity,\
             si=si)