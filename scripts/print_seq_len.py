#!/bin/python
from Bio.PDB import PDBParser
from Bio import PDB
from os import listdir
from os.path import isfile, join

p = '/scr/risa/mgeyer/ernwinenv/data/tRNA_PDBs/' #'/scr/risa/mgeyer/ernwinenv/data/tset_5st_new/'

# paths and files for comparison
u_path = '/scr/risa/mgeyer/ernwinenv/data/test_set_5st/'

m_path = '/scr/risa/mgeyer/ernwinenv/data/missing_pdbs/'

used_pdbs = [f for f in listdir(u_path) if isfile(join(u_path, f))]

m_pdbs = [f for f in listdir(m_path) if isfile(join(m_path, f))]

#print(used_pdbs)

# loop to print file and seq length
for file in listdir(p):
    if file not in used_pdbs and file not in m_pdbs:
        struc = p + file
        s = PDBParser().get_structure("RNA", struc)
        print(file)
        for chain in s.get_chains():
            chain_len = len([_ for _ in chain.get_residues() if not PDB.is_aa(_)])
            print(chain_len)
