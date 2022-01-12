#!/bin/python
from Bio.PDB import PDBParser
from Bio import PDB
from os import listdir
from os.path import isfile, join
import sys
import os

p = sys.argv[1] #'/scr/risa/mgeyer/ernwinenv/data/tset_5st_new/'

# paths and files for comparison
#u_path = '/scr/risa/mgeyer/ernwinenv/data/test_set_5st/'

#m_path = '/scr/risa/mgeyer/ernwinenv/data/missing_pdbs/'

#used_pdbs = [f for f in listdir(u_path) if isfile(join(u_path, f))]

#m_pdbs = [f for f in listdir(m_path) if isfile(join(m_path, f))]

#print(used_pdbs)

# loop to print file and seq length
c = 0
d = 0
l = []
m = []
for file in listdir(p):
    #if file not in used_pdbs and file not in m_pdbs:
    struc = p + file
    s = PDBParser().get_structure("RNA", struc)
    print(file)
    for chain in s.get_chains():
        chain_len = len([_ for _ in chain.get_residues() if not PDB.is_aa(_)])
        print(chain_len)
        if chain_len >= 130:
            c += 1
            l.append(file)
        if chain_len <= 40:
            d += 1
            m.append(file)
            os.remove(struc)
        if chain_len >= 140:
            os.remove(struc)
print("seqs greater 130", c)
print(l)
print("seqs shorter 40", d)
print(m)
