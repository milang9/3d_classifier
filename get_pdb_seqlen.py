#!/bin/python
from Bio.PDB import PDBParser
from Bio import PDB
from os import listdir
from os.path import isfile, join
from shutil import copyfile

fs_path = '/scr/risa/mgeyer/ernwinenv/data/5srRNA_PDBs/'

t_path = '/scr/risa/mgeyer/ernwinenv/data/tRNA_PDBs/'

files_5s = [f for f in listdir(fs_path) if isfile(join(fs_path, f))]

files_t = [f for f in listdir(t_path) if isfile(join(t_path, f))]

def get_seq_len(files, path):
    new_dir = '/scr/risa/mgeyer/ernwinenv/data/test_set_5st/'
    c=0
    for pdb in files:
        pdb_path = path + pdb
        struc = PDBParser().get_structure("RNA", pdb_path)
        for chain in struc.get_chains():
            chain_len = len([_ for _ in chain.get_residues() if not PDB.is_aa(_)])
            if chain_len <= 120:

                new = new_dir + pdb
                print(new)
                copyfile(pdb_path, new)
                c+=1
                if c>=50:
                    return

get_seq_len(files_t, t_path)
get_seq_len(files_5s, fs_path)
