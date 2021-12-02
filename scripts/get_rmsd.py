#!/usr/bin/env python3

#python3 scripts/get_rmsd.py play_set/train/ play_set/pdb/ play_set/RMSD_list.txt

import subprocess
from os import listdir
from os.path import isfile, join
import sys

file_path = sys.argv[1]
pdb_fp = sys.argv[2]

cg_files = [f for f in listdir(file_path) if isfile(join(file_path, f))]

pdb_files = [f for f in listdir(pdb_fp) if isfile(join(pdb_fp, f))]

l = []

for cg in cg_files:
    name = cg[:-9]
    pdb = name + "001.cg"#".pdb"
    line = ""
    if pdb in pdb_files:
        cg_p = file_path + cg
        pdb_p = pdb_fp + pdb
        try:
            r = subprocess.check_output(["compare_RNA.py", cg_p, pdb_p, "--rmsd"])
        except:
            print("Error with files", pdb, cg)
            continue
        rmsd = str(r.rstrip())[9:-1]
        line = cg + "\t" + rmsd + "\n"
        l.append(line)
    else:
        print("PDB file not found:", pdb)

with open(sys.argv[3], "w") as fh:
    for elem in l:
        fh.write(elem)
