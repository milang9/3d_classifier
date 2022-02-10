#!/usr/bin/env python3

#python3 scripts/get_rmsd.py play_set/train/ play_set/pdb/ play_set/RMSD_list.txt

import subprocess
from os import listdir
from os.path import isfile, join
import sys

file_path = sys.argv[1] #directory for selected ernwin simulated structures
pdb_fp = sys.argv[2] #directory original pdbs

#argv[3]: output rmsd list files
#argv[4]: error file, if a rmsd cant be calculated the structure name is recorded here

cg_files = [f for f in listdir(file_path) if isfile(join(file_path, f))]

pdb_files = [f for f in listdir(pdb_fp) if isfile(join(pdb_fp, f))]

l = []

error_f = []

for cg in cg_files:
    if len(cg) > 14:
        name = cg[:-9]
    else:
        name = cg[:-3]
    pdb = name + ".cg"
    #if "001.cg" in cg:
    #    pdb = cg
    #else:
    #    pdb = name + "001.cg"#".pdb"
    line = ""
    if pdb in pdb_files:
        cg_p = file_path + cg
        pdb_p = pdb_fp + pdb
        try:
            r = subprocess.check_output(["compare_RNA.py", cg_p, pdb_p, "--rmsd"])
        except:
            print("Error with files", pdb, cg)
            error_f.append(f"{pdb} {cg}\n")
            continue
        rmsd = str(r.rstrip())[9:-1]
        line = cg + "\t" + rmsd + "\n"
        l.append(line)
    else:
        print("PDB file not found:", pdb)

with open(sys.argv[3], "w") as fh:
    for elem in l:
        fh.write(elem)

with open(sys.argv[4], "w") as fj:
    for line in error_f:
        fj.write(line)
