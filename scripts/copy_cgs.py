#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
move .coord files from the ernwin output dir to another dir and rename them
"""

import os
import sys
import random
import shutil
import glob

origin = sys.argv[1] #ernwin output directory

destination = sys.argv[2]

number = int(sys.argv[3])

for rna in os.scandir(origin):
    if not os.path.isdir(rna.path):
        print(rna.path, os.path.isdir(rna))
        continue
    else:
        print(rna.path)
    #get the best rmsd structures
    for a in range(3):
        best_r = rna.path + "/best_rmsd" + str(a) + ".coord"
        shutil.copy(best_r, destination + rna.name + "_br" + str(a) + ".cg")

    rnd_list = []
    for sim in range(1, 10):
        if len(str(sim)) < 2:
            sim = "0" + str(sim)
        else:
            sim = str(sim)

        #get the best structures of each sim
        for b in range(3):
            best = rna.path + "/simulation_" + sim + "/" + "best" + str(b) + ".coord"
            shutil.copy(best, destination + rna.name + "_be" + sim + str(b) + ".cg")

        #get random structures
        rng = []
        while len(rng) < number:
            rnd = str(random.randrange(1, 1000))
            while len(rnd) < 4:
                rnd = "0" + rnd
            if rnd not in rng:
                rng.append(rnd)

        for i in rng:
            rnd_list.append(rna.path + "/simulation_" + sim + "/" + "step" + i + "00.coord")



    #copy files
    i = 1
    for x in rnd_list:
        if len(str(i)) == 1:
            j = "0" + str(i)
        else:
            j = str(i)
        shutil.copy(x, destination + rna.name + "_rn" + j + ".cg")
        i+=1
