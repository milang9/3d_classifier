#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
move .coord files from the ernwin output dir to their respective dirs (train, val, test) and rename them
"""

import os
import sys
import random
import shutil
import glob

origin = sys.argv[1] #ernwin output directory

destination = sys.argv[2]

set_dir = sys.argv[3]

number = 10

for rna in os.listdir(origin):
    #get the best rmsd structures
    for a in range(3):
        best_r = rna + "/best_rmsd" + str(a) + ".coord"
        shutil.copy(origin + best_r, destination + set_dir + rna + "_br" + str(a) + ".cg")

    rnd_list = []
    for sim in range(1, 5):

        #get the best structures of each sim
        for b in range(3):
            best = rna + "/simulation_0" + str(sim) + "/" + "best" + str(b) + ".coord"
            shutil.copy(origin + best, destination + set_dir + rna + "_be" + str(sim) + str(b) + ".cg")

        #get random structures
        rng = []
        while len(rng) < number:
            rnd = str(random.randrange(1, 100))
            while len(rnd) < 4:
                rnd = "0" + rnd
            if rnd not in rng:
                rng.append(rnd)

        for i in rng:
            rnd_list.append(rna + "/simulation_0" + str(sim) + "/" + "step" + i + "00.coord")



    #copy files
    i = 1
    for x in rnd_list:
        if len(str(i)) == 1:
            j = "0" + str(i)
        else:
            j = str(i)
        shutil.copy(origin + x, destination + set_dir + rna + "_rn" + j + ".cg")
        i+=1
