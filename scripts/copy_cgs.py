#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
move .coord files from the ernwin output dir to their respective dirs (train, val, test) and rename them
"""

import os
import sys
import random
import shutil

origin = sys.argv[1] #ernwin output directory

destination = sys.argv[2]

set_dir = sys.argv[3]

number = 8

for rna in os.listdir(origin):

    best_r0 = rna + "/best_rmsd0.coord"
    best0 = rna + "/simulation_01/best0.coord"
    data_list = [best_r0, best0]
    rng = []
    rnd_list = []

    while len(rng) < number:
        rnd = str(random.randrange(1, 100))
        while len(rnd) < 4:
            rnd = "0" + rnd
        if rnd not in rng:
            rng.append(rnd)

    for i in rng:
        rnd_list.append(rna + "/simulation_01/step" + i + "00.coord")



    #copy files

    shutil.copy(origin + best_r0, destination + set_dir + rna + "_br0.cg")
    shutil.copy(origin + best0, destination + set_dir + rna + "_be0.cg")
    i = 1
    for x in rnd_list:
        shutil.copy(origin + x, destination + set_dir + rna + "_rn" + str(i) + ".cg")
        i+=1
