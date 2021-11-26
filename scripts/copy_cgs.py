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

total = 100
train = 60
val = 20
test = 20

number = 1

train_list = []
tc_fs = 0
tc_tr = 0
val_list = []
vc_fs = 0
vc_tr = 0
test_list = []
tec_fs = 0
tec_tr = 0
for rna in os.listdir(origin):
    if "5srRNA" in rna:
        id = "5srRNA"
    elif "tRNA" in rna:
        id = "tRNA"
    best_r0 = rna + "/best_rmsd0.coord"
    best0 = rna + "/simulation_01/best0.coord"
    data_list = [best_r0, best0]
    rnd_list = []
    for i in range(number):
        rnd = str(random.randrange(1, 100))
        while len(rnd) < 4:
            rnd = "0" + rnd
        data_list.append(rna + "/simulation_01/step" + rnd + "00.coord")
        rnd_list.append(rna + "/simulation_01/step" + rnd + "00.coord")
    if len(train_list) < train*(2+number) and tc_fs < train/2 and id == "5srRNA":
        train_list += data_list
        tc_fs +=1
        set_dir = "training_set/"
    elif len(train_list) < train*(2+number) and tc_tr < train/2 and id == "tRNA":
        train_list += data_list
        tc_tr +=1
        set_dir = "training_set/"
    elif len(val_list) < val*(2+number) and vc_fs < val/2 and id == "5srRNA":
        val_list += data_list
        vc_fs +=1
        set_dir = "val_set/"
    elif len(val_list) < val*(2+number) and vc_tr < val/2 and id == "tRNA":
        val_list += data_list
        vc_tr +=1
        set_dir = "val_set/"
    else:
        test_list += data_list
        set_dir = "test_set/"
        if id == "5srRNA":
            tec_fs +=1
        if id == "tRNA":
            tec_tr +=1

    #copy files
    '''
    shutil.copy(origin + best_r0, destination + set_dir + rna + "_br0.cg")
    shutil.copy(origin + best0, destination + set_dir + rna + "_be0.cg")
    i = 1
    for x in rnd_list:
        shutil.copy(origin + x, destination + set_dir + rna + "_rn" + str(i) + ".cg")
        i+=1
    '''
print("count training set, 5srRNA:", tc_fs)
print("count training set, tRNA:", tc_tr)
print("count validation set, 5srRNA:", vc_fs)
print("count validation set, tRNA:", vc_tr)
print("count test set, 5srRNA:", tec_fs)
print("count test set, tRNA:", tec_tr)
print("length training set:", len(train_list))
print("length validation set:", len(val_list))
print("length test set:", len(test_list))
