#!/usr/bin/python
import os
import shutil
import sys

file_path = sys.argv[1]

directory = sys.argv[2]

for file in os.listdir(file_path):
    if "be0" in file:
        cg = file[:-9]
        ori = cg + "001.cg"
        shutil.copy("/scr/risa/mgeyer/ernwinenv/data/" + directory + ori, file_path + ori)
