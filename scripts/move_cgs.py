#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
move .coord files from the ernwin output dir to their respective dirs (train, val, test) and rename them
"""

import os
import sys

path = sys.argv[1] #ernwin output directory

for rna in os.listdir(path):
    print(rna)