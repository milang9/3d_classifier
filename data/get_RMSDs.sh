#!/bin/bash

for file in path/to/files; do
  base=$(basename $file .coord)
  compare_RNA.py --rmsd data/.. $file

done
