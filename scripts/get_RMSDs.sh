#!/bin/bash
#bash scripts/get_RMSDs.sh play_set/train/ play_set/pdb/ 

for file in ${1}*; do
    base=$(basename ${file} .cg)
    path=${2}${base^^}.pdb
    #if [ -f "$path" ]; then
    #    echo $path
    #fi
    #echo $file
    line=$(echo |compare_RNA.py ./${file} ./${path} --rmsd|cut -d' ' -f 2)
    echo $line    
    rmsd=$(cut -d' ' -f 2 ${line})
    echo $rmsd
done < $1
