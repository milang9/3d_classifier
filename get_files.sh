#!/bin/bash
##./get_files.sh representative_components.txt

for file in $(awk -F ' ' '{ print $1}' $1); do #get first element of each line of file $1
  echo $file

  if [ ! -f ${file}.cif ]; then
     wget http://files.rcsb.org/download/${file}.cif.gz
     ##wget http://files.rcsb.org/download/${file}.pdb.gz
  fi

done

#gunzip *.gz
