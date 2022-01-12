#!/bin/bash

while IFS=$'\t' read -r -a tmp; do

  rfam=$2
  id=${tmp[0]}
  pdb=${tmp[1]}
  sec=${tmp[2]}
  if [ $id == $rfam ]; then
    echo $id $pdb $sec
    wget https://files.rcsb.org/download/${pdb}.pdb
  fi


done < $1
