#!/bin/bash

while IFS=$'\t' read -r -a tmp; do

  fs="RF00001"
  tr="RF00005"
  id=${tmp[0]}
  pdb=${tmp[1]}
  sec=${tmp[2]}
  if [ $id == $tr ]; then
    echo $id $pdb $sec
    wget https://files.rcsb.org/download/${pdb}.pdb
  fi


done < $1
