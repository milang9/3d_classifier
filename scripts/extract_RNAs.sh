#!/bin/bash
while IFS=$'\t' read -r -a tmp; do

  rf="RF00027"
  #tr="RF00005"
  id=${tmp[0]}
  pdb=${tmp[1]}
  sec=${tmp[2]}
  if [ $id == $rf ]; then
    if [ -e ${pdb}.pdb ]; then
      pdb_selchain -$sec ${pdb}.pdb > ../test_set_16-25/${pdb}_${sec}_rf27.pdb
    fi
  fi
#  if [ $id == $tr ]; then
#    if [ -e ${pdb}.pdb ]; then
#      pdb_selchain -$sec ${pdb}.pdb > ../tRNA_PDBs/${pdb}_${sec}_tRNA.pdb
#    ficcd
#  fi


done < $1
