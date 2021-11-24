#!/bin/bash
while IFS=$'\t' read -r -a tmp; do

  fs="RF00001"
  tr="RF00005"
  id=${tmp[0]}
  pdb=${tmp[1]}
  sec=${tmp[2]}
  if [ $id == $fs ]; then
    if [ -e ${pdb}.pdb ]; then
      pdb_selchain -$sec ${pdb}.pdb > ../5srRNA_PDBs/${pdb}_${sec}_5srRNA.pdb
    fi
  fi
  if [ $id == $tr ]; then
    if [ -e ${pdb}.pdb ]; then
      pdb_selchain -$sec ${pdb}.pdb > ../tRNA_PDBs/${pdb}_${sec}_tRNA.pdb
    fi
  fi


done < $1
