#!/bin/bash

for line in $(awk '/a/ {print $1 "\t" $2 "\t" $3}' $1); do
  echo $line
  #id=$(awk '{print $1}' $line)
  #echo $id
  #if [ $id == "RF00001" ]; then
  #  echo $line
  #fi
done
