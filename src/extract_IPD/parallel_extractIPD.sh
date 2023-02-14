#!/usr/bin/env bash

####################################################
# This script reads input files from a directory
# and runs IPD extraction using GNU parallel
####################################################

# input file should have tab-deliminated where each line contains
# directory_name / monotherapy file prefix / combination file prefix
wrapper() {
  while read line
    do
        indir=`echo "$line" | cut -f1`
        mono=`echo "$line" | cut -f2`
        comb=`echo "$line" | cut -f3`
        #echo "$indir"
        #echo "$mono"
        #echo "$comb"
        # change source file location as necessary
        Rscript Implementation_hh_cmd.R $indir $mono $comb
    done < $1
}

# just to check if input files are correct
wrapper2() {
  while read line
    do
        indir=`echo "$line" | cut -f1`
        mono=`echo "$line" | cut -f2`
        comb=`echo "$line" | cut -f3`
        echo "$indir"
        echo "$mono"
        echo "$comb"
    done < $1
}

export -f wrapper

# use gnu parallel for parallel computing
wrapper ../../data/raw/all_phase3_trials/all_phase3_input_list.txt

