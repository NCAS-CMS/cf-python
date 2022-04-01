#!/bin/sh

# -------------------------------------------------------------------------
# Test cfa
# -------------------------------------------------------------------------
# Note: use 'exit N' for different N on cfa commands instead of 'set -e'
# once to make it easier to determine downstream which cfa command fails
# (first), since the Python unittest calls this script by subprocess
# which just conveys that the script, not a specific line, errors.

sample_files=tmp_cfa_dir_in
test_file=tmp_test_cfa.nc
test_dir=tmp_cfa_dir

rm -fr $sample_files $test_dir $test_file
mkdir -p $sample_files

for f in `ls -1 test_file.nc file*.nc *.pp`
do
    ln -sf $PWD/$f $sample_files/$f
done

mkdir -p $test_dir

cfa -vm $sample_files/test_file.nc 2>&1 >/dev/null || exit 2
cfa --overwrite -o $test_file $sample_files/test_file.nc 2>&1 || exit 3
cfa --overwrite -d $test_dir  $sample_files/test_file.nc 2>&1 || exit 4

cfa -vm $sample_files/* 2>&1 >/dev/null || exit 5
cfa -vc $sample_files/* 2>&1 >/dev/null || exit 6
cfa -1 -vs $sample_files/file* 2>&1 >/dev/null || exit 7

#echo 0
cfa --overwrite -d $test_dir $sample_files/* 2>&1 || exit 8

#echo 1
rm -f $test_file
cfa -o $test_file $sample_files 2>&1 || exit 9

#echo 2
rm -f $test_file
cfa -n -o $test_file $sample_files/* 2>&1 || exit 10

#echo 3
rm -fr $test_dir $test_file $sample_files

# Test base="" for CFA fragment paths
mkdir $test_dir
cp test_file.nc $test_dir
cfa --cfa_base="" -f CFA4 -o $test_dir/test_file.nca $test_dir/test_file.nc
cp -a $test_dir ${test_dir}2
cfa -f NETCDF4 -o $test_file ${test_dir}2/test_file.nca
rm -fr $test_file $test_dir ${test_dir}2
