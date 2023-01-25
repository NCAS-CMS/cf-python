#!/bin/bash

python create_test_files.py
rc=$?
if [[ $rc != 0 ]]; then
  exit $rc
fi

python setup_create_field.py
rc=$?
if [[ $rc != 0 ]]; then
  exit $rc
fi

for file in test_*.py
do
  python $file
  rc=$?
  if [[ $rc != 0 ]]; then
    exit $rc
    # echo -e "\n\n$file FAILED \n\n"
  fi
done

