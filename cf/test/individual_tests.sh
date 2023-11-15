#!/bin/bash

for file in create_test_files*.py
do
  echo "Running $file"
  python $file
  rc=$?
  if [[ $rc != 0 ]]; then
    exit $rc
  fi
done

file=setup_create_field.py
echo "Running $file"
python $file
rc=$?
if [[ $rc != 0 ]]; then
  exit $rc
fi

for file in test_*.py
do
  echo "Running $file"
  python $file
  rc=$?
  if [[ $rc != 0 ]]; then
    exit $rc
    # echo -e "\n\n$file FAILED \n\n"
  fi
done

