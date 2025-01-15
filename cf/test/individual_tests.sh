#!/bin/bash

for file in create_test_files*.py
do
  echo "Running $file"
  python $file
  rc=$?
done

file=setup_create_field.py
echo "Running $file"
python $file
rc=$?
if [[ $rc != 0 ]]; then
  exit $rc
fi

style="lots"

for file in test_*.py
do
  echo "Running $file"
  python $file
  rc=$?
  if [[ $rc != 0 ]]; then
      if [[ "$file" == "test_style.py" ]] ; then
	  style="none"
      else
	  exit $rc
          # echo -e "\n\n$file FAILED \n\n"
      fi
  fi
done

echo
if [[ "$style" == "none" ]] ; then
    echo "------------------------------------------"
    echo "All tests passed, APART FROM test_style.py"
    echo "------------------------------------------"
else
    echo "================"
    echo "All tests passed"    
    echo "================"
fi
