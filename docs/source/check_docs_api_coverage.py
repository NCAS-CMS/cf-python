'''Check the method-coverage of all classes listed in
docs/source/class.rst

All non-private methods of all such classes are checked for having an
entry in their corresponding class's file in docs/source/class/

For example, all cf.Field methods that do not start with an underscore
are checked for having an entry in docs/source/class/cf.Field.rst

Call as:

   $ python check_api_coverage.py <docs/source/ directory path>

'''
import os
import sys

import cf as package

if len(sys.argv) == 2:
    source = sys.argv[1]
else:
  raise ValueError(
      "Must provide the 'source' directory as the "
      "only positional argument"
    )


if not source.endswith('source'):
    raise ValueError(
        "Given directory {} does not end with 'source'".format(source)
    )

n_undocumented_methods = 0
n_missing_files = 0

for core in ('', '_core'):
    if core:
        if package.__name__ != 'cfdm':
            # Only check core methods on cfdm package
            continue
        
        package = getattr(package, 'core')
        
    with open(os.path.join(source, 'class'+core+'.rst')) as f:
        api_contents = f.read()
    
    class_names = [i.split('.')[-1] for i in api_contents.split('\n')
                   if package.__name__+'.' in i]
    
    for class_name in class_names:
        class_name = class_name.rstrip()
        
        klass = getattr(package, class_name)
        
        methods = [method for method in dir(klass)
                   if not method.startswith('_')]
        
        class_name = '.'.join([package.__name__, class_name])
        
        rst_file = os.path.join(source, 'class', class_name+'.rst')

        
        try:
            with open(rst_file) as f:
                rst_contents = f.read()
                
            for method in methods:
                method = '.'.join([class_name, method])
                if method not in rst_contents:
                    n_undocumented_methods += 1
                    print (
                        "Method {} not in {}".format(
                            method, os.path.join(source, 'class', rst_file)
                       )
                    )
        except FileNotFoundError:
            n_missing_files += 1
            print (
                "File {} does not exist".format(rst_file)
            )
# --- End: for

# Raise an exception to ensure a non-zero shell return code
if n_undocumented_methods:
    print("Found undocumented method(s)")
    
if n_missing_files:
    print("Found missing .rst file(s)")
    
if n_undocumented_methods or n_missing_files:
    raise ValueError(
        "Found undocumented methods ({}) or missing .rst files ({})".format(
            n_undocumented_methods, n_missing_files
        )
    )

print('All methods are documented')
