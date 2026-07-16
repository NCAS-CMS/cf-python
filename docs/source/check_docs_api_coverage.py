"""Check the method-coverage of all classes in docs/source/class.rst.

All non-private methods of all such classes are checked for having an
entry in their corresponding class's file in docs/source/class/

For example, all cf.Field methods that do not start with an underscore
are checked for having an entry in docs/source/class/cf.Field.rst

Call as:

   $ python check_api_coverage.py <relative path to docs/source/>

"""

import inspect
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


if not source.endswith("source"):
    raise ValueError(f"Given directory {source} does not end with 'source'")

n_undocumented_methods = 0
n_missing_files = 0


for core in ("", "_core"):
    if core:
        if package.__name__ != "cfdm":
            # Only check core methods on cfdm package
            continue

        package = getattr(package, "core")

    with open(os.path.join(source, "class" + core + ".rst")) as f:
        api_contents = f.read()

    # TODO: after #958 is resolved, replace this by grabbing all classes from
    # '__all__', which defines the public API and therefore what must be
    # include - and do the same for the methods etc.
    class_names = [
        name
        for name, klass in inspect.getmembers(package, inspect.isclass)
        if klass.__module__.startswith(package.__name__ + ".")
        # Because of docstring substitution in cfdm, all functions imported
        # from there emerge as classes, with:
        # type= <class 'cfdm.core.meta.docstringrewrite.DocstringRewriteMeta'>
        # so when we try to extract classes only we end up with a lot of
        # functions mixed in. To filter these out, we can use the fact that
        # the functions emerge from just some modules, notably .functions etc.:
        and not klass.__module__.startswith(package.__name__ + ".functions")
        and not klass.__module__.startswith(package.__name__ + ".constants")
        # This just counts top-level read-write i.e. cf.read and cf.write
        and not klass.__module__.startswith(package.__name__ + ".read_write")
    ]

    for class_name in class_names:
        class_name = class_name.rstrip()

        full_class_name = f"{package.__name__}.{class_name}"

        if full_class_name not in api_contents:
            print(f"Class {full_class_name} not in class{core}.rst")
            n_missing_files += 1
            continue

        klass = getattr(package, class_name)

        methods = [
            method for method in dir(klass) if not method.startswith("_")
        ]

        rst_file = os.path.join(source, "class", full_class_name + ".rst")

        try:
            with open(rst_file) as f:
                rst_contents = f.read()

            for method in methods:
                method = ".".join([full_class_name, method])
                if method not in rst_contents:
                    n_undocumented_methods += 1
                    print(f"Method {method} not in {rst_file}")
        except FileNotFoundError:
            n_missing_files += 1
            print(f"File {rst_file} does not exist")

if n_undocumented_methods or n_missing_files:
    raise ValueError(
        f"Found {n_undocumented_methods} undocumented methods and "
        f"{n_missing_files} missing .rst files"
    )

print("All methods are documented")
