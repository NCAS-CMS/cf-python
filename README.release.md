* Change the version and date in `cf/__init__.py` (`__version__` and
  `__date__` variables)

* Ensure that the requirements on dependencies & their versions are
  up-to-date and consistent in both the `requirements.txt` and in the
  `_requires` list and `LooseVersion` checks in `cf/__init__.py`.

* Make sure that `README.md` is up to date.

* Make sure that `Changelog.rst` is up to date.

* Make sure that any new attributes, methods, functions and keyword arguments
  (as listed in the change log) have on-line documentation. This may
  require additions to the `.rst` files in `docs/source/classes/` and/or
  `docs/source/function`.

* Check external links to the CF conventions are up to date in
  `docs/source/tutorial.rst` and `docs/source/field_analysis.rst`

* Create a link to the new documentation in
  `docs/source/releases.rst`, including the release date.

* Make sure that all the Data tests will run by setting
  `self.test_only = []` in `cf/test/test_Data.py` (check that none of
  the individual tests are un-commented so as to override this in the
  commented listing beneath).

* Test tutorial code:

  ```bash
  export PYTHONPATH=$PWD:$PYTHONPATH
  d=$PWD
  cd docs/source
  ./extract_tutorial_code
  ./reset_test_tutorial
  cd test_tutorial
  python ../tutorial.py
  cd $d
  ```

* Build a development copy of the documentation using to check API
  pages for any new methods are present & correct, & that the overall
  formatting has not been adversely affected for comprehension by any
  updates in the latest Sphinx or theme etc. (Do not manually commit
  the dev build.)

  ```bash
  ./release_docs <vn> dev-clean # E.g. ./release_docs 3.3.0 dev-clean
  ```
  
* Create an archived copy of the documentation:

  ```bash
  ./release_docs <vn> archive # E.g. ./release_docs 3.3.0 archive
  ```

* Update the latest documentation:

  ```bash
  ./release_docs <vn> latest # E.g. ./release_docs 3.3.0 latest
  ```
  
* Create a source tarball:

  ```bash
  python setup.py sdist
  ```

* Test the tarball release using

  ```bash
  ./test_release <vn> # E.g. ./test_release 3.3.0
  ```

* Push recent commits using

  ```bash
  git push origin master
  ```
  
* Tag the release:

  ```bash
  ./tag <vn> # E.g. ./tag 3.3.0
  ```
  
* Upload the source tarball to PyPI. Note this requires the `twine`
  library (which can be installed via `pip`) and relevant project
  privileges on PyPI.

  ```bash
  ./upload_to_pypi <vn> # E.g. ./upload_to_pypi 3.3.0
  ```
