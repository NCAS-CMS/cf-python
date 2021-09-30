* Change the version and date in `cf/__init__.py` (`__version__` and
  `__date__` variables)

* Ensure that the requirements on dependencies & their versions are
  up-to-date and consistent in both the `requirements.txt` and in
  `docs/source/installation.rst`; and in the `_requires` list and
  `LooseVersion` checks in `cf/__init__.py`.

* Make sure that `README.md` is up to date.

* Make sure that the `long_description` in `setup.py` is up to date.

* Make sure that `Changelog.rst` is up to date.

* Check that the documentation API coverage is complete:

  ```bash
  ./check_docs_api_coverage
  ```

  * If it is not complete, add any undocumented attributes, methods,
    functions and keyword arguments (e.g. as listed in the change log)
    to the `.rst` files in `docs/source/class/`.

* Check external links to the CF conventions are up to date in
  `docs/source/tutorial.rst` and `docs/source/field_analysis.rst`

* Create a link to the new documentation in
  `docs/source/releases.rst`, including the release date.

* Make sure that all the Data tests will run by setting
  `test_only = []` in `cf/test/test_Data.py` (check that none of
  the individual tests are un-commented so as to override this in the
  commented listing beneath).

* Test tutorial code:

  ```bash
  export PYTHONPATH=$PWD:$PYTHONPATH
  ./test_tutorial_code
  ```

* Build a development copy of the documentation using to check API
  pages for any new methods are present & correct, & that the overall
  formatting has not been adversely affected for comprehension by any
  updates in the latest Sphinx or theme etc. (Do not manually commit
  the dev build.)

  ```bash
  ./release_docs <vn> dev-clean # E.g. ./release_docs 3.3.0 dev-clean
  ```

* Check that no typos or spelling mistakes have been introduced to the
  documentation:

  * Run a dummy build of the documentation to detect invalid words:

     ```console
     $ cd docs
     $ make spelling build
     ```

  * If there are words raised with 'Spell check' warnings for the dummy
    build, such as:

    ```bash
    /home/sadie/cf-python/docs/source/class/cf.NetCDFArray.rst:18: Spell check: isw: element in the sequence isw the name of the group in which.
    Writing /home/sadie/cf-python/docs/spelling/class/cf.NetCDFArray.spelling
    /home/sadie/cf-python/docs/source/class/cf.Query.rst:3: Spell check: encapulates:  object encapulates a condition, such as.
    ```

    they may or may not be typos or mis-spellings. Address all the warnings
    (except those relating to files under `docs/source/class/`,
    `/attribute` or `/function` which will be fixed along with the origin
    docstrings after a 'latest' build) as follows:

    * If there are words that are in fact valid, add the valid words to
      the list of false positives for the spelling checker extension,
      `docs/source/spelling_false_positives.txt`.
    * Correct any words that are not valid in the codebase under `cf` or
      in the `docs/source` content files.

  * Note that, in the case there are many words raised as warnings, it
    helps to automate the above steps. The following commands are a means
    to do this processing:

    1. Copy all 'spell check' warnings output to STDOUT during the build to
       a file (here we use `spellings-file-1` as an example name).
    2. Cut all of the words from the warnings via
       `cat spellings-file-1 | cut -d':' -f 4 > spellings-file-2`
    3. Sift through these new words and remove any words that are true
       positives i.e. typos or mis-spellings. Correct them in the
       docstrings or documentation source files. If there are many
       instances across the docs, it helps to do a substitution of all
       occurences, e.g. via `find . -type f | xargs sed -i 's/<typo>/<correction>/g'`,
       though take care to have spaces surrounding words which may be
       part of other words, e.g. use
       `find . -type f | xargs sed -i 's/ ot / to /g'` to correct `ot` to `to`.
    4. Remove the leading whitespace character on each line and add
       all the new words to the current list of false positives:
       `sed 's/^.//' spellings-file-2 >> docs/source/spelling_false_positives.txt`
    5. Remove duplicate words and sort alphabetically via:
       `sort -u -o docs/source/spelling_false_positives.txt docs/source/spelling_false_positives.txt`

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
