* Change the version and date in `cf/__init__.py` (`__version__` and
  `__date__` variables).

* Make sure that `README.md` is up to date.

* Make sure that `Changelog.rst` is up to date.

* Check external links to the CF conventions are up to date in
  `docs/source/tutorial.rst` and `docs/source/field_analysis.rst`.

* Create a link to the new documentation in `docs/source/releases.rst`.

* Make sure that all the Data tests will run by setting `self.test_only = []`
  in `cf/test/test_Data.py` (check that none of the individual tests are
  un-commented so as to override this in the commented listing beneath).

* Create a source tarball: `python setup.py sdist`.

* Testing for Linux OS. Using a Linux machine:
  * test the tarball release using `test_release <vn>`
    (e.g. `./test_release 3.0.7`);
  * test the tutorial code:

        cd docs/source
        ./extract_tutorial_code
        ./reset_test_tutorial
        cd test_tutorial
        python ../tutorial.py

* Repeat both testing steps above for Mac OS using a Mac machine.

* Build a development copy of the documentation using
  `./release_docs <vn> dev` (e.g. `./release_docs 3.0.7 dev`) to check API
  pages for any new methods are present & correct, & that the overall
  formatting has not been adversely affected for comprehension by any updates
  in the latest Sphinx or theme etc. (Do not manually commit the dev build.)

* Create an archived copy of the documentation using
  `./release_docs <vn>` (e.g. `./release_docs 3.0.7`).

* Update the latest documentation using `./release_docs <vn> latest`
  (e.g. `./release_docs 3.0.7 latest`).

* Push recent commits using `git push origin master`.

* Tag the release using `./tag <vn>` (e.g. `./tag 3.0.6`).

* Upload the source tarball to PyPi via `./upload_to_pypi <vn>` (e.g.
  ./upload_to_pypi 3.1.0). Note this requires the `twine` library (which
  can be installed via `pip`) and relevant project privileges on PyPI.
