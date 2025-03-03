.. currentmodule:: cf
.. default-role:: obj

.. _Contributing:

**Contributing**
================

----

Version |release| for version |version| of the CF conventions.

.. contents::
   :local:
   :backlinks: entry

**Reporting bugs**
------------------

Please report bugs via a new issue in issue tracker
(https://github.com/NCAS-CMS/cf-python/issues), using the **Bug
report** issue template.

----

**Feature requests and suggested improvements**
-----------------------------------------------

Suggestions for new features and any improvements to the
functionality, API, documentation and infrastructure can be submitted
via a new issue in issue tracker
(https://github.com/NCAS-CMS/cf-python/issues), using the **Feature
request** issue template.

----

**Questions**
-------------

Questions, such as "how can I do this?", "why does it behave like
that?", "how can I make it faster?", etc., can be raised via a new
issue in issue tracker (https://github.com/NCAS-CMS/cf-python/issues),
using the **Question** issue template.

----

**Preparing pull requests**
---------------------------

Pull requests should follow on from a discussion in the issue tracker
(https://github.com/NCAS-CMS/cf-python/issues).

Fork the cf-python GitHub repository
(https://github.com/NCAS-CMS/cf-python).

..  note::
    The cf-python GitHub repository uses ``main`` as the name of its
    default branch, so you must refer to ``main`` when you need to
    reference the default branch. It is useful to use this as the
    name of the default branch on your fork, if you use one, too.

Clone your fork locally and create a branch:

.. code-block:: console
	  
    $ git clone git@github.com:<YOUR GITHUB USERNAME>/cf-python.git
    $ cd cf-python
    $ git checkout -b <your-bugfix-feature-branch-name main>

Break your edits up into reasonably-sized commits, each representing
a single logical change:

.. code-block:: console
	  
    $ git commit -a -m "<COMMIT MESSAGE>"

Create a new changelog entry in ``Changelog.rst``. The entry should be
written (where ``<description>`` should be a *brief* description of
the change) as:

.. code-block:: rst

   * <description> (https://github.com/NCAS-CMS/cf-python/issues/<issue number>)

Run the test suite to make sure the tests all pass:
	
.. code-block:: console

   $ cd cf/test
   $ python run_tests.py

Add your name to the list of contributors list at
``docs/source/contributing.rst``.

Finally, make sure all commits have been pushed to the remote copy of
your fork and submit the pull request via the GitHub website, to the
``main`` branch of the ``NCAS-CMS/cf-python`` repository. Make sure
to reference the original issue in the pull request's description.

Note that you can create the pull request while you're working on
this, as it will automatically update as you add more commits. If it is
a work in progress, you can mark it initially as a draft pull request.

----

**Contributors**
----------------

We would like to acknowledge and thank all those who have contributed
ideas, code, and documentation to the cf library:

* A James Phillips
* Alan Iwi
* Ankit Bhandekar
* Bruno P. Kinoshita
* Bryan Lawrence
* Charles Roberts
* David Hassell
* Evert Rol  
* Javier Dehesa
* Jonathan Gregory
* Klaus Zimmermann
* Kristian Sebastián
* Mark Rhodes-Smith
* Matt Brown
* Michael Decker
* Sadie Bartholomew
* Thibault Hallouin
* Tim Bradshaw
* Tyge Løvset
