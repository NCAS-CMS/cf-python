Thank you for taking the time to consider making a contribution to the
cf-python package.

# General Guidelines

All questions, reports of bugs, and suggestions for enhancements
should be raised as GitHub issues:
https://github.com/NCAS-CMS/cf-python/issues

# Versioning

A three-level ``major.minor.trivial`` numeric version scheme is used,
e.g. ``3.3.1``

**Major** changes include
  * comprehensive backwards-incompatible changes to the API;
  * comprehensive refactoring of the code base;
  * support for major-version upgrades to the CF conventions (e.g
    upgrading from CF-0.9 to CF-1.0);
  * support for major version changes to the Python language (e.g
    upgrading from Python 2 to Python 3).

**Minor** changes include
  * support for minor-version upgrades to the CF conventions (e.g
    upgrading from CF-1.8 to CF-1.9);
  * limited backwards-incompatible changes to the API, such as
    - changing the name of an existing function or method;
    - changing the behaviour of an existing function or method;
    - changing the name of an existing keyword parameter;
    - changing the permitted values of an existing keyword parameter.

**trivial** changes include
  * bug fixes;
  * backwards-compatible changes to the API, such as
    - introducing a new function or method;
    - introducing a new keyword parameter;
    - introducing new permitted values of a keyword parameter;
  * changes to required versions of the dependencies;
  * changes to the documentation;
  * code tidying.
