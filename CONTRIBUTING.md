Thank you for taking the time to consider making a contribution to the
cf-python package.

# General Guidelines

All questions, reports of bugs, and suggestions for enhancements
should be raised initially as GitHub issues at
https://github.com/NCAS-CMS/cf-python/issues

# Versioning

A three-level ``epoch.major.minor`` numeric version scheme is used,
e.g. ``3.4.1``

**Epoch** changes comprise

  * support for epoch-version changes to the CF conventions (e.g.
    upgrading from CF-1.x to CF-2.y);
  * support for epoch-version changes to the Python language (e.g.
    upgrading from Python 2 to Python 3);
  * *comprehensive* *backwards-incompatible* changes to the API.

**Major** changes comprise

  * support for new releases to the CF conventions (e.g. upgrading
    from CF-1.8 to CF-1.9);    
  * *limited* *backwards-incompatible* changes to the API, such as
    - changing the name of an *existing* function or method;
    - changing the behaviour of an *existing* function or method;
    - changing the name of an *existing* keyword parameter;
    - changing the default value of an *existing* keyword parameter;
    - changing the meaning of a value of an *existing* keyword
      parameter.

**Minor** changes comprise

  * *backwards-compatible* changes to the API, such as
    - introducing a *new* function or method;
    - introducing a *new* keyword parameter;
    - introducing a *new* permitted value of a keyword parameter;
  * changes to required versions of the dependencies;
  * bug fixes;
  * changes to the documentation;
  * code tidying.

# Change log

See the change log
(https://github.com/NCAS-CMS/cf-python/blob/main/Changelog.rst)
for the changes introduced by each version.
