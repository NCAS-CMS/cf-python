class Container:
    '''Base class for storing components.

    .. versionadded:: 3.7.0

    '''
    def __docstring_substitution__(self):
        '''TODO
    Substitutions may be easily modified by overriding the
    __docstring_substitution__ method.
    Modifications can be applied to any class, and will only apply to
    that class and all of its subclases.
    If the key is a string then the special subtitutions will be
    applied to the dictionary values after replacement in the
    docstring.
    If the key is a compiled regular expession then the special
    subtitutions will be applied to the match of the regular
    expression after replacement in the docstring.
    For example::
       def __docstring_substitution__(self):
           def _upper(match):
               return match.group(1).upper()
           out = {
                  # Simple substitutions 
                  '{{repr}}': 'CF '
                  '{{foo}}': 'bar'
                  '{{parameter: `int`}}': """parameter: `int`
               This parameter does something to `{{class}}`
               instances. It has no default value.""",
                   # Regular expression subsititions
                   # 
                   # Convert text to upper case
                   re.compile('{{<upper (.*?)>}}'): _upper
            }
           return out
        '''
        return {
            # Examples with repr output
            '{{repr}}': 'CF ',

            # i
            '{{i: deprecated at version 3.0.0}}':
            """i: deprecated at version 3.0.0
            Use the *inplace* parameter instead.""",
        }

# --- End: class
