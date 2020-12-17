'''Extract tutorial python code into an executable python script.

'''
import re

tutorial = []
code_block = False
caption = False

with open('tutorial.rst', mode='r') as f:
    for line in f.readlines():
        line = line.strip()

        code = re.split('^\s*>>>\s', line)

        if len(code) == 2:
            if re.findall('# Raises Exception\s*$', line):
                code[1] = 'try:\n    ' + code[1]+ '\nexcept:\n    pass'
        else:
            code = re.split('^\s*\.\.\.\s', line)

        if len(code) == 2:
            tutorial.append(code[1])
            continue

        if re.match('\s*\.\. Code Block Start', line):
            code_block = True
            tutorial.append('# Start of code block')
            continue

        if code_block:
            if re.match('\s*\.\. code-block::', line):
                continue
            
            if not caption and re.match('\s*:caption:', line):
                caption = True
                continue

            if caption:
                # Blank line marks end of caption
                if re.match('\s*$', line):
                    caption = False

                continue
                    
            if re.match('\.\. Code Block End', line):
                code_block = False
                tutorial.append('# End of code block')
                continue
            
            tutorial.append(line.replace('   ', '', 1))
# --- End: with

tutorial.append('')
            
with open('tutorial.py', 'r+') as f:
    f.seek(0)
    f.write('\n'.join(tutorial))
    f.truncate()
