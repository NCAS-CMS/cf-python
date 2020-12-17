'''Extract tutorial python code into a executable python scripts.

'''
import re
import sys

if len(sys.argv) >= 2:
    tutorials_rst = sys.argv[1:]
else:
    raise ValueError(
        "Must provide at least one .rst file contaiing code examples"
    )

for tutorial_rst in tutorials_rst:
    tutorial = []
    code_block = False
    caption = False

    with open(tutorial_rst, mode='r') as f:
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

    tutorial_py = tutorial_rst.replace('.rst', '.py')
     
    with open(tutorial_py, 'w') as f:
        f.write('\n'.join(tutorial))
# --- End: for
