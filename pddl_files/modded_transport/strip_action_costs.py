# CHANGE DOMAIN FILES MANUALLY
import glob

BAD_STRINGS = ['(:metric','(=',' -> ']

def strip_file(filename, bad_strings=BAD_STRINGS):
  print('cleaning file {}'.format(filename), end='  ---  ')
  with open(filename, 'r') as f:
    lines = f.read().split('\n')
  print('it had {} lines'.format(len(lines)), end='  ---  ')
  for bad_string in bad_strings:
    lines = [line for line in lines if bad_string not in line]
  print('but now only {} lines!'.format(len(lines)))
  with open(filename, 'w') as f:
    f.write('\n'.join(lines))

for filename in glob.glob('./**/*.pddl', recursive=True):
  strip_file(filename)