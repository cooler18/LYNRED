import sys
import lynred_py

sys.stderr.write('test2.py: starting\n')
sys.stderr.flush()
import numpy as np

next_line = int(sys.stdin.readline())
print(next_line)
for i in range(next_line):
    word = ' \n' + str(i)
    sys.stdout.write(word)
    sys.stdout.flush()

sys.stderr.write('test2.py: exiting\n')
sys.stderr.flush()
