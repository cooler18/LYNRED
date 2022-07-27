import subprocess

from numpy import byte
python_bin = "/home/godeta/PycharmProjects/LYNRED/venv/bin/python"

# Path to the script that must run under the virtualenv
script_file = "/home/godeta/PycharmProjects/LYNRED/FUSION/script/test2.py"

# print('All output at once:')
# proc = subprocess.Popen([python_bin, script_file], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
# output = proc.communicate(b'20')[0]
# print(output.decode(encoding="utf-8", errors="strict"))


print('One line at a time:')

proc = subprocess.Popen([python_bin, script_file], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
for i in range(5):
    proc = subprocess.Popen([python_bin, script_file], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    output = proc.communicate(b'%d' % (i+1))[0]
    # line = proc.stdout.readline()
    print(output.decode(encoding="utf-8", errors="strict"))
    # the real code does filtering here
    # print(line.rstrip())

#