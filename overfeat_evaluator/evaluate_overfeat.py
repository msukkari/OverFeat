from subprocess import call
from os import walk

images = 0
import subprocess

file_ = open("ouput.txt", "w")

def get_overfeat_output(filename):
    # call(["./../bin/macos/overfeat", "-n", "1", './images/' + filename ])
    subprocess.Popen(["./../bin/macos/overfeat", "-n", "6", './images/' + filename ], stdout=file_)
    file_.write('\n')

for filename in range(1, 101):
    get_overfeat_output(str(filename))