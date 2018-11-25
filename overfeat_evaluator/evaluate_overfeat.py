from subprocess import call
from os import walk

images = 0
import subprocess

processes = []

file_ = open("ouput.txt", "w")

def get_overfeat_output(filename):
    # call(["./../bin/macos/overfeat", "-n", "1", './images/' + filename ])
    out = subprocess.Popen(["./../src/overfeat", "-n", "1", "-l", './images/' + filename ], stdout=file_)
    out.wait()


# for filename in range(1, 10):
    # p = subprocess.Popen(["./../bin/linux_64/overfeat", "-n", "1", './images/' + str(filename) ], stdout=file_)
    # processes.append(p)
for filename in range(1, 100):
    get_overfeat_output(str(filename) + '.jpg')
# for p in processes:
    # if p.wait() != 0:
        # print("Error")
# print("all done")
