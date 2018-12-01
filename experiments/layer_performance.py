import subprocess

FAST_MODEL_LAYER = 8
ACCURATE_MODEL_LAYER = 9

file = open("weight_dropping_effect_to_100.txt", "w")

def test_layer(layer, percentage, accurate):
	if accurate:
		out = subprocess.Popen(["./../src/overfeat", "-n", "1000", "-l", "./../samples/bee.jpg", "-z", str(layer), "-c", str(percentage)], stdout=subprocess.PIPE)
		out2 = subprocess.Popen(["grep", "-w", "^bee [0-9]"], stdin=out.stdout, stdout=file)
		out2.communicate()[0]
	else:
		out = subprocess.Popen(["./../src/overfeat", "-n", "1000", "./../samples/bee.jpg", "-z", str(layer), "-c", str(percentage)], stdout=subprocess.PIPE)
		out2 = subprocess.Popen(["grep", "-w", "^bee [0-9]"], stdin=out.stdout, stdout=file)
		out2.communicate()[0]
	print('saved layer:', layer, ' percentage', percentage)

# test fast model
for layer in range(1, FAST_MODEL_LAYER + 1):
	for percentage in range(0, 101, 5):
		for i in range(5):
			test_layer(layer, percentage, False)

# test accurate model
for layer in range(1, ACCURATE_MODEL_LAYER + 1):
	for percentage in range(0, 101, 5):
		for i in range(5):
			test_layer(layer, percentage, True)

