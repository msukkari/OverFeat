import subprocess

FAST_MODEL_LAYER = 8
ACCURATE_MODEL_LAYER = 9

file = open("weight_dropping_effect_to_100.txt", "w");

def test_layer(layer, percentage, accurate):
	if accurate:
		out = subprocess.Popen(["./../src/overfeat", "-n", "1", "-l", "./../samples/bee.jpg", "-z", str(layer), "-c", str(percentage)], stdout=file)
		out.wait();
	else:
		out = subprocess.Popen(["./../src/overfeat", "-n", "1", "./../samples/bee.jpg", "-z", str(layer), "-c", str(percentage)], stdout=file)
		out.wait();
	print('saved layer:', layer, ' percentage', percentage)

# test fast model
for layer in range(1, FAST_MODEL_LAYER + 1):
	for percentage in range(0, 101, 5):
	        test_layer(layer, percentage, False)
# test accurate model
for layer in range(1, ACCURATE_MODEL_LAYER + 1):
	for percentage in range(0, 101, 5):
		test_layer(layer, percentage, True)
