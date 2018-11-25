import os

for name in range(1, 1500):
    try:
        filename = './images/' + str(name) + '.jpg'
        b = os.path.getsize(filename)
        if b < 5000: # < 5kb
            os.remove(filename)
        print(b)
    except:
        print('nope')

