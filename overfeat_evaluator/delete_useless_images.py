import os

for name in range(1, 101):
    try:
        filename = './images/' + str(name)
        b = os.path.getsize(filename)
        if b < 5000: # < 5kb
            os.remove(filename)
        print(b)
    except:
        print('nope')

