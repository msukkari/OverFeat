import csv
import requests

def download_images(fname):
    with open(fname) as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    filename = 0
    total_images = 100
    for url in content:
        # if filename == total_images:
            # break
        filename = filename + 1
        print(url)
        try:
            result = requests.get(url, timeout=2, stream=True)
            if result.status_code == 200:
                image = result.raw.read()
                open(str(filename) + ".jpg", "wb").write(image)
                print("saved", filename)
        except:
            print('hi')
# download_with_url('http://www.philipphauer.de/info/bio/wechselwarm-gleichwarm/gleichwarm-k03.jpg')
download_images('urls.txt')
