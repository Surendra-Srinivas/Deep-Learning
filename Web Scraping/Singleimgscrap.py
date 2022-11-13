from selenium import webdriver
from PIL import Image
import requests
import io

PATH = "/Users/surendrasrinivas/Downloads/chromedriver"
#wd = webdriver.Chrome(PATH)
image_url = "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fwww.looper.com%2Fimg%2Fgallery%2Fwe-now-know-another-reason-why-captain-america-was-able-to-survive-being-frozen%2Fl-intro-1603636388.jpg&f=1&nofb=1&ipt=7d0d350ad7d3879e2832a85e5deb7155f4d8a034ffd1daba30dc502bf395345f&ipo=images"
def download_image(download_path, url, file_name):
    try: 
        image_content = requests.get(url).content
        image_file = io.BytesIO(image_content)
        image = Image.open(image_file)
        file_path = download_path+file_name

        with open(file_path, "wb") as f:
            image.save(f, "JPEG")

        print("Downloaded !")
    except Exception as e:
        print("Failed - ", e)

download_image(" ", image_url, "test.jpg")
    


