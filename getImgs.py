from seleniumwire import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
from PIL import Image
import requests
from io import BytesIO

def get_image():
    driver = webdriver.Chrome()
    driver.get("https://www.google.com/recaptcha/api2/demo")

    try:
        iframe = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.XPATH, "//iframe[contains(@src, 'recaptcha')]")))
        driver.switch_to.frame(iframe)
        checkbox = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.XPATH, '//*[@id="recaptcha-anchor"]')))
        del driver.requests  # Clear previous requests
        checkbox.click()
        driver.switch_to.default_content()
        time.sleep(3)
        
        for request in driver.requests:
            if "https://www.google.com/recaptcha/api2/payload?" in request.url:
                return request.url

    except Exception as error:
        print("Error:", error)
        return None
    finally:    
        # come back here and click results
        driver.quit()

def split_image_into_grid(image_url):
    response = requests.get(image_url)
    img = Image.open(BytesIO(response.content))
    
    img_width, img_height = img.size
    
    tile_width = img_width // 3
    tile_height = img_height // 3
    
    tiles = []
    for r in range(3):
        for c in range(3):
            left = c * tile_width
            upper = r * tile_height
            right = left + tile_width
            lower = upper + tile_height
            tile = img.crop((left, upper, right, lower))
            tiles.append(tile)
    return tiles

# Main execution
img_url = get_image()
print(f"Image URL: {img_url}")
image_parts = split_image_into_grid(img_url)

for i, part in enumerate(image_parts):
    part.save(f"results/test_{i+1}.jpg")

print(f"Saved {len(image_parts)} image tiles")