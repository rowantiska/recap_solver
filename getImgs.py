from seleniumwire import webdriver  # Only import this one
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time

driver = webdriver.Chrome()
driver.get("https://www.google.com/recaptcha/api2/demo")

try:
    iframe = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.XPATH, "//iframe[contains(@src, 'recaptcha')]")))
    driver.switch_to.frame(iframe)
    checkbox = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.XPATH, '//*[@id="recaptcha-anchor"]')))
    del driver.requests
    checkbox.click()
    driver.switch_to.default_content()
    time.sleep(5)
    
    for request in driver.requests:
        if "https://www.google.com/recaptcha/api2/payload?" in request.url:
            print(request.url)

except Exception as error:
    print("Error:", error)
finally:    
    driver.quit()