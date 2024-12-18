import os
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.firefox.service import Service as FirefoxService
from selenium.webdriver.edge.service import Service as EdgeService
from selenium.webdriver.common.by import By
import time

def test_extension(driver):
    # Mở trang quản lý extension hoặc giao diện popup của extension
    driver.get("chrome://extensions/" if "chrome" in driver.capabilities['browserName'] else "edge://extensions/")

    # Mở giao diện popup của extension
    driver.execute_script("window.open('popup.html', '_blank');")
    driver.switch_to.window(driver.window_handles[-1])

    # Kiểm tra tiêu đề của popup
    assert "Popup" in driver.title, "Popup không hiển thị đúng."

    # Tương tác với giao diện
    try:
        chat_input = driver.find_element(By.ID, "chatInput")
        chat_input.send_keys("Tôi muốn tìm trang về công nghệ AI")
        chat_button = driver.find_element(By.ID, "chatButton")
        chat_button.click()
        time.sleep(5)  # Chờ API xử lý và trả kết quả
        response = driver.find_element(By.ID, "chatResponse").text
        print("Phản hồi của chatbot:", response)
    except Exception as e:
        print("Không tìm thấy phần tử hoặc xảy ra lỗi:", e)

def setup_driver(browser):
    # Lấy thông tin từ biến môi trường
    chrome_driver_path = os.getenv("CHROME_DRIVER_PATH")
    edge_driver_path = os.getenv("EDGE_DRIVER_PATH")
    firefox_driver_path = os.getenv("GECKO_DRIVER_PATH")
    extension_path = os.getenv("EXTENSION_PATH")

    if not extension_path:
        raise ValueError("EXTENSION_PATH không được cung cấp trong biến môi trường.")

    if browser == "chrome":
        options = webdriver.ChromeOptions()
        options.add_argument(f"--load-extension={extension_path}")
        return webdriver.Chrome(service=ChromeService(chrome_driver_path), options=options)
    elif browser == "edge":
        options = webdriver.EdgeOptions()
        options.add_argument(f"--load-extension={extension_path}")
        return webdriver.Edge(service=EdgeService(edge_driver_path), options=options)
    elif browser == "firefox":
        options = webdriver.FirefoxOptions()
        options.set_preference("xpinstall.signatures.required", False)
        options.add_argument(f"--load-extension={extension_path}")
        return webdriver.Firefox(service=FirefoxService(firefox_driver_path), options=options)
    else:
        raise ValueError("Trình duyệt không được hỗ trợ")

def main():
    for browser in ["chrome", "edge", "firefox"]:
        print(f"Đang kiểm thử trên {browser}...")
        try:
            driver = setup_driver(browser)
            test_extension(driver)
        except Exception as e:
            print(f"Lỗi khi kiểm thử trên {browser}: {e}")
        finally:
            if 'driver' in locals():
                driver.quit()

if __name__ == "__main__":
    main()
