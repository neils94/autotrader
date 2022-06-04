import pyautogui
import datetime
import time

def screenshot():
    path_to_save_to = "/Users/neilsuji/Downloads/screenshot_images/"
    screenshot_taken = pyautogui.screenshot(path_to_save_to+str(datetime.datetime.now()) + ".png")

for i in range(1):
    pyautogui.moveTo(0, 500)

    while True:
        time.sleep(60 - time.gmtime()[5] % 60)
        screenshot()

        break;





