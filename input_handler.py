import pynput
import time


class InputHandler:

    def __init__(self):
        self.capturing = False
        self.mouse = pynput.mouse.Controller()
        self.last_click_t = time.perf_counter()

    def on_press(self, key):
        if key == pynput.keyboard.Key.f7:
            if not self.capturing:
                print("starting capturing")
            self.capturing = True
        if key == pynput.keyboard.Key.f8:
            if self.capturing:
                print("stopping capturing")
            self.capturing = False

    def click(self):
        self.mouse.click(pynput.mouse.Button.left)

    def maybe_click(self):
        t = time.perf_counter()
        if self.last_click_t + 0.1 < t:
            self.last_click_t = t
            self.click()
