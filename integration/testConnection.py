import os
import sys
sys.path.insert(0, "{}/integration".format(os.getcwd()))
sys.path.insert(0, "{}/control".format(os.getcwd()))
import penguinPi as ppi
import keyboardControl as Keyboard
import time
import matplotlib.pyplot as plt
from pynput.keyboard import Key, Listener

if __name__ == "__main__":
    keyboard = Keyboard.Keyboard()      
    Listener(on_press=keyboard.on_press).start()
    fig, ax = plt.subplots()
    while True:
        img = ppi.get_image()
        ax.cla()
        ax.imshow(img)
        plt.draw()
        plt.pause(0.01)
