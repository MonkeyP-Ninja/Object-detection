from tkinter import *
from tkinter import ttk

import cv2

root = Tk()
cam=cv2.VideoCapture(0)
ret, imge=cam.read()

canvas = Canvas(root, width = 1280, height = 720)
canvas.pack()

img = PhotoImage(imge)
canvas.create_image(20,20, anchor=NW, image=img)

root.mainloop()