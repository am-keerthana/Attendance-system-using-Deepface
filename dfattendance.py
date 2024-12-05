from deepface import DeepFace
import cv2 as cv 
import pandas as pd
#training
elon = cv.imread('C:/Users/vasu8/Documents/Python Scripts/faces/train/elon musk/th.jpeg')
sunny = cv.imread("C:/Users/vasu8/Documents/Python Scripts/faces/train/sunny leone/images (11).jpeg")
tom = cv.imread("C:/Users/vasu8/Documents/Python Scripts/faces/train/tom cruise/th (4).jpeg")

#veryfying
elon1 = cv.imread("C:/Users/vasu8/Documents/Python Scripts/faces/train/elon musk/th (9).jpeg")
sunny1 = cv.imread("C:/Users/vasu8/Documents/Python Scripts/faces/train/sunny leone/images (4).jpeg")

#application 

li = [elon,sunny,tom]
li_names = ['elon','sunny','tom']
li2 = [elon1,sunny1]
at = []
k = 0
for x in li :
    k = 0
    for y in li2:
        if DeepFace.verify(x,y)['verified'] == True :
            k = 1
            break 
    at.append(k)
print(at)
         