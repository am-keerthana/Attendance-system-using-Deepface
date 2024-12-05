from deepface import DeepFace
import cv2 as cv 
import pandas as pd 

group_img = cv.imread("C:/Users/vasu8/Desktop/dooram.jpg")

satvik = cv.imread("C:/Users/vasu8/Desktop/satvik.jpg")
khasib = cv.imread("C:/Users/vasu8/Desktop/khasib.jpg")
anand = cv.imread("C:/Users/vasu8/Desktop/anand.jpg")
sai_jayanth = cv.imread("C:/Users/vasu8/Desktop/sai_jayanth.jpg")

li = [satvik,khasib,anand,sai_jayanth]
li_names = ['satvik','khasib','anand','sai_jayanth']
l = []
count = 0
for person in li :

    if DeepFace.verify(person,group_img,model_name='Facenet')['verified'] == True  :
        name = li_names[count]
        status = 'present'
        
        l.append([name,status])
    else:
        name = li_names[count]
        status = 'absent'
        
        l.append([name,status])
    count = count+1
print(l)


