# from deepface import DeepFace
# import cv2 as cv 
# import pandas as pd 

# excel_file = r"C:\Users\Keerthana\OneDrive\Desktop\deepface\Attendance.xlsx"

# try:
#     df = pd.read_excel(excel_file)
# except FileNotFoundError:
#     df = pd.DataFrame(columns=['Name', 'Date'])

# #training
# elon = cv.imread(r'C:\Users\Keerthana\OneDrive\Desktop\deepface\train\elon musk\th.jpeg')
# sunny = cv.imread(r"C:\Users\Keerthana\OneDrive\Desktop\deepface\train\sunny leone\images (11).jpeg")
# tom = cv.imread(r"C:\Users\Keerthana\OneDrive\Desktop\deepface\train\tom cruise\th (4).jpeg")

# #veryfying
# elon1 = cv.imread(r"C:\Users\Keerthana\OneDrive\Desktop\deepface\train\elon musk\th (9).jpeg")
# sunny1 = cv.imread(r"C:\Users\Keerthana\OneDrive\Desktop\deepface\train\sunny leone\images (4).jpeg")
# li = [elon,sunny,tom]
# li_names = ['elon','sunny','tom']
# l = []
# cap = cv.VideoCapture(0)
# while True:
#     ret,frame = cap.read()
    
#     for x in li:
#         if DeepFace.verify(frame,x,enforce_detection=False)['verified'] == True:
#             #l[li.index(x)] = l[li.index(x)]+1
#             name = li_names[li.index(x)]
#             if name not in l:
#                 l.append(name)
#                 current_date = pd.Timestamp('today').date()
#                 df1 = pd.DataFrame([[name ,current_date]],columns=['Name', 'Date'])
#                 df = pd.concat([df,df1],ignore_index=True)
#                 df.to_excel(excel_file, index=False)

#             cv.putText(frame, 'face recognized', (20,20), cv.FONT_HERSHEY_COMPLEX, 1.0, (255,0,0), thickness=1)
#         else:
#             continue     

#     cv.imshow('images',frame)
#     key=cv.waitKey(1)
   
# print(l)

from deepface import DeepFace
import cv2 as cv
import pandas as pd
from datetime import datetime

# File path for the Excel sheet
excel_file = r"C:\Users\Keerthana\OneDrive\Desktop\deepface\Attendance.xlsx"

# Try to read existing Excel file, or create a new DataFrame
try:
    df = pd.read_excel(excel_file)
except FileNotFoundError:
    df = pd.DataFrame(columns=['Name', 'Date', 'Time'])

# Training images
training_images = {
    'elon': cv.imread(r'C:\Users\Keerthana\OneDrive\Desktop\deepface\train\elon musk\th.jpeg'),
    'sunny': cv.imread(r"C:\Users\Keerthana\OneDrive\Desktop\deepface\train\sunny leone\images (11).jpeg"),
    'tom': cv.imread(r"C:\Users\Keerthana\OneDrive\Desktop\deepface\train\tom cruise\th (4).jpeg")
}

def check_attendance(name, current_date):
    return ((df['Name'] == name) & (df['Date'] == current_date)).any()

cap = cv.VideoCapture(0)

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        current_date = datetime.now().date()
        current_time = datetime.now().time()
        face_recognized = False

        for name, img in training_images.items():
            try:
                if DeepFace.verify(frame, img, enforce_detection=False, model_name="VGG-Face", distance_metric="cosine")['verified']:
                    face_recognized = True
                    if not check_attendance(name, current_date):
                        new_row = pd.DataFrame([[name, current_date, current_time.strftime("%H:%M:%S")]], 
                                               columns=['Name', 'Date', 'Time'])
                        df = pd.concat([df, new_row], ignore_index=True)
                        df.to_excel(excel_file, index=False)
                        cv.putText(frame, f'{name} attendance recorded', (20, 50), 
                                   cv.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), thickness=2)
                        print(f"Recorded attendance for {name}")
                    else:
                        cv.putText(frame, f'{name} attendance already taken', (20, 50), 
                                   cv.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 255), thickness=2)
                        print(f"{name}'s attendance already taken today")
                    break  # Exit the loop after recognizing a face
            except Exception as e:
                print(f"Error processing {name}: {str(e)}")

        if not face_recognized:
            cv.putText(frame, "Unknown", (20, 50), cv.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255), thickness=2)

        cv.imshow('Face Recognition', frame)
        
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv.destroyAllWindows()

print("Attendance tracking completed.")