from deepface import DeepFace

result = DeepFace.verify(img1_path = "C:/Users/vasu8/Documents/Python Scripts/faces/train/elon musk/download (1).jpeg",
                img2_path = "C:/Users/vasu8/Documents/Python Scripts/faces/train/elon musk/download (3).jpeg",model_name='Facenet'
)
print(result)
print('hello')
objs = DeepFace.analyze(img_path = "C:/Users/vasu8/Documents/Python Scripts/faces/train/elon musk/download (1).jpeg", 
        actions = ['age', 'gender', 'race', 'emotion']
)
print(objs)