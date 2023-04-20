import cv2
import os

# Path Declared
data_path = 'C:/Users/Pika pie/OneDrive/Desktop/Py_Projects/OpenCV_Projects/FaceMaskDetectionSystem/Dataset'

categories = os.listdir(data_path)  # List type
labels = [lbl for lbl in range(len(categories))]
label_dict = dict(zip(categories, labels))

# Collecting every imgs to a list
img_size = 100
images = []
target = []

for category in categories:
    folder_path = os.path.join(data_path, category)
    img_names = os.listdir(folder_path)
    
    
    for img_name in img_names:
        img_path = os.path.join(folder_path, img_name)
        img = cv2.imread(img_path)
        try:
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            resized_img = cv2.resize(gray_img, (img_size, img_size))
            images.append(resized_img)
            target.append(label_dict[category])
        except Exception as e:
            print("Exception :\t", e)

print(len(images), "s")
print(len(target), "s")

