# This program will demand two images with clear faces in each of them
# Then, it will compare the difference between the two images

# For example,

# If the two images are of the same person but with two different angles, lightning, contrast or anything,
# Program will return "TRUE" indicating that the person in the two images are the same.

# If the two images are of different persons
# Program will return "FALSE" indicating that the persons in the two images are different.

# Also, the program will return a value between 0 and 1, representing the similarity of faces in the two images.
# More the value, less similar are the two images.
# Less the value, more similar are the two images.

# Importing important dependencies such as CV2, Numpy and face_recognition
import cv2
import numpy as np
import face_recognition

# Function to load an image from a specified file
# Put your path of Image1 here
img_kim = face_recognition.load_image_file('Images/Kunal.jpeg')

# cvtColor used to convert the color of the image from BGR to RGB
img_kim = cv2.cvtColor(img_kim, cv2.COLOR_BGR2RGB)

# Put your path of Image2 here
img_test = face_recognition.load_image_file('Images/Pranshu.jpeg')
img_test = cv2.cvtColor(img_test, cv2.COLOR_BGR2RGB)

# face_locations used for defining the 68 locations
faceLoc = face_recognition.face_locations(img_kim)[0]

# face_encodings to give unique encodings to every location of the face
encodeKim = face_recognition.face_encodings(img_kim)[0]

# Creating a rectangle around the face in Image1 to define the image with pink color
cv2.rectangle(img_kim, (faceLoc[3], faceLoc[0]), (faceLoc[1], faceLoc[2]), (255, 0, 255), 2)

faceLocTest = face_recognition.face_locations(img_test)[0]
encodeKimTest = face_recognition.face_encodings(img_test)[0]

# Creating a rectangle around the face in Image2 to define the image with green color
cv2.rectangle(img_test, (faceLocTest[3], faceLocTest[0]), (faceLocTest[1], faceLocTest[2]), (255, 0, 255), 2)

# print(faceLoc)

results = face_recognition.compare_faces([encodeKim], encodeKimTest)
faceDis = face_recognition.face_distance([encodeKim], encodeKimTest)

# results will print either True or False
# faceDis will print the value between 0 and 1 which will represent the similarity between the two images.
print(results, faceDis)

# Some extra editing in Image2
cv2.putText(img_test, f'{results} {round(faceDis[0], 3)}', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 255), 2)

# Printing the images
cv2.imshow('Image 1', img_kim)
cv2.imshow('Image 2', img_test)

cv2.waitKey(0)
