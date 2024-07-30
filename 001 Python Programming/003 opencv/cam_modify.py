import cv2
import dlib
import numpy as np
from imutils import face_utils

############################################
# Task1 : Perform all of the given processes
# Author: Jebeom Chae
# Date:   2024-06-30
###########################################

# # Process 1: Read and rotate the image
# image = cv2.imread("person.jpg", cv2.IMREAD_COLOR)

# height, width, channel = image.shape
# matrix = cv2.getRotationMatrix2D((width/2, height/2), -90, 0.7)
# rotated_image = cv2.warpAffine(image, matrix, (width, height))

# # Process 2: Resize the image to 50%
# resized_image = cv2.resize(rotated_image, (width // 3, height // 3), interpolation=cv2.INTER_AREA)

# # Process 3: Extract facial landmarks using dlib
# p = "shape_predictor_68_face_landmarks.dat"
# detector = dlib.get_frontal_face_detector()
# predictor = dlib.shape_predictor(p)

# gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
# rects = detector(gray, 0)

# while True:
#     for (i, rect) in enumerate(rects):
#         shape = predictor(gray, rect)
#         shape = face_utils.shape_to_np(shape)

#         # Process 4: Crop the area in the image that contains the face.
#         (x, y, w, h) = cv2.boundingRect(np.array([shape]))
#         cropped_face = resized_image[y:y + h, x:x + w]

#         # Process 5: Read the sunglasses image (sunglasses.png) and resize it to fit the face image that you cropped.
#         sunglasses = cv2.imread("sunglasses.png", cv2.IMREAD_UNCHANGED)  # Ensure alpha channel is loaded

#         # Calculate the width and height of the sunglasses
#         eye_width = shape[45][0] - shape[36][0] # left eye (36) and right eye (45)
#         sunglasses_width = int(eye_width * 1.4)  
#         aspect_ratio = sunglasses.shape[0] / sunglasses.shape[1]
#         sunglasses_height = int(sunglasses_width * aspect_ratio)

#         resized_sunglasses = cv2.resize(sunglasses, (sunglasses_width, sunglasses_height), interpolation=cv2.INTER_AREA)

#         # Process 6: Put the sunglasses on a face (cropped image)
#         eyes_center = (shape[36] + shape[45]) // 2  
#         top_left = (eyes_center[0] - resized_sunglasses.shape[1] // 2, eyes_center[1] - resized_sunglasses.shape[0] // 2)

#         for i in range(resized_sunglasses.shape[0]):
#             for j in range(resized_sunglasses.shape[1]):
#                 if resized_sunglasses[i, j, 3] != 0:  
#                     y_offset = top_left[1] + i
#                     x_offset = top_left[0] + j
#                     if 0 <= y_offset < resized_image.shape[0] and 0 <= x_offset < resized_image.shape[1]:
#                         resized_image[y_offset, x_offset] = resized_sunglasses[i, j, :3]
#     cv2.imshow("Output", resized_image)
#     k = cv2.waitKey(5) & 0xFF
#     if k == 27:
#         break

# cv2.destroyAllWindows()


############################################
# Task2 : Transform cam.py to find the face in the camera in real time, 
#         and put sunglasses over that face
# Author: Jebeom Chae
# Date:   2024-06-30
###########################################

p = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)

sunglasses = cv2.imread("sunglasses.png", cv2.IMREAD_UNCHANGED)

# cap = cv2.VideoCapture("video_for_opencv.mp4")
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    resized_frame = cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2))
    
    gray = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
    
    rects = detector(gray, 0)
    
    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        
        (x, y, w, h) = cv2.boundingRect(np.array([shape]))
        cropped_face = resized_frame[y:y + h, x:x + w]

        aspect_ratio = sunglasses.shape[0] / sunglasses.shape[1]
        sunglasses_width = w
        sunglasses_height = int(sunglasses_width * aspect_ratio)

        resized_sunglasses = cv2.resize(sunglasses, (sunglasses_width, sunglasses_height), interpolation=cv2.INTER_AREA)
        
        eyes_center = (shape[36] + shape[45]) // 2 
        top_left = (eyes_center[0] - sunglasses_width // 2, eyes_center[1] - sunglasses_height // 2)

        # Overlay the sunglasses
        for i in range(resized_sunglasses.shape[0]):
            for j in range(resized_sunglasses.shape[1]):
                if resized_sunglasses[i, j, 3] != 0: 
                    y_offset = top_left[1] + i
                    x_offset = top_left[0] + j
                    if 0 <= y_offset < resized_frame.shape[0] and 0 <= x_offset < resized_frame.shape[1]:
                        resized_frame[y_offset, x_offset] = resized_sunglasses[i, j, :3]

    cv2.imshow('Video', resized_frame)
    
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
