"""
    Main File
    Modified from two_handed_gestures/gesture_mapping/video_recognition/
    

"""
import sys
import numpy as np
import cv2
import mediapipe as mp
import time


sys.path.append('../two_handed_gestures/gesture_mapping')
sys.path.append('./mnist_model/')

import alignment as alignment
import util
import torch
import pytorch_model_class

def draw_path(image, points, color=((0, 255, 0)), thickness=10):
    for i in range(0, len(points) - 1):
        image = cv2.line(image, points[i], points[i+1], color, thickness)

    return image

# custom mediapipe_hand_setup (instead of using util.mediapipe_hand_setup)
# such that we can set detection and tracking confidence
def mediapipe_hand_setup(model_complexity=0,
                         min_detection_confidence=0.9,
                         min_tracking_confidence=0.95):
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(model_complexity=model_complexity,
                           min_detection_confidence=min_detection_confidence,
                           min_tracking_confidence=min_tracking_confidence)
    return hands, mp_hands

def crop_and_draw_path(drawn_image, points):
    # do not draw if there are not enough points
    if len(points) < 2 :
        return None

    # factor that is scaled around image
    factor = 1.7
    
    # get the min and max x and y coordinate
    points_arr = np.array(points)
    min_x = min(points_arr[:, 0])
    max_x = max(points_arr[:, 0])
    min_y = min(points_arr[:, 1])
    max_y = max(points_arr[:, 1])

    # get the center of the drawn path
    center_x = (max_x + min_x) / 2
    center_y = (max_y + min_y) / 2

    # get the dimension of half of one of the sides of the square bounding box
    dist_from_center = max(center_x - min_x, center_y - min_y)

    # get x and y coordinates for cropping. Includes error checking for out of bounds and
    # a factor (so that the drawn path does not go to the edge of the image)
    cropped_x_min = int(max((center_x - factor * dist_from_center), 0))
    cropped_x_max = int(min((center_x + factor * dist_from_center), drawn_image.shape[1]))
    cropped_y_min = int(max((center_y - factor * dist_from_center), 0))
    cropped_y_max = int(min((center_y + factor * dist_from_center), drawn_image.shape[0]))

    # draw the path on an image the same size as input
    drawn_image.fill(0)
    
    # adjust the thickness based on the size of the overall drawing
    # ensure the thickness is also greater than 1
    thickness = max(1, int(dist_from_center / 10))

    drawn_image = draw_path(drawn_image, fingertip_path_right, color=255, thickness=thickness)

    # show that drawn image
    # cv2.imshow('Drawn Image', drawn_image)

    # notice that the slices are flipped, as x is the second dimension and y is the first dimension
    drawn_image = drawn_image[cropped_y_min:cropped_y_max, cropped_x_min:cropped_x_max]

    # resize image to 28x28 for MNIST dataset
    resized_image = cv2.resize(drawn_image, (28, 28))

    cv2.imshow('Resized Image', resized_image)

    return resized_image

# mediapipe setup
mp_drawing, mp_drawing_styles = util.mediapipe_draw_setup()
hands, mp_hands = mediapipe_hand_setup()
# gesture recognition setup
templates, templates_category = util.load_temp(\
    '../two_handed_gestures/gesture_mapping/data/template_image_new_wide_data/')

# video streaming setup
cap = cv2.VideoCapture(0)

# array to hold drawn points
fingertip_path_right = []

# store previous gesture (to test if path should be redrawn)
prev_left_gesture = None

drawn_image = None

# Set the parameters and create the model
D_in = 28 * 28
H1 = 100
H2 = 100
D_out = 10

# load ML classification model
model = pytorch_model_class.NetReluShallow(D_in, H1, H2, D_out)
model.load_model(path = './mnist_model/saved_models/')

# loop start
while cap.isOpened():
    # get tick count for measuring latency
    # https://docs.opencv.org/4.x/dc/d71/tutorial_py_optimization.html
    tick_start = cv2.getTickCount()

    # image shape: height * width = 720 * 1280, 9:16
    success, image = cap.read()
    image = cv2.flip(image, 1)
    image_height, image_width, channels = image.shape

    # Initialize the Drawn Image into a new image and display window
    if drawn_image is None:
        drawn_image = np.zeros(image.shape[0:2], dtype=np.uint8)
        # cv2.imshow('Drawn Image', drawn_image)

    if not success:
        print("Ignoring empty camera frame.")
        # streaming: continue; vedio: break
        continue
    # get hand keypoints
    mp_success, num_hands, results = util.mediapipe_process(image, hands)
    if mp_success:
        for i in range(num_hands):
            score, handedness, hand_landmarks = util.get_mediapipe_result(results, i)

            category = util.recognize_gesture(templates, templates_category, hand_landmarks)
            util.mediapipe_draw(image, hand_landmarks, mp_hands, mp_drawing, mp_drawing_styles)
            cv2.putText(image, 'pose: ' + category, (10, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (209, 80, 0, 255), 3)

            # if right hand is pose 'one', append fingertip path
            if category == 'one' and handedness == 'Right':
                x_pixel = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x\
                                  * image_width)
                y_pixel = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y\
                                * image_height)
                fingertip_path_right.append((x_pixel, y_pixel))

            # if left hand is a fist, show path on rising edge and reset path
            if category == 'fist_left' and handedness == 'Left':

                # on rising edge, draw path in separate window
                if prev_left_gesture != 'fist_left':
                    drawn_path_resized = crop_and_draw_path(drawn_image, fingertip_path_right)

                    if drawn_path_resized is not None:
                        tensor_input_image = drawn_path_resized.reshape((-1, 28*28))
                        character_confidences = model(torch.from_numpy(tensor_input_image) / 255)
                        
                        _, label = torch.max(character_confidences, axis=1)
                        print(label)


                # reset path
                fingertip_path_right = []

            # update previous left gesture
            if handedness == 'Left':
                prev_left_gesture = category

        # draw the path of the fingertip
        image = draw_path(image, fingertip_path_right)
        
    else:
        fingertip_path_right = []

    cv2.imshow('MediaPipe Hands', image)

    # calculate latency
    tick_end = cv2.getTickCount()
    latency = (tick_end - tick_start) / cv2.getTickFrequency()

    # print('Latency: {} seconds'.format(latency))

    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
