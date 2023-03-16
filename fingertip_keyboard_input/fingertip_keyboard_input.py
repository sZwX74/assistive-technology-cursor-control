"""
    Main File
    Modified from two_handed_gestures/gesture_mapping/video_recognition/
    

"""
import sys
import numpy as np
import cv2
import mediapipe as mp
import time
from pynput.keyboard import Key, Controller

sys.path.insert(0, '../two_handed_gestures/gesture_mapping')
sys.path.insert(1, './emnist_model/')

import alignment as alignment
import util
import torch
import pytorch_model_class
from pytorch_model_class import DEVICE

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

def draw_modifiers_boxes(image, percent=0.2, color=((0, 255, 0)), thickness=3):
    image_height, image_width, __ = image.shape
    bksp_start_point = [0,0]
    height_ratio = 0.5
    bksp_end_point = [int(image_width * percent), int(image_height * height_ratio)]

    image = cv2.rectangle(image, bksp_start_point, bksp_end_point, color, thickness)

    cv2.putText(image, 'Back', [int(image_width*percent / 5), int(image_height * 0.5 * (2./5))],
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, thickness)
    cv2.putText(image, 'space', [int(image_width*percent / 5), int(image_height * 0.5 * (3./5))],
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, thickness)

    bksp_end_percentage = [percent, height_ratio]

    return image, bksp_end_percentage

def modifiers_hand_position(mp_hands, hand_landmarks):
    hand_points_interest_x = \
        [hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x, 
         hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x,
         hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC].x,
         hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].x,
         hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].x,
         hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].x,
         hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].x,
         hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].x
        ]

    hand_points_interest_y = \
        [hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y, 
         hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y,
         hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC].y,
         hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].y,
         hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y,
         hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y,
         hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].y,
         hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].y
        ]

    hand_pos = [np.mean(hand_points_interest_x), np.mean(hand_points_interest_y)]

    return hand_pos

#new
def most_frequent(gesture_list):
    return max(set(gesture_list), key = gesture_list.count)

#average recognized gestures in a sized list to pick most reliable category
def avg_gesture(gesture_list, category, max_size=3):
    if len(gesture_list) >= max_size:
        gesture_list.pop(0)
    gesture_list.append(category)
    return most_frequent(gesture_list)

#set up left and right list for gesture averaging
right_gesture_list = []
left_gesture_list = []

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

# store previous gesture for rising edge of gesture
prev_left_gesture = None
prev_right_gesture = None
backspace_available = False

drawn_image = None

# Set the parameters and create the model
D_in = 28 * 28
H1 = 100
H2 = 100
D_out = 10

# load ML classification model
model = pytorch_model_class.CNN_SRM().to(DEVICE)
model.load_model(path = './emnist_model/saved_models')

# keyboard setup
keyboard = Controller()

# mapping of characters to digits
# mapping based on https://arxiv.org/pdf/1702.05373.pdf, balanced dataset
char_map = { 0: '0',  1: '1',  2: '2',  3: '3',  4: '4',  5: '5',  6: '6',  7: '7',  8: '8', 9: '9',
            10: 'A', 11: 'B', 12: 'C', 13: 'D', 14: 'E', 15: 'F', 16: 'G', 17: 'H', 18: 'I',
            19: 'J', 20: 'K', 21: 'L', 22: 'M', 23: 'N', 24: 'O', 25: 'P', 26: 'Q', 27: 'R',
            28: 'S', 29: 'T', 30: 'U', 31: 'V', 32: 'W', 33: 'X', 34: 'Y', 35: 'Z', 
            36: 'a', 37: 'b', 38: 'd', 39: 'e', 40: 'f', 41: 'g', 42: 'h', 43: 'n', 44: 'q',
            45: 'r', 46: 't'}


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

    # draw the bounding box for the backspace area of the screen
    image, bksp_end_percentage = draw_modifiers_boxes(image)

    # get hand keypoints
    mp_success, num_hands, results = util.mediapipe_process(image, hands)
    if mp_success:
        for i in range(num_hands):
            score, handedness, hand_landmarks = util.get_mediapipe_result(results, i)

            category = util.recognize_gesture(templates, templates_category, hand_landmarks)
            util.mediapipe_draw(image, hand_landmarks, mp_hands, mp_drawing, mp_drawing_styles)

            #gesture averaging
            if handedness == 'Right':
                category = avg_gesture(right_gesture_list, category)

            if handedness == "Left":
                category = avg_gesture(left_gesture_list, category)
        
            cv2.putText(image, 'pose: ' + category, (10, image_height - 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (209, 80, 0, 255), 3)

            if handedness == 'Right':
                # if right hand is pose 'one', append fingertip path
                if category == 'one':
                    x_pixel = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x\
                                    * image_width)
                    y_pixel = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y\
                                    * image_height)
                    fingertip_path_right.append((x_pixel, y_pixel))

                # if right hand is a fist, show path on rising edge and reset path
                if category == 'fist' and prev_right_gesture != 'fist':

                    # on rising edge, draw path in separate window
                    drawn_path_resized = crop_and_draw_path(drawn_image, fingertip_path_right)

                    if drawn_path_resized is not None:
                        tensor_input_image_reshaped = drawn_path_resized.T.reshape((1, 1, 28, 28))
                        tensor_input_image = torch.from_numpy(tensor_input_image_reshaped) / 255
                        character_confidences = model(tensor_input_image.to(DEVICE))
                        
                        _, label = torch.max(character_confidences, axis=1)
                        print(f'Recognized character: {char_map[int(label)]}')
                        keyboard.press(char_map[int(label)])
                        keyboard.release(char_map[int(label)])

                    # reset path
                    fingertip_path_right = []

                # if right hand is a four, discard drawing
                if category == "four" and prev_right_gesture != "four":
                    fingertip_path_right = []

                # update previous right gesture
                prev_right_gesture = category


            if handedness == "Left":
                # get hand position (as a percentage)
                hand_pos_percent = modifiers_hand_position(mp_hands, hand_landmarks)
                
                # if in backspace box, backspace once
                if hand_pos_percent[0] < bksp_end_percentage[0] and hand_pos_percent[1]\
                                                                    < bksp_end_percentage[1]:
                    if not backspace_available:
                        keyboard.press(Key.backspace)
                        keyboard.release(Key.backspace)
                        backspace_available = True
                
                # once hand leaves the box, make backspace possible again
                else:
                    backspace_available = False

                # backspace once using gesture
                if category == "one_left" and prev_left_gesture != "one_left":
                    keyboard.press(Key.backspace)
                    keyboard.release(Key.backspace)

                # update previous right gesture
                prev_left_gesture = category

        # draw the path of the fingertip
        print(len(fingertip_path_right))
        if len(fingertip_path_right) > 0:
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
