import mediapipe as mp
import numpy as np
import cv2
import pyautogui
from pynput import keyboard
from pynput.keyboard import Controller, Key

import collections
import time
import os
import re

import alignment
import util
import sys
sys.path.insert(0, './emnist_model/')

import torch
import pytorch_model_class
from pytorch_model_class import DEVICE

import keyboard_util

def load_temp():
    confident_col = np.ones((21,1))
    templates = []
    templates_category = []
    # files = ["temp_data/arrow_temp.csv", "temp_data/fist_temp.csv","temp_data/five_temp.csv","temp_data/four_temp.csv","temp_data/one_temp.csv",]
    file_path = "./data/template_image_new_wide_data/"
    files = os.listdir(file_path)
    print(files)
    for file in files:
        name = re.findall(r'(\w+).', file)[0]
        temp = np.hstack((np.loadtxt(file_path + file, dtype = float, delimiter=','), confident_col))
        templates.append(temp)
        templates_category.append(name)
    return templates, templates_category


def flip_hand(handedness):
    if handedness == 'Right':
        return 'Left'
    else:
        return 'Right'

# organize landmarks for cursor control hand tracking
def hand_keypoints(hand_landmarks):
    points = []
    for landmark in hand_landmarks.landmark:
        points.append([landmark.x, landmark.y, landmark.z])
    return np.array(points)

def shoelace_area(points):
    x0, y0 = np.hsplit(points, 2)
    points1 = np.roll(points, -1, axis=0)
    x1, y1 = np.hsplit(points1, 2)
    combination = x0 * y1 - x1 * y0
    area = np.sum(combination) / 2
    return area, x0 + x1, y0 + y1, combination

def palm_center(keypoints):
    indices = [0, 1, 5, 9, 13, 17]
    points = keypoints[indices, :2]
    area, x, y, combination = shoelace_area(points)
    center_x = np.sum(x * combination) / (6 * area)
    center_y = np.sum(y * combination) / (6 * area)
    center = np.array([center_x, center_y])
    radius = int(np.min(np.linalg.norm(points - center, axis=1)) * np.mean(image.shape[:2]))
    center = tuple(np.int32(center * image.shape[1::-1]))
    return center, radius

def absolute(center):
    scale = 2
    print("center:", center)
    x = center[0] * screen_width // (scale * image_width)
    y = center[1] * screen_height // (scale * image_height)
    print(x, y)
    pyautogui.moveTo(x, y, _pause=False)

def absolute_scale(center):
    start_point = [0.50 * image_width, 0.25 * image_height]
    scale = screen_width // (0.25 * image_width)
    #scale_y = screen_height // (0.5 * height)
    out_x = scale * (center[0] - start_point[0])
    out_y = scale * (center[1] - start_point[1])
    pyautogui.moveTo(out_x, out_y, _pause=False)


def joystick(center, frame):
    mouse_vector = center - joystick_center
    length = np.linalg.norm(mouse_vector)
    if length > joystick_radius:
        mouse_vector = mouse_vector - np.array([joystick_radius, joystick_radius])
        # mouse_vector = mouse_vector / length * (length - joystick_radius)
        # mouse_move = np.multiply(np.power(abs(mouse_vector), 1.75) * 0.05, np.sign(mouse_vector))
        pyautogui.move(np.int32(mouse_vector)[0], np.int32(mouse_vector)[1], _pause=False)
        print('mouse vector', mouse_vector)
    cv2.line(frame, tuple(joystick_center), tuple(np.int32(center)), (255, 0, 0), 2)
    


# mediapipe setup
mp_drawing, mp_drawing_styles = util.mediapipe_draw_setup()
hands, mp_hands = util.mediapipe_hand_setup()

# gesture recognition setup
templates, templates_category = util.load_temp()

# video streaming setup
cap = cv2.VideoCapture(0)

# -------- begin keyboard input setup --------
keyboard = Controller()
is_keyboard_mode = True # TODO: will need to modify

# set up left and right list for gesture averaging
right_gesture_list = []
left_gesture_list = []

# array to hold drawn points
fingertip_path_right = []

# store previous gesture for rising edge of gesture
prev_left_gesture = None
prev_right_gesture = None
backspace_available = False
right_gesture = None
left_gesture = None
right_landmarks = None
left_landmarks = None
left_rising_edge_gesture = False
right_rising_edge_gesture = False

drawn_image = None

# load ML classification model
model = pytorch_model_class.CNN_SRM().to(DEVICE)
model.load_model(path = './emnist_model/saved_models')

# mapping of characters to digits
# mapping based on https://arxiv.org/pdf/1702.05373.pdf, balanced dataset+++
char_map = { 0: '0',  1: '1',  2: '2',  3: '3',  4: '4',  5: '5',  6: '6',  7: '7',  8: '8', 9: '9',
            10: 'A', 11: 'B', 12: 'C', 13: 'D', 14: 'E', 15: 'F', 16: 'G', 17: 'H', 18: 'I',
            19: 'J', 20: 'K', 21: 'L', 22: 'M', 23: 'N', 24: 'O', 25: 'P', 26: 'Q', 27: 'R',
            28: 'S', 29: 'T', 30: 'U', 31: 'V', 32: 'W', 33: 'X', 34: 'Y', 35: 'Z', 
            36: 'a', 37: 'b', 38: 'd', 39: 'e', 40: 'f', 41: 'g', 42: 'h', 43: 'n', 44: 'q',
            45: 'r', 46: 't'}
# -------- end keyboard input setup --------


# mode_mapping = {1: 'cursor', 2: 'scroll', 3: 'volume', 4: 'window', 5: 'safari'}
mode_mapping = {"one_left": 'cursor', "two_left": 'scroll', "three_left": 'volume', "four_left": 'window', "five_left": 'safari'}

time_start = None
leftclick_start = None
rightclick_start = None
center_queue = collections.deque(5 * [(0, 0)], 5)

success, image = cap.read()
screen_width, screen_height = pyautogui.size()
image_height, image_width = image.shape[:2]
joystick_center = np.array([int(0.75 * image_width), int(0.5 * image_height)])
joystick_radius = 40

pyautogui.FAILSAFE = False

win_name = "Mouse/Keyboard Controller"
# cv2.namedWindow(win_name, cv2.WND_PROP_ASPECT_RATIO)
cv2.imshow(win_name, image)
cv2.setWindowProperty(win_name, cv2.WND_PROP_TOPMOST, 1)

while cap.isOpened():
    # get tick count for measuring latency
    # https://docs.opencv.org/4.x/dc/d71/tutorial_py_optimization.html
    tick_start = cv2.getTickCount()

    success, image = cap.read()
    image = cv2.flip(image, 1)

    if not success:
        print("Ignoring empty camera frame.")
        # If loading a video, use 'break' instead of 'continue'.
        continue

    # always show bounding boxes if in keyboard mode
    if is_keyboard_mode:
        # draw the bounding box for the backspace area of the screen
        image, bksp_end_percentage = keyboard_util.draw_modifiers_boxes(image)

    # get hand keypoints
    mp_success, num_hands, results = util.mediapipe_process(image, hands)
    if mp_success:
        for i in range(num_hands):
            score, handedness, hand_landmarks = util.get_mediapipe_result(results, i)

            hand_gesture = util.recognize_gesture(templates, templates_category, hand_landmarks)
            util.mediapipe_draw(image, hand_landmarks, mp_hands, mp_drawing, mp_drawing_styles)

            landmark_data = []
            for point in hand_landmarks.landmark:
                landmark_data.append([point.x, point.y])
            
            if handedness == 'Right':
                right_gesture = keyboard_util.avg_gesture(right_gesture_list, hand_gesture)
                right_landmarks = hand_landmarks
                cv2.putText(image, 'right hand gesture: ' + str(right_gesture),
                (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (209, 80, 0, 255), 3)
            else:
                left_gesture = keyboard_util.avg_gesture(left_gesture_list, hand_gesture)
                left_landmarks = hand_landmarks
                cv2.putText(image, 'left hand gesture: ' + str(left_gesture),
                (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (209, 80, 0, 255), 3)

            # update rising edge for updating previous gestures
            if prev_right_gesture != right_gesture:
                right_rising_edge_gesture = True
            if prev_left_gesture != left_gesture:
                left_rising_edge_gesture = True

        if left_gesture == "fist_left" and right_gesture == "fist" \
           and (prev_left_gesture != "fist_left" or prev_right_gesture != "fist"):
            is_keyboard_mode = not is_keyboard_mode

# ------------------ BEGIN MOUSE MODE ------------------
        if not is_keyboard_mode:
            if left_gesture in mode_mapping:
                mode = mode_mapping[left_gesture]
                cv2.putText(image, 'mode: ' + mode, (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (209, 80, 0, 255), 3)
                if mode == 'scroll':
                    if right_gesture == "one":
                        # pyautogui.press('up')
                        pyautogui.scroll(5)
                    elif right_gesture == 'arrow':
                        pyautogui.press('down')
                        # pyautogui.press('down')
                        pyautogui.scroll(-5)
                    elif right_gesture == "two":
                        pyautogui.hscroll(10)
                    elif right_gesture == "three":
                        pyautogui.hscroll(-10) 
                elif mode == 'cursor' and right_landmarks is not None:
                    keypoints = hand_keypoints(right_landmarks)
                    center, radius = palm_center(keypoints)
                    center_queue.appendleft(center)
                    center = np.mean(center_queue, axis=0)
                    #absolute(center)
                    absolute_scale(center)
                    scale = screen_width // (0.5 * image_width)
                    #joystick(center, image)

                    # center and palm tracking
                    cv2.circle(image, tuple(np.int32(center)), 2, (0, 255, 0), 2)
                    cv2.circle(image, tuple(np.int32(center)), radius, (0, 255, 0), 2)
                    # cv2.circle(image, tuple(joystick_center), joystick_radius, (255, 0, 0), 2)

                    # hand movement area
                    # cv2.line(image, (0.5 * width, 0.25 * image_height), (0.5 * width, 0.75 * image_height), (0, 255, 0), 3)
                    # cv2.line(image, (1 * width, 0.25 * image_height), (1 * width, 0.75 * image_height), (0, 255, 0), 3)
                    # cv2.line(image, (0.5 * width, 0.25 * image_height), (1 * width, 0.25 * image_height), (0, 255, 0), 3)
                    # cv2.line(image, (0.5 * width, 0.75 * image_height), (1 * width, 0.75 * image_height), (0, 255, 0), 3)
                    cv2.rectangle(image, (int(0.50 * image_width), int(0.75 * image_height)), (int(0.80 * image_width), int(0.25 * image_height)), (0, 255, 0), 3)

                    if right_gesture == 'arrow':
                        if not leftclick_start:
                            leftclick_start = time.time()
                        elif time.time() - leftclick_start <= 1:
                            continue
                            # cv2.putText(image, "leftclick: %d" %( - (time.time() - leftclick_start)), (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (209, 80, 0, 255), 3)
                        else:
                            leftclick_start = None
                            pyautogui.click()
                    elif right_gesture == "two":
                        pyautogui.doubleClick()
                    elif right_gesture == "one":
                        pyautogui.rightClick()
                elif mode == 'volume':
                    if time_start is None:
                        time_start = time.time()
                    time_stamp = time.time()
                    if (time_stamp - time_start > 1):
                        time_start = time_stamp
                        if right_gesture == "one":
                            keyboard.tap(Key.media_volume_up)
                        elif right_gesture == 'arrow':
                            keyboard.tap(Key.media_volume_down)
                        elif right_gesture == "two":
                            keyboard.tap(Key.media_volume_mute)
                elif mode == 'window':
                    if time_start is None:
                        time_start = time.time()
                    time_stamp = time.time()
                    if (time_stamp - time_start > 1):
                        time_start = time_stamp
                        if right_gesture == "one": #switch to previous app
                            pyautogui.hotkey('command', 'tab')
                        elif right_gesture == "two": #browse windows
                            pyautogui.hotkey('ctrl', 'up')
                        elif right_gesture == "three": #minimize active window
                            pyautogui.hotkey('command', 'm')
                elif mode == 'safari':
                    if time_start is None:
                        time_start = time.time()
                    time_stamp = time.time()
                    if (time_stamp - time_start > 1):
                        time_start = time_stamp
                        if right_gesture == "one": # new tab
                            keyboard.press(Key.cmd)
                            keyboard.press('t')
                            keyboard.release('t')
                            keyboard.release(Key.cmd)
                            #time.sleep(0.5)
                        
                        if right_gesture == "two": # address bar
                            keyboard.press(Key.cmd)
                            keyboard.press('l')
                            keyboard.release('l')
                            keyboard.release(Key.cmd)
                            #time.sleep(0.5)

                        if right_gesture == "four": # decrease text size
                            keyboard.press(Key.cmd)
                            keyboard.press('-')
                            keyboard.release('-')
                            keyboard.release(Key.cmd)
                            #time.sleep(0.5)
                        
                        if right_gesture == "five": # increase text size
                            keyboard.press(Key.cmd)
                            keyboard.press('+')
                            keyboard.release('-')
                            keyboard.release(Key.cmd)
                            #time.sleep(0.5)
                            
                            
                        if right_gesture == 'arrow': # switch tab
                            keyboard.press(Key.ctrl)
                            keyboard.press(Key.tab)
                            keyboard.release(Key.tab)
                            keyboard.release(Key.ctrl)
                            #time.sleep(0.5)


                        if right_gesture == "three": #close tab
                            keyboard.press(Key.cmd)
                            keyboard.press('w')
                            keyboard.release('w')
                            keyboard.release(Key.cmd)
                            #time.sleep(0.5)
            
# ------------------ END MOUSE MODE ------------------
# ------------------ BEGIN KEYBOARD MODE ------------------
        else:
            # Initialize the Drawn Image into a new image and display window
            if drawn_image is None:
                drawn_image = np.zeros(image.shape[0:2], dtype=np.uint8)
                # cv2.imshow('Drawn Image', drawn_image)

            # if right hand is pose 'one', append fingertip path
            if right_gesture == 'one':
                x_pixel = int(right_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x\
                                * image_width)
                y_pixel = int(right_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y\
                                * image_height)
                fingertip_path_right.append((x_pixel, y_pixel))

            # if right hand is a fist, show path on rising edge and reset path
            if right_gesture == 'fist' and prev_right_gesture != 'fist':
                # on rising edge, draw path in separate window
                drawn_path_resized = keyboard_util.crop_and_draw_path(drawn_image, fingertip_path_right)

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
            if right_gesture == "four" and prev_right_gesture != "four":
                fingertip_path_right = []

            # get hand position (as a percentage)
            if left_landmarks is not None:
                hand_pos_percent = keyboard_util.modifiers_hand_position(mp_hands, left_landmarks)
            
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
            if left_gesture == "one_left" and prev_left_gesture != "one_left":
                keyboard.press(Key.backspace)
                keyboard.release(Key.backspace)

        # draw the path of the fingertip
        if len(fingertip_path_right) > 0:
            image = keyboard_util.draw_path(image, fingertip_path_right)        

        if right_rising_edge_gesture:
            prev_right_gesture = right_gesture
            right_rising_edge_gesture = False

        if left_rising_edge_gesture:
            prev_left_gesture = left_gesture
            left_rising_edge_gesture = False

    else:
        fingertip_path_right = []
# ------------------ END KEYBOARD MODE ------------------

    # Flip the image horizontally for a selfie-view display.
    cv2.imshow(win_name, image)
    # cv2.resizeWindow(win_name, 320, 180)

    # # calculate latency
    # tick_end = cv2.getTickCount()
    # latency = (tick_end - tick_start) / cv2.getTickFrequency()
    # print('Latency: {} seconds'.format(latency))
    
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()