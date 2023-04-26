import cv2
import numpy as np

def draw_path(image, points, color=((0, 255, 0)), thickness=10):
    for i in range(0, len(points) - 1):
        image = cv2.line(image, points[i], points[i+1], color, thickness)
    return image

def crop_and_draw_path(drawn_image, points_list):
    points =  [item for sublist in points_list for item in sublist]
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

    for points_arr in points_list:
        if len(points_arr) < 2 :
            continue
        drawn_image = draw_path(drawn_image, points_arr, color=255, thickness=thickness)

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
    height_ratio = 0.5
    bksp_start_point = [0,0]
    space_start_point = [0, int(image_height * height_ratio)]
    bksp_end_point = [int(image_width * percent), int(image_height * height_ratio)]
    space_end_point = [int(image_width * percent), image_height]

    # backspace
    image = cv2.rectangle(image, bksp_start_point, bksp_end_point, color, thickness)

    cv2.putText(image, 'Back', [int(image_width*percent / 5), int(image_height * 0.5 * (2./5))],
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, thickness)
    cv2.putText(image, 'space', [int(image_width*percent / 5), int(image_height * 0.5 * (3./5))],
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, thickness)
    
    # space
    image = cv2.rectangle(image, space_start_point, space_end_point, color, thickness)

    cv2.putText(image, 'Space', [int(image_width*percent / 5), int(image_height * (3./4))],
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, thickness)

    bksp_space_end_percentage = [percent, height_ratio]

    return image, bksp_space_end_percentage

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

def most_frequent(gesture_list):
    return max(set(gesture_list), key = gesture_list.count)

#average recognized gestures in a sized list to pick most reliable category
def avg_gesture(gesture_list, category, max_size=3):
    if len(gesture_list) >= max_size:
        gesture_list.pop(0)
    gesture_list.append(category)
    return most_frequent(gesture_list)