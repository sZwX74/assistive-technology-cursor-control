import cv2
import mediapipe as mp
import pyautogui

cam = cv2.VideoCapture(0)
face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
screen_w, screen_h = pyautogui.size()

def set_mouse_pos(anchor, pupil):
    eye_anchor_x, eye_anchor_y = anchor
    pupil_center_x, pupil_center_y = pupil
    x_rel = pupil_center_x - eye_anchor_x
    y_rel = pupil_center_y - eye_anchor_y

    cursor = (x_rel*20 + screen_w/2, y_rel*40 + screen_h/2)
    pyautogui.moveTo(*cursor)

while True:
    _, frame = cam.read()
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output = face_mesh.process(rgb_frame)
    landmark_points = output.multi_face_landmarks
    frame_h, frame_w, _ = frame.shape
    if landmark_points:
        landmarks = landmark_points[0].landmark

        # Right eye ball landmarks (pupil boundaries)
        right_eyeball = range(474, 478)
        for id in right_eyeball:
            landmark = landmarks[id]
            x = int(landmark.x * frame_w)
            y = int(landmark.y * frame_h)
            cv2.circle(frame, (x, y), 3, (0, 255, 0))
            # if id == right_eyeball[0]:
            #     screen_x = screen_w * landmark.x
            #     screen_y = screen_h * landmark.y
            #     pyautogui.moveTo(screen_x, screen_y)
        
        # Calculate eye center
        eye_anchor = [263, 463, 374, 386]

        eye_anchor_x = 0
        eye_anchor_y = 0

        for id in eye_anchor:
            landmark = landmarks[id]
            x = int(landmark.x * frame_w)
            y = int(landmark.y * frame_h)

            eye_anchor_x += int(x / len(eye_anchor))
            eye_anchor_y += int(y / len(eye_anchor))
            cv2.circle(frame, (x, y), 3, (255, 0, 255))
        cv2.circle(frame, (eye_anchor_x, eye_anchor_y), 3, (255, 0, 0))

        # -------------------------------
        # Left eye tracking for blinking
        # -------------------------------
        left = [landmarks[145], landmarks[159]]
        for landmark in left:
            x = int(landmark.x * frame_w)
            y = int(landmark.y * frame_h)
            cv2.circle(frame, (x, y), 3, (0, 255, 255))
        if (left[0].y - left[1].y) < 0.004:
            pyautogui.click()
            pyautogui.sleep(1)
        
        # -------------------------------
        # Create vector between pupil and anchor
        # -------------------------------
        pupil_x, pupil_y = 0, 0
        for id in right_eyeball:
            landmark = landmarks[id]
            pupil_x += landmark.x
            pupil_y += landmark.y
        
        pupil_x /= len(right_eyeball)
        pupil_y /= len(right_eyeball)
        pupil_center_x = int(pupil_x * frame_w)
        pupil_center_y = int(pupil_y * frame_h)

        # pupil center
        cv2.circle(frame, (pupil_center_x, pupil_center_y), 3, (255, 0, 0), -1)

        # vector
        cv2.line(frame, (eye_anchor_x, eye_anchor_y), (pupil_center_x, pupil_center_y), (0, 0, 255), 2)

        set_mouse_pos((eye_anchor_x, eye_anchor_y), (pupil_center_x, pupil_center_y))
    cv2.imshow('Eye Controlled Mouse', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Add a way to break the loop and close the window
        break
