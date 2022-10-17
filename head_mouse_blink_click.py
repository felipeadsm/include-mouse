import autopy
import cv2 as cv2
import numpy as np
import mediapipe as mp
from scipy.spatial import distance as dist


# Defining a function to calculate the Eye Aspect Ratio (EAR)
def calculate_ear(eye):
    # Calculate the vertical distances
    y1 = dist.euclidean(eye[12], eye[4])
    y2 = dist.euclidean(eye[11], eye[5])

    # Calculate the horizontal distance
    x1 = dist.euclidean(eye[0], eye[8])

    # Calculate the  (EAR)
    EAR = (y1 + y2) / x1
    return EAR


def smooth_move(x, y, x_current_point, y_current_point):
    # It is interesting to vary the value of A to obtain a faster movement
    A = 0.001

    x_inc = A * (x - x_current_point) ** 2 * np.sign(x - x_current_point)
    y_inc = A * (y - y_current_point) ** 2 * np.sign(y - y_current_point)

    x_final = x_inc + x_current_point
    y_final = y_inc + y_current_point

    return x_final, y_final


def move_mouse(img, initial_point, final_point, c_left):
    # Draw a reference rectangle
    cv2.rectangle(img, initial_point, final_point, (255, 255, 255), 1)

    # Take a center of eye
    x1, y1 = c_left[0], c_left[1]

    # Get the screen size
    w, h = autopy.screen.size()

    # Interpolates between the reference square and the screen
    X = int(np.interp(x1, [initial_point[0], final_point[0]], [0, w - 1]))
    Y = int(np.interp(y1, [initial_point[1], final_point[1]], [0, h - 1]))

    current_point = autopy.mouse.location()

    # Call smoothing function
    x_move, y_move = smooth_move(X, Y, current_point[0], current_point[1])

    try:
        autopy.mouse.move(x_move, y_move)
    except Exception as e:
        print(e)


# Main program
mp_face_mesh = mp.solutions.face_mesh

# Right Eyes Points
RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_IRIS = [474, 475, 476, 477]

# Left Eyes Points
LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
LEFT_IRIS = [469, 470, 471, 472]

# Variables
i = 0
blink_thresh = 0.35
blink_frame = 4
count_frame = 0

camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

with mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5,
                           min_tracking_confidence=0.5) as face_mesh:
    while camera.isOpened():
        ret, frame = camera.read()
        frame = cv2.flip(frame, 1)

        if not ret:
            break

        # Code block to increase camera size
        # Percent of original size
        width = frame.shape[1]
        height = frame.shape[0]

        # Convert the bgr frame to rgb
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Takes a heigth and width of the image
        image_h, image_w = frame.shape[:2]
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            mesh_points = np.array([np.multiply([p.x, p.y], [image_w, image_h]).astype(int) for p in
                                    results.multi_face_landmarks[0].landmark])

            # Iris Detection: Create Circle
            (l_cx, l_cy), l_radius = cv2.minEnclosingCircle(mesh_points[LEFT_IRIS])
            (r_cx, r_cy), r_radius = cv2.minEnclosingCircle(mesh_points[RIGHT_IRIS])

            # Calculate the center of both eyes using numpy
            center_left = np.array([l_cx, l_cy], dtype=np.int32)
            center_right = np.array([r_cx, r_cy], dtype=np.int32)

            # Calculates the midpoint between the two eyesCalculates a midpoint between the two eyes
            diff_eyes = ((center_right[0] - center_left[0]), (center_right[1] - center_left[1]))
            center_eyes = ((center_left[0] + diff_eyes[0] / 2), (center_left[1] + diff_eyes[1] / 2))

            # Sets the initial and final coordinate for the reference rectangle
            pi = (256, 192)
            pf = (384, 288)

            # Call the move mouse function
            move_mouse(frame, pi, pf, center_eyes)

            # Call the blink eye function
            # Calculate the EAR
            left_ear = calculate_ear(mesh_points[LEFT_EYE])
            right_ear = calculate_ear(mesh_points[RIGHT_EYE])

            # Average of left and right eye EAR
            avg_ear_eyes = (left_ear + right_ear) / 2

            # Check if the average EAR is less than the threshold, if not, the click event is fired
            if avg_ear_eyes < blink_thresh:
                # Incrementing the frame count
                count_frame += 1
            else:
                if count_frame >= blink_frame:
                    autopy.mouse.click()
                    count_frame = 0

        cv2.imshow('image', frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

camera.release()
cv2.destroyAllWindows()
