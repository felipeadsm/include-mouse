import autopy
import math
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


def suavize_move(current_point, future_point):
    current_point = current_point[0] * current_point[1]
    future_point = future_point[0] * future_point[1]

    diference_between_points = future_point - current_point
    abs_between = abs(diference_between_points)
    b = 100
    expression = ((abs_between * math.e + b - abs_between)/b)
    out_point = current_point + round(diference_between_points * math.log(expression))

    print(round(out_point))

    return round(out_point)


# TODO: Receber os valores do retângulo em volta dos olhos
def move_mouse(frame, initial_point, final_point, c_left):
    # Draw a reference rectangle
    cv2.rectangle(frame, initial_point, final_point, (255, 255, 255), 1)

    # Take a center of eye
    x1, y1 = c_left[0], c_left[1]

    # Get the screen size
    w, h = autopy.screen.size()

    # Interpolates between the reference square and the screen
    X = int(np.interp(x1, [initial_point[0], final_point[0]], [0, w - 1]))
    Y = int(np.interp(y1, [initial_point[1], final_point[1]], [0, h - 1]))

    # # TODO: Função de transferência
    # if X % 2 != 0:
    #     X = X - X % 2
    # if Y % 2 != 0:
    #     Y = Y - Y % 2

    suavize_move(autopy.mouse.location(), (X, Y))

    # Move the cursor for the interpolation position
    autopy.mouse.move(X, Y)


# Main program
mp_face_mesh = mp.solutions.face_mesh

# Right Eyes Points
RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_IRIS = [474, 475, 476, 477]

R_H_LEFT = [33]  # index 0
R_H_RIGHT = [133]  # index 8
RIGHT_HORIZONTAL_LIST = [R_H_LEFT, R_H_RIGHT]

R_V_TOP_1 = [159]  # index 12
R_V_BOT_1 = [145]  # index 4
R_V_TOP_2 = [158]  # index 11
R_V_BOT_2 = [153]  # index 5
RIGHT_VERTICAL_LIST = [R_V_TOP_1, R_V_BOT_1, R_V_TOP_2, R_V_TOP_2]


# Left Eyes Points
LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
LEFT_IRIS = [469, 470, 471, 472]

L_H_LEFT = [362]  # index 0
L_H_RIGHT = [263]  # index 8
LEFT_HORIZONTAL_LIST = [L_H_LEFT, L_H_RIGHT]

L_V_TOP_1 = [386]  # index 12
L_V_BOT_1 = [374]  # index 4
L_V_TOP_2 = [87]  # index 11
L_V_BOT_2 = [373]  # index 5
LEFT_VERTICAL_LIST = [L_V_TOP_1, L_V_BOT_1, L_V_TOP_2, L_V_TOP_2]

# Variables
i = 0
blink_thresh = 0.35
blink_frame = 4
count_frame = 0

camera = cv2.VideoCapture(0)

with mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5,
                           min_tracking_confidence=0.5) as face_mesh:
    while True:
        ret, frame = camera.read()
        frame = cv2.flip(frame, 1)

        if not ret:
            break

        # Code block to increase camera size
        # Percent of original size
        scale_percent = 200
        width = int(frame.shape[1] * scale_percent / 100)
        height = int(frame.shape[0] * scale_percent / 100)
        dim = (width, height)
        frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

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

            # Iris Detection: Circle
            cv2.circle(frame, center_left, int(l_radius), (255, 0, 255), 1, cv2.LINE_AA)
            cv2.circle(frame, center_right, int(r_radius), (255, 0, 255), 1, cv2.LINE_AA)

            # Create a line betwen two eyes
            cv2.line(frame, center_left, center_right, (0, 0, 0), 1, cv2.LINE_AA)

            # Calculates the midpoint between the two eyesCalculates a midpoint between the two eyes
            diff_eyes = ((center_right[0] - center_left[0]), (center_right[1] - center_left[1]))
            center_eyes = ((center_left[0] + diff_eyes[0] / 2), (center_left[1] + diff_eyes[1] / 2))

            # TODO: Automatizar essa função criando o retângulo dependendo do tamanho da tela
            # Sets the initial and final coordinate for the reference rectangle
            pi = (512, 384)
            pf = (768, 576)

            # Call the move mouse function
            move_mouse(frame, pi, pf, center_eyes)

            # Call the blink eye function
            # Calculate the EAR
            left_ear = calculate_ear(mesh_points[LEFT_EYE])
            right_ear = calculate_ear(mesh_points[RIGHT_EYE])

            # Avg of left and right eye EAR
            avg_ear_eyes = (left_ear + right_ear) / 2

            # print(avg_ear_eyes)

            if avg_ear_eyes < blink_thresh:
                # Incrementing the frame count
                count_frame += 1
            else:
                if count_frame >= blink_frame:
                    cv2.putText(frame, 'Blink Detected', (30, 30), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 200, 0), 1)
                    autopy.mouse.click()
                    count_frame = 0

        cv2.imshow('image', frame)
        key = cv2.waitKey(1)

        if key == ord('q'):
            break

camera.release()
cv2.destroyAllWindows()
