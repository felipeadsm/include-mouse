import cv2 as cv2
import numpy as np
import mediapipe as mp
import math

mp_face_mesh = mp.solutions.face_mesh

# Right Eyes Points
RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_IRIS = [474, 475, 476, 477]
R_H_LEFT = [362]
R_H_RIGHT = [263]

# Left Eyes Points
LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
LEFT_IRIS = [469, 470, 471, 472]
L_H_LEFT = [33]
L_H_RIGHT = [133]


def euclidian_distance(point_one, point_two):
    x1, y1 = point_one.ravel()
    x2, y2 = point_two.ravel()
    distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return distance


def iris_position(iris_center, right_point, left_point, eye):
    position_iris = ''
    ratio = None

    center_to_right_distance = euclidian_distance(iris_center, right_point)
    center_to_left_distance = euclidian_distance(iris_center, left_point)
    total_distance = euclidian_distance(right_point, left_point)

    if eye == 'left':
        ratio = center_to_left_distance / total_distance
    elif eye == 'right':
        ratio = center_to_right_distance / total_distance

    if ratio <= 0.42:
        position_iris = 'right'
    elif 0.42 < ratio <= 0.57:
        position_iris = 'center'
    elif ratio > 0.57:
        position_iris = 'left'

    return position_iris, ratio


camera = cv2.VideoCapture(0)

with mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5,
                           min_tracking_confidence=0.5) as face_mesh:
    while True:
        ret, frame = camera.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_h, image_w = frame.shape[:2]
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            mesh_points = np.array([np.multiply([p.x, p.y], [image_w, image_h]).astype(int) for p in
                                    results.multi_face_landmarks[0].landmark])

            # # Eyes Detection
            # cv2.polylines(frame, [mesh_points[LEFT_EYE]], True, (255, 0, 255), 1, cv2.LINE_AA)
            # cv2.polylines(frame, [mesh_points[RIGHT_EYE]], True, (255, 0, 255), 1, cv2.LINE_AA)
            #
            # # Iris Detection: Rectangle
            # cv2.polylines(frame, [mesh_points[LEFT_IRIS]], True, (255, 0, 255), 1, cv2.LINE_AA)
            # cv2.polylines(frame, [mesh_points[RIGHT_IRIS]], True, (255, 0, 255), 1, cv2.LINE_AA)

            # Iris Detection: Create Circle
            (l_cx, l_cy), l_radius = cv2.minEnclosingCircle(mesh_points[LEFT_IRIS])
            (r_cx, r_cy), r_radius = cv2.minEnclosingCircle(mesh_points[RIGHT_IRIS])
            center_left = np.array([l_cx, l_cy], dtype=np.int32)
            center_right = np.array([r_cx, r_cy], dtype=np.int32)

            # Iris Detection: Circle
            cv2.circle(frame, center_left, int(l_radius), (255, 0, 255), 1, cv2.LINE_AA)
            cv2.circle(frame, center_right, int(r_radius), (255, 0, 255), 1, cv2.LINE_AA)

            # Iris Detection: Right Points
            cv2.circle(frame, mesh_points[R_H_RIGHT][0], 3, (255, 0, 255), -1, cv2.LINE_AA)
            cv2.circle(frame, mesh_points[R_H_LEFT][0], 3, (0, 255, 255), -1, cv2.LINE_AA)

            # Iris Detection: Left Points
            cv2.circle(frame, mesh_points[L_H_RIGHT][0], 3, (0, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(frame, mesh_points[L_H_LEFT][0], 3, (255, 0, 255), -1, cv2.LINE_AA)

            # Iris Detection: Position
            left_iris_pos, ratio_l = iris_position(center_left, mesh_points[L_H_LEFT],
                                                   mesh_points[L_H_RIGHT][0], 'left')
            right_iris_pos, ratio_r = iris_position(center_right, mesh_points[R_H_RIGHT],
                                                    mesh_points[R_H_LEFT][0], 'right')

            # Iris Detection: Result With Detection
            cv2.putText(frame, 'Left Iris Position: {} | Right Iris Position {}'.format(left_iris_pos, right_iris_pos),
                        (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1, cv2.LINE_AA)

        cv2.imshow('image', frame)
        key = cv2.waitKey(1)

        if key == ord('q'):
            break

camera.release()
cv2.destroyAllWindows()
