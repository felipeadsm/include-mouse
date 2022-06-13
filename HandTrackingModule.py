import cv2
import mediapipe as mp
import time


class hand_detector:
    def __init__(self, mode=False, max_hands=2, detection_con=0.5, model_complexity=1,  track_con=0.5):
        self.mode = mode
        self.maxHands = max_hands
        self.model_complexity = model_complexity
        self.detectionCon = detection_con
        self.trackCon = track_con
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.model_complexity,
                                        self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
        self.results = 0

    def find_hands(self, img, draw=True):
        image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(image_rgb)
        # print(results.multi_hand_landmarks)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms,
                                               self.mpHands.HAND_CONNECTIONS)
        return img

    def find_position(self, img, hand_no=0, draw=True, color=(255, 0, 255), z_axis=False):

        lm_list = []
        if self.results.multi_hand_landmarks:
            my_hand = self.results.multi_hand_landmarks[hand_no]
            for id_s, lm in enumerate(my_hand.landmark):
                #   print(id_s, lm)
                h, w, c = img.shape
                if not z_axis:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    # print(id_s, cx, cy)
                    lm_list.append([id_s, cx, cy])
                elif z_axis:
                    cx, cy, cz = int(lm.x * w), int(lm.y * h), round(lm.z, 3)
                    # print(id_s, cx, cy, cz)
                    lm_list.append([id_s, cx, cy, cz])

                if draw:
                    cv2.circle(img, (cx, cy), 5, color, cv2.FILLED)

        return lm_list


def main():
    p_time = 0
    c_time = 0
    cap = cv2.VideoCapture(0)
    detector = hand_detector(max_hands=1)
    while True:
        success, img = cap.read()
        img = detector.find_hands(img)
        lm_list = detector.find_position(img, draw=False, z_axis=True)
        if len(lm_list) != 0:
            print(lm_list[4])

        c_time = time.time()
        fps = 1 / (c_time - p_time)
        p_time = c_time

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                    (255, 0, 255), 3)

        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == "__main__":
    main()
