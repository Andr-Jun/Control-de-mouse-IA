import cv2
import mediapipe as mp
import numpy as np
import pyautogui

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

color_mouse_pointer = (253, 255, 0)

# Puntos de la pantalla
SCREEN_X_INI = 0
SCREEN_Y_INI = 0
SCREEN_X_FIN = 0 + 2560
SCREEN_Y_FIN = 0 + 1600

X_Y_INI = 0

def calcular_distancias(x1, y1, x2, y2):
    p1 = np.array([x1, y1])
    p2 = np.array([x2, y2])
    return np.linalg.norm(p1 - p2)


def detectar_indice_abajo(hand_landmarks):
    indice_abajo = False
    color_base = (255, 113, 113)
    color_dedos = (255, 255, 255)

    x_base1 = int(hand_landmarks.landmark[0].x * width)
    y_base1 = int(hand_landmarks.landmark[0].y * height)

    x_base2 = int(hand_landmarks.landmark[9].x * width)
    y_base2 = int(hand_landmarks.landmark[9].y * height)

    x_indice = int(hand_landmarks.landmark[8].x * width)
    y_indice = int(hand_landmarks.landmark[8].y * height)

    x_medio = int(hand_landmarks.landmark[12].x * width)
    y_medio = int(hand_landmarks.landmark[12].y * height)

    d_base = calcular_distancias(x_base1, y_base1, x_base2, y_base2)
    d_indice = calcular_distancias(x_base1, y_base1, x_indice, y_indice)

    cv2.circle(frame, (x_medio, y_medio), 5, color_dedos, 2)
    cv2.circle(frame, (x_base1, y_base1), 5, color_dedos, 2)
    cv2.circle(frame, (x_indice, y_indice), 5, color_dedos, 2)
    cv2.line(frame, (x_base1, y_base1), (x_medio, y_medio), color_dedos, 3)
    cv2.line(frame, (x_base1, y_base1), (x_indice, y_indice), color_dedos, 3)
    cv2.line(frame, (x_base1, y_base1), (x_base2, y_base2), color_base, 3)

    if d_indice < d_base:
        indice_abajo = True
        color_base = (255, 0, 255)
        color_dedos = (255, 0, 255)
        
    cv2.circle(frame, (x_base1, y_base1), 5, color_base, 2)
    cv2.circle(frame, (x_indice, y_indice), 5, color_dedos, 2)
    cv2.line(frame, (x_base1, y_base1), (x_base2, y_base2), color_base, 3)
    cv2.line(frame, (x_base1, y_base1), (x_indice, y_indice), color_dedos, 3)

    return indice_abajo


def detectar_medio_abajo(hand_landmarks):
     medio_abajo = False
     color_base = (255, 113, 113)
     color_dedos = (255, 255, 255)

     x_base1 = int(hand_landmarks.landmark[0].x * width)
     y_base1 = int(hand_landmarks.landmark[0].y * height)

     x_base2 = int(hand_landmarks.landmark[9].x * width)
     y_base2 = int(hand_landmarks.landmark[9].y * height)

     x_indice = int(hand_landmarks.landmark[8].x * width)
     y_indice = int(hand_landmarks.landmark[8].y * height)

     x_medio = int(hand_landmarks.landmark[12].x * width)
     y_medio = int(hand_landmarks.landmark[12].y * height)

     d_base = calcular_distancias(x_base1, y_base1, x_base2, y_base2)
     d_medio = calcular_distancias(x_base1, y_base1, x_medio, y_medio)

     cv2.circle(frame, (x_medio, y_medio), 5, color_dedos, 2)
     cv2.circle(frame, (x_base1, y_base1), 5, color_dedos, 2)
     cv2.circle(frame, (x_indice, y_indice), 5, color_dedos, 2)
     cv2.line(frame, (x_base1, y_base1), (x_medio, y_medio), color_dedos, 3)
     cv2.line(frame, (x_base1, y_base1), (x_indice, y_indice), color_dedos, 3)
     cv2.line(frame, (x_base1, y_base1), (x_base2, y_base2), color_base, 3)

     if d_medio < d_base:
          medio_abajo = True
          color_base = (255, 0, 255)
          color_dedos = (255, 0, 255)

     cv2.circle(frame, (x_base1, y_base1), 5, color_base, 2)
     cv2.circle(frame, (x_medio, y_medio), 5, color_dedos, 2)
     cv2.line(frame, (x_base1, y_base1), (x_base2, y_base2), color_base, 3)
     cv2.line(frame, (x_base1, y_base1), (x_medio, y_medio), color_dedos, 3)
    
     return medio_abajo


with mp_hands.Hands(
    
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.9) as hands:

    while True:
        ret, frame = cap.read()
        if ret == False:
            break

        height, width, _ = frame.shape

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(frame_rgb)
        
        if results.multi_hand_landmarks is not None:
            for hand_landmarks in results.multi_hand_landmarks:
                x = int(hand_landmarks.landmark[9].x * width)
                y = int(hand_landmarks.landmark[9].y * height)

                xm = np.interp(x, (X_Y_INI, X_Y_INI + 1200), (SCREEN_X_INI, SCREEN_X_FIN))
                ym = np.interp(y, (X_Y_INI, X_Y_INI + 700), (SCREEN_Y_INI, SCREEN_Y_FIN))

                cv2.circle(frame, (x, y), 10, color_mouse_pointer, 3)
                cv2.circle(frame, (x, y), 5, color_mouse_pointer, -1)

                pyautogui.moveTo(int(xm), int(ym))

                if detectar_indice_abajo(hand_landmarks):
                    pyautogui.click()
        
                if detectar_medio_abajo(hand_landmarks):
                     pyautogui.rightClick()
                    
        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
