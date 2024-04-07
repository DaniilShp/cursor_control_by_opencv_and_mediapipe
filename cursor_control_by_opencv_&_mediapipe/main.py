import cv2
import mediapipe as mp
import time
import pyautogui
from pynput.mouse import Controller, Button
from typing import NamedTuple

_RMV_RATTLING_CNST = 0.01 # константа для удаления дребезга
_VID_DEAD_ZONE_CNST = 0.2 #80% области относительно центра влияет на передвижение курсора

count = 0
class FpsCounter:
    def __init__(self):
        self.fps = 0
        self.frames_count = 0
        self.period = 1 #соответствует частоте обновления 1 герц
        self.start_time = time.time()
    def start(self):
        self.start_time = time.time()
    def set_frequency(self, frequency_Hz): #default frequency_Hz = 1
        self.period = 1 / frequency_Hz
    def update(self): #пришел новый кадр => актуализируем данные
        self.frames_count += 1
        current_time = time.time()
        if current_time - self.start_time >= self.period: #происходит подсчет кадров за каждую последнюю секунду
            self.fps = round(self.frames_count / (time.time() - self.start_time), 2) #обновление показателя fps раз в 1 сек.
            self.frames_count = 0
            self.start_time = time.time()
        return self.fps
    def get_fps(self):
        return self.fps
    def get_frames_count(self):
        return self.frames_count

class GestureInterpreter:
    __GESTURE_NO = -1
    __GESTURE_OK = 0
    __GESTURE_PALM = 1
    __GESTURE_OTHER = 2
    def __init__(self):
        self.count = 0
        self.status = self.__GESTURE_NO
        self.history_length = 10
        self.gestures_history = [self.status] * self.history_length
        screen_width, screen_height = pyautogui.size()
        self.mouse_x, self.mouse_y = screen_width // 2, screen_height // 2
        self.mouse = Controller()
    def __points_distance(self, A, B):
        return abs(((B[1]-A[1])**2 + (B[0]-A[0])**2)**(1 / 2))
    def __update_history(self, new_info):
        self.gestures_history.pop(0)
        self.gestures_history.append(new_info)
    def detect_gesture(self, hand: NamedTuple):
        if hand is None or hand.multi_hand_landmarks is None or len(hand.multi_hand_landmarks) == 0:
            return 0, 0
        index_finger_top = [hand.multi_hand_landmarks[0].landmark[8].x, hand.multi_hand_landmarks[0].landmark[8].y]
        thumb_finger_top = [hand.multi_hand_landmarks[0].landmark[4].x, hand.multi_hand_landmarks[0].landmark[4].y]
        index_finger_mcp = [hand.multi_hand_landmarks[0].landmark[5].x, hand.multi_hand_landmarks[0].landmark[5].y]
        pinky_mcp = [hand.multi_hand_landmarks[0].landmark[17].x, hand.multi_hand_landmarks[0].landmark[17].y]
        wrist = [hand.multi_hand_landmarks[0].landmark[0].x, hand.multi_hand_landmarks[0].landmark[0].y]
        palm_width = self.__points_distance(index_finger_mcp, pinky_mcp)
        thumb_length = self.__points_distance(thumb_finger_top, wrist)
        if self.__points_distance(index_finger_top, thumb_finger_top) * 6 < max(palm_width, thumb_length): #много меньше
            self.__update_history(self.__GESTURE_OK)
            self.count += 1
        else:
            self.__update_history(self.__GESTURE_PALM)
        return index_finger_top[0], index_finger_top[1]
    def interpret_gesture(self, x, y):
        for i in range(self.history_length-1, -1, -1):
            flag = 1
            gesture_palm_required_num = gesture_ok_required_num = 2
            for p in range(1, gesture_palm_required_num+1):
                if self.gestures_history[i-p] is not self.__GESTURE_PALM:
                    flag = 0
                    break
            for o in range(1, gesture_ok_required_num+1):
                if self.gestures_history[i-o-gesture_palm_required_num] is not self.__GESTURE_OK:
                    flag = 0
                    break
            if flag == 1:
                print("break")
                print(self.gestures_history)
                self.mouse.click(Button.left, 1)
            break


def get_mouse_coord_offset(x, y, w, h): #вычисление смещения мыши в зависимости от удалаенности управляющего жеста от центра изображения с камеры
    if x > (1-_VID_DEAD_ZONE_CNST) * w:
        x = int((1-_VID_DEAD_ZONE_CNST) * w)
    if y > (1-_VID_DEAD_ZONE_CNST) * h:
        y = int((1-_VID_DEAD_ZONE_CNST) * h)
    center_x, center_y = w // 2, h // 2
    offset_x, offset_y = - int((center_x - x) / 2), - int((center_y - y) / 2)
    exponent = lambda t: 1/1200000 * t ** 3 - 3/4000 * t ** 2 + 19/60 * t #кубическая функция зависимости скорости изменения координат курсора от координат управляющего жеста относительно камеры
    offset_x, offset_y = exponent(offset_x), exponent(offset_y)
    if abs(offset_x) < _RMV_RATTLING_CNST * w: #remove rattling - удаление дребезга
        offset_x = 0
    if abs(offset_y) < _RMV_RATTLING_CNST * h:
        offset_y = 0
    return offset_x, offset_y

if __name__ == "__main__":
    interpreter = GestureInterpreter()
    cap = cv2.VideoCapture(0)  #Загрузка видео или создание объекта VideoCapture для камеры
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
    hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=1,
                                     min_tracking_confidence=0.5, min_detection_confidence=0.5)
    mp_draw = mp.solutions.drawing_utils
    screen_width, screen_height = pyautogui.size()
    pyautogui.FAILSAFE = False
    mouse_x, mouse_y = screen_width // 2, screen_height // 2
    mouse = Controller()
    pyautogui.moveTo(mouse_x, mouse_y)

    frame_monitor1 = FpsCounter()
    frame_monitor1.start()

    while cap.isOpened():
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        if not ret:
            break
        result = hands.process(frame)

        if result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, result.multi_hand_landmarks[0], mp.solutions.hands.HAND_CONNECTIONS)
            x, y = interpreter.detect_gesture(result)
            offset_x, offset_y = get_mouse_coord_offset(x*screen_width, y*screen_height, screen_width, screen_height)
            if frame_monitor1.get_frames_count() % 3 == 0:  # передвижение курсора каждые 3 кадра
                mouse.move(offset_x, offset_y)
            interpreter.interpret_gesture(x, y)

        cv2.putText(frame, text=f'fps count: {frame_monitor1.get_fps()}',
                    org=(40, 40), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                    color=(0, 0, 0), thickness=1)
        cv2.imshow('hand detecting', frame) # Отображение кадра
        frame_monitor1.update()

        if cv2.waitKey(1) & 0xFF == ord('q'): # Выход из цикла, если нажата клавиша 'q'
            break

    cap.release()
    cv2.destroyAllWindows() # Освобождение ресурсов и закрытие окон
