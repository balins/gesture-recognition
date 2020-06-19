import copy
import cv2
import numpy as np
from imutils.video import WebcamVideoStream
from keras.engine.saving import load_model
import pyautogui

pyautogui.FAILSAFE = False
accent_color = (0, 255, 0)  # RED
FONT = cv2.FONT_HERSHEY_SIMPLEX
fx, fy, fh = 10, 10, 45
classname = 'NONE'
classes = 'NONE OKAY OPEN_HAND PEACE POINTING SHAKA THUMBS_UP'.split()


def binary_mask(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img, (5, 5), 3)
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    _, mask = cv2.threshold(img, 25, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return mask


UNIT = 20

actions = {'NONE': pyautogui.move, 'OKAY': pyautogui.move, 'OPEN_HAND': pyautogui.move,
           'PEACE': pyautogui.move, 'POINTING': pyautogui.move,
           'SHAKA': pyautogui.click, 'THUMBS_UP': pyautogui.click}

args = {'NONE': (0, 0), 'OKAY': (-UNIT, 0), 'OPEN_HAND': (0, -UNIT),
        'PEACE': (UNIT, 0), 'POINTING': (0, UNIT),
        'SHAKA': (), 'THUMBS_UP': ()}


def main():
    global FONT, fx, fy, fh, accent_color
    model = load_model('models/final.h5')
    x0, y0, size = 200, 180, 300

    cam = WebcamVideoStream(src=0).start()
    cv2.namedWindow('Capture', cv2.WINDOW_NORMAL)

    while True:
        frame = cam.read()
        frame = cv2.flip(frame, 1)
        window = copy.deepcopy(frame)
        cv2.rectangle(window, (x0, y0), (x0 + size, y0 + size), accent_color, 12)

        # get region of interest
        roi = frame[y0:y0 + size, x0:x0 + size]
        roi = binary_mask(roi)

        # apply processed roi in frame
        window[y0:y0 + size, x0:x0 + size] = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)

        img = cv2.resize(roi, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)

        img = np.float32(img) / 255.
        img = np.expand_dims(img, axis=0)
        img = np.expand_dims(img, axis=-1)
        predictions = model.predict(img)
        if predictions.any() > 0.9:
            pred = classes[predictions.argmax()]
            actions[pred](*args[pred])
            cv2.putText(window, 'Prediction: %s' % pred, (fx, fy + fh), FONT, 1.0, (245, 210, 65), 2, 1)
        else:
            cv2.putText(window, 'No gesture found', (fx, fy + fh), FONT, 1.0, (245, 210, 65), 2, 1)

        # show the window
        cv2.imshow('Capture', window)

        # Keyboard inputs
        key = cv2.waitKey(120) & 0xff

        # use q key to close the program
        if key == ord('q'):
            break

        # adjust the position of window
        elif key == ord('i'):
            y0 = max((y0 - 5, 0))
        elif key == ord('k'):
            y0 = min((y0 + 5, window.shape[0] - size))
        elif key == ord('j'):
            x0 = max((x0 - 5, 0))
        elif key == ord('l'):
            x0 = min((x0 + 5, window.shape[1] - size))

    cv2.destroyAllWindows()
    cam.stop()


if __name__ == '__main__':
    main()
