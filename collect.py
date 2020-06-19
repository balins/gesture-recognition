import copy
import cv2
import os

from imutils.video import WebcamVideoStream

accent_color = (0, 255, 0)  # RED
FONT = cv2.FONT_HERSHEY_SIMPLEX
fx, fy, fh = 10, 50, 45
taking_data = False
classname = 'NONE'
count = 0
size = 300

classes = 'NONE OPEN_HAND THUMBS_UP POINTING PEACE OKAY SHAKA'.split()


def init_class(name):
    global classname, count
    classname = name
    os.system('mkdir -p data/train/%s' % name)
    count = len(os.listdir('data/train/%s' % name))


def binary_mask(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img, (5, 5), 3)
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    _, mask = cv2.threshold(img, 25, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return mask


def main():
    global FONT, fx, fy, fh, size
    global taking_data, accent_color
    global classname, count

    x0, y0 = 100, 100

    cam = WebcamVideoStream(src=0).start()
    cv2.namedWindow('Capture', cv2.WINDOW_NORMAL)

    while True:
        frame = cam.read()
        frame = cv2.flip(frame, 1)
        window = copy.deepcopy(frame)
        cv2.rectangle(window, (x0, y0), (x0 + size, y0 + size), accent_color, 12)

        # draw text
        if taking_data:
            accent_color = (0, 0, 255)
            cv2.putText(window, 'Data Taking: ON', (fx, fy), FONT, 1.2, accent_color, 2, 1)
        else:
            accent_color = (255, 0, 0)
            cv2.putText(window, 'Data Taking: OFF', (fx, fy), FONT, 1.2, accent_color, 2, 1)

        # get region of interest
        roi = frame[y0:y0 + size, x0:x0 + size]
        roi = binary_mask(roi)

        # apply processed roi in frame
        window[y0:y0 + size, x0:x0 + size] = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)

        # take data
        if taking_data:
            cv2.putText(window, 'Class: %s (%d)' % (classname, count),
                        (fx, fy + fh), FONT, 1.0, accent_color, 2, 1)
            roi = cv2.resize(roi, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
            cv2.imwrite('data/train/{0}/{0}_{1}.png'.format(classname, count), roi)
            count += 1

        # show the window
        cv2.imshow('Capture', window)

        # Keyboard inputs
        key = cv2.waitKey(10) & 0xff

        # use q key to close the program
        if key == ord('q'):
            break

        # Toggle data taking
        elif key == ord('s'):
            taking_data = not taking_data

        # Toggle class
        elif key == ord('0'):
            init_class('NONE')
        elif key == ord('1'):
            init_class('OPEN_HAND')
        elif key == ord('2'):
            init_class('THUMBS_UP')
        elif key == ord('3'):
            init_class('POINTING')
        elif key == ord('4'):
            init_class('PEACE')
        elif key == ord('5'):
            init_class('OKAY')
        elif key == ord('6'):
            init_class('SHAKA')

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
    init_class('NONE')
    main()
