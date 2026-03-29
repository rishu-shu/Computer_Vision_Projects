import cv2
import numpy as np
import time
import sys
from pathlib import Path

# ===== PATH SETUP =====
SCRIPT_DIR = Path(__file__).resolve().parent
img_path = SCRIPT_DIR / "test.jpg"

output_dir = SCRIPT_DIR / "outputs"
output_dir.mkdir(parents=True, exist_ok=True)

# ===== LOAD IMAGE =====
img = cv2.imread(str(img_path))

if img is None:
    print("Image not found")
    sys.exit(1)

img = cv2.resize(img, (500, 500))
canvas = img.copy()
orig = img.copy()  # keep a copy of the original resized image for eraser restore

# ===== SETTINGS =====
drawing = False
color = (0, 255, 0)
brush_size = 3
eraser = False

# ===== HISTORY =====
history = [canvas.copy()]
redo_stack = []

def save_state():
    history.append(canvas.copy())
    redo_stack.clear()

def undo():
    global canvas
    if len(history) > 1:
        redo_stack.append(history.pop())
        canvas[:] = history[-1]

def redo():
    global canvas
    if redo_stack:
        canvas[:] = redo_stack.pop()
        history.append(canvas.copy())

# ===== DRAW FUNCTION =====
def draw(event, x, y, flags, param):
    global drawing, canvas, eraser, brush_size, color, orig

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        save_state()

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            if eraser:
                # restore pixels from original image within the brush circle
                mask = np.zeros(canvas.shape[:2], dtype=np.uint8)
                cv2.circle(mask, (x, y), brush_size, 255, -1)
                canvas[mask == 255] = orig[mask == 255]
            else:
                cv2.circle(canvas, (x, y), brush_size, color, -1)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False

cv2.namedWindow("APP")
cv2.setMouseCallback("APP", draw)

# ===== MAIN LOOP =====
while True:
    display = canvas.copy()

    # ===== UI TEXT =====
    cv2.putText(display, "g-gray b-blur e-edge t-thresh r-rotate", (10,20),cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255),1)
    cv2.putText(display, "m-dilate n-erode j-open k-close", (10,40),cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255),1)
    cv2.putText(display, "R/G/B-color x-eraser +/- brush", (10,60),cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255),1)
    cv2.putText(display, "u-undo y-redo c-clear s-save q-quit", (10,80),cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255),1)
    cv2.putText(display, "1-line 2-rectangle 3-circle", (10,100),cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255),1)


    cv2.imshow("APP", display)
    key = cv2.waitKey(1) & 0xFF

    # ===== BASIC FILTERS =====
    # GRAYSCALE
    if key == ord("g"):
        gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        canvas = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        save_state()

    # IMAGE ROTATION
    elif key == ord("r"):
        h, w = canvas.shape[:2]
        matrix = cv2.getRotationMatrix2D((w//2, h//2), 45, 1)
        canvas = cv2.warpAffine(canvas, matrix, (w, h))
        save_state()

    # BLUR
    elif key == ord("b"):
        canvas = cv2.GaussianBlur(canvas, (15,15), 0)
        save_state()

    # EDGE DETECTION
    elif key == ord("e"):
        gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        edge = cv2.Canny(gray, 100, 200)
        canvas = cv2.cvtColor(edge, cv2.COLOR_GRAY2BGR)
        save_state()

    # IMAGE THRESHOLDING
    elif key == ord("t"):
        gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        canvas = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
        save_state()

    # ===== MORPHOLOGY =====
    # DILUTION
    elif key == ord("m"):
        gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        kernel = np.ones((3,3), np.uint8)
        canvas = cv2.cvtColor(cv2.dilate(edges, kernel), cv2.COLOR_GRAY2BGR)
        save_state()

    # EROSION
    elif key == ord("n"):
        gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        kernel = np.ones((3,3), np.uint8)
        canvas = cv2.cvtColor(cv2.erode(edges, kernel), cv2.COLOR_GRAY2BGR)
        save_state()

    # OPENING
    elif key == ord("j"):
        gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        kernel = np.ones((3,3), np.uint8)
        canvas = cv2.cvtColor(cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel), cv2.COLOR_GRAY2BGR)
        save_state()

    # CLOSING
    elif key == ord("k"):
        gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        kernel = np.ones((3,3), np.uint8)
        canvas = cv2.cvtColor(cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel), cv2.COLOR_GRAY2BGR)
        save_state()

    # BRIGHTNESS
    elif key == ord("l"):
        canvas = cv2.convertScaleAbs(canvas, alpha=1, beta=50)
        save_state()

    elif key == ord("d"):
        canvas = cv2.convertScaleAbs(canvas, alpha=1, beta=-50)
        save_state()
      
    # CONTRAST
    elif key == ord("h"):
        canvas = cv2.convertScaleAbs(canvas, alpha=1.5, beta=0)
        save_state()

    # HSV
    elif key == ord("v"):
        hsv = cv2.cvtColor(canvas, cv2.COLOR_BGR2HSV)
        canvas = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        save_state()

    # NEGATIVE IMAGE
    elif key == ord("i"):
        canvas = cv2.bitwise_not(canvas)
        save_state()

    # SEPIA
    elif key == ord("p"):  
        kernel = np.array([[0.272, 0.534, 0.131],
                           [0.349, 0.686, 0.168],
                           [0.393, 0.769, 0.189]])
        canvas = cv2.transform(canvas, kernel)
        save_state()

    # CARTOON
    elif key == ord("o"):
        gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 5)
        edges = cv2.adaptiveThreshold(gray, 255,
                                      cv2.ADAPTIVE_THRESH_MEAN_C,
                                      cv2.THRESH_BINARY, 9, 9)
        color_img = cv2.bilateralFilter(canvas, 9, 250, 250)
        canvas = cv2.bitwise_and(color_img, color_img, mask=edges)
        save_state()

    # ===== COLORS =====
    # RED
    elif key == ord("R"):
        color = (0,0,255)
        eraser = False

    # GREEN
    elif key == ord("G"):
        color = (0,255,0)
        eraser = False

    # BLUE
    elif key == ord("B"):
        color = (255,0,0)
        eraser = False

    # ===== SHAPES =====
    # LINE
    elif key == ord("1"):
        cv2.line(canvas, (50,50), (450,50), color, brush_size)
        save_state()

    # RECTANGLE
    elif key == ord("2"):
        cv2.rectangle(canvas, (100,100), (300,300), color, brush_size)
        save_state()

    # CIRCLE
    elif key == ord("3"):
        cv2.circle(canvas, (250,250), 80, color, brush_size)
        save_state()

    # POLYGON
    elif key == ord("4"):
        pts = np.array([[200,100],[300,200],[250,300],[150,200]], np.int32)
        pts = pts.reshape((-1,1,2))
        cv2.polylines(canvas, [pts], True, color, brush_size)
        save_state()

    # ERASER
    elif key == ord("x"):
        eraser = True

    # BRUSH SIZE
    elif key == ord("+"):
        brush_size += 2
    elif key == ord("-"):
        brush_size = max(1, brush_size - 2)

    # ===== UNDO / REDO =====
    elif key == ord("u"):
        undo()
    elif key == ord("y"):
        redo()

    # CLEAR
    elif key == ord("c"):
        canvas = img.copy()
        save_state()

    # SAVE
    elif key == ord("s"):
        filename = output_dir / f"output_{int(time.time())}.png"
        cv2.imwrite(str(filename), canvas)
        print(f"Saved: {filename}")

    # QUIT
    elif key == ord("q"):
        break

cv2.destroyAllWindows()