# air-canvas
import cv2
import numpy as np
from collections import deque
from datetime import datetime
import os


save_folder = "SavedDrawings"
os.makedirs(save_folder, exist_ok=True)


buffer_size = 1024
points = deque(maxlen=buffer_size)
canvas = None


lower_blue = np.array([100, 150, 0])
upper_blue = np.array([140, 255, 255])


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    if canvas is None:
        canvas = np.zeros_like(frame)

    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

   
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    center = None

    if contours and len(contours) > 0:
        c = max(contours, key=cv2.contourArea)
        if cv2.contourArea(c) > 500:
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            M = cv2.moments(c)
            if M["m00"] > 0:
                center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                cv2.circle(frame, center, int(radius), (255, 0, 0), 2)
                points.appendleft(center)
            else:
                points.appendleft(None)
        else:
            points.appendleft(None)
    else:
        points.appendleft(None)

    
    for i in range(1, len(points)):
        if points[i - 1] is None or points[i] is None:
            continue
        cv2.line(canvas, points[i - 1], points[i], (255, 0, 0), 5)

    
    output = cv2.add(frame, canvas)

    
    cv2.imshow("Air Canvas", output)
    cv2.imshow("Mask", mask)

   
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('c'):
        canvas = np.zeros_like(frame)
        points.clear()
    elif key == ord('s'):
        filename = os.path.join(save_folder, f"drawing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        cv2.imwrite(filename, canvas)
        print(f"✅ Canvas saved as {filename}")


cap.release()
cv2.destroyAllWindows()
