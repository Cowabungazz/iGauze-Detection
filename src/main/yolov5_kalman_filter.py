import torch
import numpy as np
import cv2
import time
import os
import threading
from filterpy.kalman import KalmanFilter  # Make sure you have filterpy installed: pip install filterpy

import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

r1 = []
r2 = []
f = 0

# Load the YOLOv5 model
combined_model = torch.hub.load(
    'ultralytics/yolov5',   # GitHub repo (auto-cloned/cached)
    'custom',               # use a custom trained model
    path=r'runs\exp21\weights\best.pt',
    force_reload=True,     # set True if you want to refresh the cache
    trust_repo=True
)

combined_model.cuda()
combined_model.amp = False
combined_model.conf = 0.20  # Confidence threshold
combined_model.classes = [0, 1]  # Class 0 for gauze, 1 for hand

# Initialize variables for both cameras
onscreenIn, onscreenOut, countIn, countOut, startTime, endTime, countPlay = 0, 0, 0, 0, time.time(), 0, 0
frameCountIn, frameCountOut = 0, 0
isPaused = False
condition = 7

# Initialize video capture for two cameras
cap1 = cv2.VideoCapture(r'videos\Clean Test.mp4')  # First camera
cap2 = cv2.VideoCapture(r'videos\Dirty Test.mp4')   # Second camera
#uncomment this if want to use actual camera
#cap2 = cv2.VideoCapture(1)

# Check if cameras opened successfully
if not cap1.isOpened() or not cap2.isOpened():
    print("Error: Could not open one or both cameras.")
    exit()

# Display resolution and aspect ratio calculations
display_width_per_camera = 1920 // 2
display_height_per_camera = int(display_width_per_camera * (9 / 16))
display_height = display_height_per_camera

# Initialize the dictionary to track the previous key states
prev_keys = {}

framesToCapture = 100  # Change Value to change how long to capture when Pressing A for anomaly

PAUSE_DURATION = 0.7  # in seconds
UNPAUSE_DURATION = 0.1  # in seconds
UNPAUSE_DURATION += PAUSE_DURATION  # Adding to how long the pause will be
pause_timer = 0  # Timer variable to track the pause duration

prev_red_frame_onscreenIn = None
change_in_onscreenIn = 0
change_in_onscreenOut = 0
change_display_time = 0  # Variable to keep track of when to display the change

# Kalman filter tracking initialization
class KalmanBoxTracker:
    count = 0

    def __init__(self, bbox):
        self.kf = KalmanFilter(dim_x=7, dim_z=4)

        self.kf.F = np.array([[1, 0, 0, 0, 1, 0, 0],
                              [0, 1, 0, 0, 0, 1, 0],
                              [0, 0, 1, 0, 0, 0, 1],
                              [0, 0, 0, 1, 0, 0, 0],
                              [0, 0, 0, 0, 1, 0, 0],
                              [0, 0, 0, 0, 0, 1, 0],
                              [0, 0, 0, 0, 0, 0, 1]])

        self.kf.H = np.array([[1, 0, 0, 0, 0, 0, 0],
                              [0, 1, 0, 0, 0, 0, 0],
                              [0, 0, 1, 0, 0, 0, 0],
                              [0, 0, 0, 1, 0, 0, 0]])

        self.kf.R[2:, 2:] *= 10.0
        self.kf.P[4:, 4:] *= 1000.0  # Give high uncertainty to the unobservable initial velocities
        self.kf.P *= 10.0

        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        self.kf.x[:4] = self.convert_bbox_to_z(bbox)

        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

    def update(self, bbox):
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(self.convert_bbox_to_z(bbox))

    def predict(self):
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] = 0
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(self.convert_x_to_bbox(self.kf.x))
        return self.history[-1]

    def get_state(self):
        return self.convert_x_to_bbox(self.kf.x).flatten()

    @staticmethod
    def convert_bbox_to_z(bbox):
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        x = bbox[0] + w / 2.0
        y = bbox[1] + h / 2.0
        s = w * h    # scale is just area
        r = w / float(h)
        return np.array([x, y, s, r]).reshape((4, 1))

    @staticmethod
    def convert_x_to_bbox(x, score=None):
        w = np.sqrt(x[2] * x[3])
        h = x[2] / w
        if score is None:
            return np.array([x[0] - w / 2.0, x[1] - h / 2.0, x[0] + w / 2.0, x[1] + h / 2.0]).reshape((1, 4))
        else:
            return np.array([x[0] - w / 2.0, x[1] - h / 2.0, x[0] + w / 2.0, x[1] + h / 2.0, score]).reshape((1, 5))

def compute_iou(box1, box2):
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    intersect_min_x = max(x1_min, x2_min)
    intersect_min_y = max(y1_min, y2_min)
    intersect_max_x = min(x1_max, x2_max)
    intersect_max_y = min(y1_max, y2_max)

    intersect_area = max(0, intersect_max_x - intersect_min_x) * max(0, intersect_max_y - intersect_min_y)
    
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    
    union_area = box1_area + box2_area - intersect_area
    
    iou = intersect_area / union_area
    return iou

def draw_colored_box(image, color, thickness=2):
    height, width = image.shape[:2]
    start_point = (5, 5)  # Starting coordinate, (5, 5) indicates the top-left corner.
    end_point = (width - 5, height - 5)  # Ending coordinate, (width-5, height-5) indicates the bottom-right corner.
    cv2.rectangle(image, start_point, end_point, color, thickness)

def video(frames, filename):
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter(filename, fourcc, 30.0, (frames[0].shape[1], frames[0].shape[0]))

    for frame in frames:
        out.write(frame)

    out.release()
    print('video stored')

def put_stroked_text(image, text, position, font, font_scale, text_color, stroke_color, thickness, stroke_thickness):
    # Draw the stroke (outline)
    cv2.putText(image, text, position, font, font_scale, stroke_color, stroke_thickness)
    # Draw the actual text
    cv2.putText(image, text, position, font, font_scale, text_color, thickness)

def handle_key_press(key_char, action):
    if prev_keys.get(key_char, -1) != key and key == ord(key_char):
        action()
        prev_keys[key_char] = key
    elif key != ord(key_char):
        prev_keys[key_char] = -1

timestamp = time.strftime("%Y%m%d_%H%M%S")
folder_name = f"captured_videos_{timestamp}"
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

video_count = 0
count = 0
min_hits = 7  # Minimum number of hits before displaying the tracker

trackers = []
frame_count = 0

while True:
    ret, frame1 = cap1.read()
    ret1, frame2 = cap2.read()

    if not ret or not ret1:
        print("Failed to capture frames from cameras")
        break

    frame1 = cv2.resize(frame1, (display_width_per_camera, display_height_per_camera))
    frame2 = cv2.resize(frame2, (display_width_per_camera, display_height_per_camera))

    combined_frame = np.hstack((frame1, frame2))

    results = combined_model(combined_frame)

    prevOnScreenIn, prevOnScreenOut = onscreenIn, onscreenOut
    a, b, handDetected = 0, 0, False
    key = cv2.waitKey(10) & 0xFF

    handle_key_press('1', lambda: globals().update(countIn=countIn + 1))
    handle_key_press('2', lambda: globals().update(countIn=countIn - 1))
    handle_key_press('3', lambda: globals().update(countOut=countOut + 1))
    handle_key_press('4', lambda: globals().update(countOut=countOut - 1))
    handle_key_press('c', lambda: globals().update(countOut=0, countIn=0))

    # Kalman filter tracking
    detections = []
    for detection in results.xyxy[0]:
        detected_class = detection[5].item()
        if detected_class == 1:  # Apply Kalman filter only to gauze detections
            detections.append(detection[:4].cpu().numpy())
            if detection[0].item() < frame1.shape[1]:
                a += 1
            else:
                b += 1
        elif detected_class == 0:
            handDetected = True

    # Update Kalman trackers
    trackers_to_remove = []
    for tracker in trackers:
        tracker.predict()
        if tracker.time_since_update > 1:
            trackers_to_remove.append(tracker)

    trackers = [t for t in trackers if t not in trackers_to_remove]

    for det in detections:
        matched = False
        for tracker in trackers:
            iou = compute_iou(det, tracker.get_state())
            if iou > 0.9:
                tracker.update(det)
                matched = True
                break
        if not matched:
            trackers.append(KalmanBoxTracker(det))

    # Manually draw bounding boxes and labels
    image = combined_frame
    for detection in results.xyxy[0]:
        x1, y1, x2, y2, conf, cls = detection
        if int(cls) in combined_model.classes:
            color = (0, 255, 0) if int(cls) == 0 else (0, 0, 255)
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            label = f'{combined_model.names[int(cls)]} {conf:.2f}'
            font_scale = 0.5
            thickness = 1
            cv2.putText(image, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

    if handDetected and f == 0:
        pause_timer = time.time()
        isPaused = True
        box_color = (0, 255, 255)
        draw_colored_box(image, box_color)
    elif pause_timer > 0 and time.time() - pause_timer >= PAUSE_DURATION and time.time() - pause_timer <= UNPAUSE_DURATION and f == 0:
        isPaused = False
        box_color = (0, 0, 255)
        draw_colored_box(image, box_color)
    elif pause_timer > 0 and time.time() - pause_timer >= UNPAUSE_DURATION and f == 0:
        isPaused = True
        box_color = (0, 255, 0)
        draw_colored_box(image, box_color)
    elif f == 0:
        isPaused = True
        box_color = (0, 255, 255)
        draw_colored_box(image, box_color)
    onscreenInTemp, onscreenOutTemp = a, b
    if not isPaused:
        onscreenIn, onscreenOut = a, b
        if onscreenIn > prevOnScreenIn:
            countIn += onscreenIn - prevOnScreenIn
            change_in_onscreenIn = onscreenIn - prevOnScreenIn
        if onscreenIn < prevOnScreenIn:
            change_in_onscreenIn = onscreenIn - prevOnScreenIn
        if onscreenOut > prevOnScreenOut:
            countOut += onscreenOut - prevOnScreenOut
            change_in_onscreenOut = onscreenOut - prevOnScreenOut
        if onscreenOut < prevOnScreenOut:
            change_in_onscreenOut = onscreenOut - prevOnScreenOut
        countPlay = countIn - countOut - onscreenIn

    # Display only trackers that have been updated at least `min_hits` times
    for tracker in trackers:
        if tracker.hits >= min_hits:
            pred_bbox = tracker.get_state()
            x1, y1, x2, y2 = int(pred_bbox[0]), int(pred_bbox[1]), int(pred_bbox[2]), int(pred_bbox[3])
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)

    endTime = time.time()
    fps = 1 / (endTime - startTime)
    startTime = endTime

    put_stroked_text(image, f'On Screen = {onscreenInTemp}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                     1, (255, 255, 255), (0, 0, 0), 2, 8)
    put_stroked_text(image, f'Total In = {countIn}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX,
                     1, (255, 255, 255), (0, 0, 0), 2, 8)
    put_stroked_text(image, f'{change_in_onscreenIn}', (10, 110), cv2.FONT_HERSHEY_SIMPLEX,
                     1, (255, 255, 255), (0, 0, 0), 2, 8)
    put_stroked_text(image, f'On Screen = {onscreenOutTemp}', (display_width_per_camera + 10, 30),
                     cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), (0, 0, 0), 2, 8)
    put_stroked_text(image, f'Total Out = {countOut}', (display_width_per_camera + 10, 70),
                     cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), (0, 0, 0), 2, 8)
    put_stroked_text(image, f'{change_in_onscreenOut}', (display_width_per_camera + 10, 110), cv2.FONT_HERSHEY_SIMPLEX,
                     1, (255, 255, 255), (0, 0, 0), 2, 8)
    put_stroked_text(image, f'In Play = {countPlay}', (960 - 100, display_height - 50),
                     cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), (0, 0, 0), 2, 8)
    put_stroked_text(image, f'FPS = {round(fps, 1)}', (960 - 100, display_height - 20),
                     cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), (0, 0, 0), 2, 8)

    if key == ord('q'):
        break

    if key == ord('a'):
        f = 1
        print('frames storing')
        box_color = (0, 0, 255)
        draw_colored_box(image, box_color)
        isPaused = True

    if f == 1:
        if count <= framesToCapture:
            r1.append(frame1)
            r2.append(frame2)
            count += 1
            box_color = (0, 0, 255)
            draw_colored_box(image, box_color)
            text_size = cv2.getTextSize('CAPTURING FRAMES', cv2.FONT_HERSHEY_SIMPLEX, 4, 4)[0]
            text_x = (image.shape[1] - text_size[0]) // 2
            text_y = (image.shape[0] + text_size[1]) // 2
            cv2.putText(image, 'CAPTURING FRAMES', (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 255), 4)
            isPaused = True
        else:
            f = 0
            count = 0
            video_count += 1
            video(r1, f'{folder_name}/video_{video_count}_camera1.mp4')
            video(r2, f'{folder_name}/video_{video_count}_camera2.mp4')
            r1.clear()
            r2.clear()

    cv2.imshow('Gauze Detection', image)

cap1.release()
cap2.release()
cv2.destroyAllWindows()