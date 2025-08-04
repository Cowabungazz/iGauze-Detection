# Gauze Detection System
Uploaded weights and script are old version. Latests are not released

# How to run?
Create src\videos -> put Clean and Dirty video inside
1. Open terminal in \src
2. python main\yolov5_anomaly_detection.py (anomaly warning)
3. python main\yolov5_kalman_filter.py (no anomaly warning)

# Kalman_filter in this code
How it’s used in your loop
Run YOLO, keep detections of class 1 (gauze).

predict() all existing trackers (advance them one frame).

Data association: for each detection, find a tracker whose predicted box overlaps (IoU) > 0.9 and update() it; otherwise start a new tracker.

Drop trackers that haven’t been updated for a bit (time_since_update > 1).

Why use it
Smooths jittery boxes.

Bridges missed detections for a frame or two by propagating motion.

Provides a lightweight notion of ID persistence (each KalmanBoxTracker has an id and hit_streak).

# Others
Frame Extraction to batch extract images from videos

COCO to Yolo and Split to split to convert COCO.json annotation from CVAT into Yolo format + Splitting into Train + Test + Val combos
