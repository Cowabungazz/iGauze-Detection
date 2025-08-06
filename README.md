# Gauze Detection System
Uploaded weights and script are old version. Latests are not released
<img width="1910" height="576" alt="image" src="https://github.com/user-attachments/assets/2fe99c18-6def-406a-b4e2-4703b24ff261" /> by tee chang zen

Demo:
https://www.youtube.com/watch?v=rdUgqIHsrIA&ab_channel=VictorRaj 

This was my NTU URECA project with Singapore General Hospital, where we built an AI vision system to prevent retained surgical gauze, or gossypiboma. The clinical problem is high-stakes: manual counts are reliable most of the time, but when they fail the consequences are severe. I joined an SGH-NTU team that already had a bench prototype at ~50% accuracy, and my brief from Aug 2023 to Jul 2024 was to harden it for real operating-theatre use. On the model side, I consolidated detection onto a YOLOv5 pipeline trained to spot two things—gauze and hands—so we could reason about when to count and when to pause. I expanded and re-annotated the dataset, adding about eight thousand OT images with realistic lighting, glove colors, and blood saturation, and simplified the label schema to reduce class confusion. I tuned thresholds and augmentations to cut false positives and false negatives, and I introduced on-the-fly error logging: during tests, a single keystroke would cache recent frames into an ‘edge-case bank’ that we could immediately re-label and push back into training. In our first live OT trial we discovered a ‘ghost bug’—the system would increment counts when nothing was there, often triggered by specular highlights or overlapping textures. I treated that as both a data and logic issue: we retrained with a stronger backbone and those new negatives, and I added a simple but effective ‘traffic light’ state machine in the app—yellow when hands are present, red while the system stabilizes, and green only when it’s safe to commit a count. We also gave nurses manual in/out adjustments and a full reset so the workflow never stalls. Hardware-wise, I worked with the clinicians to shrink the footprint and reposition the twin cameras so they wouldn’t obstruct sterile movements. By the end of the cycle, our mock-OT evaluations showed precision and recall above 0.90 with mAP@0.5 around 0.90, and the ghost-counting issue was eliminated in our subsequent theatre run. The project received NMRC Clinician Innovator Award support for field trials, and my biggest takeaway was how much real-world performance depends on small, human-centered changes: cleaner labels, a better pause-and-count policy, ergonomic hardware, and fast feedback loops. If I had another six months, I’d productionize the UI for voice or foot-pedal control and add a formal drift-monitor that auto-suggests when retraining is needed.

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
