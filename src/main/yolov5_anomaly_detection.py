import torch
import numpy as np
import cv2
import time
import os
import threading
import collections
import numpy as np
import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath


def draw_colored_box(image, color, thickness=5):
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


r1 = []
r2 = []
f = 0

# Load the YOLOv5 model
#combined_model = torch.hub.load(r"C:\Users\iGauze\yolov5", 'custom', trust_repo=True, source='local',
#                                path=r"C:\Users\iGauze\Desktop\Yolov5Exp\exp19\weights\best.pt", force_reload=True)

# Load the YOLOv5 model
#combined_model = torch.hub.load(r"C:\Users\iGauze\yolov5", 'custom', trust_repo=True, source='local',
#                                path=r"C:\Users\iGauze\Desktop\Yolov5Exp\exp25\weights\best.pt", force_reload=True)

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


capvalues = [0,1,2,3]
selected_cameras = []

#tcz
key = ord('t')  # Initialize key to 't' to enter the camera selection loop

# Select two cameras
for i in range(2):  # We need to select two cameras
    for cap_index in capvalues:
        while True:
            tempcap = cv2.VideoCapture(cap_index, cv2.CAP_DSHOW)
            while True:
                ret, frame = tempcap.read()
                if not ret:
                    print(f"Camera {cap_index} not accessible.")
                    break  # Break out of the inner while loop if the camera is not accessible

                cv2.imshow('frame', frame)
                key = cv2.waitKey(10) & 0xFF

                if key == ord('y'):
                    tempcap.release()
                    cv2.destroyAllWindows()
                    selected_cameras.append(cap_index)
                    capvalues.remove(cap_index)
                    k=0
                    break  # Break out of the inner while loop and move to the next camera selection
                if key == ord('n'):
                    tempcap.release()
                    cv2.destroyAllWindows()
                    break  # Break out of the inner while loop and try the next camera index
                if key == ord('t'):
                    tempcap.release()
                    cv2.destroyAllWindows()
                    # cap1 = cv2.VideoCapture(r'videos\Clean Test.mp4')  # First camera
                    # cap2 = cv2.VideoCapture(r'videos\Dirty Test.mp4')   # Second camera
                    cap1, cap2 = cv2.VideoCapture(r'videos\Clean Test.mp4'), cv2.VideoCapture(r'videos\Dirty Test.mp4')
                    k=1
                    print("Switched to test videos.")
                    break  # Break out of the inner while loop and end camera selection

            if key == ord('y') or key == ord('n') or key == ord('t'):
                break  # Break out of the outer while loop to re-evaluate the next camera index

        if key == ord('t') or key == ord('y'):
            break  # Break out of the for loop if 't' is pressed to switch to test videos

    if key == ord('t'):
        break  # Break out of the main for loop if 't' is pressed to switch to test videos

print(f"Selected cameras: {selected_cameras}")

# Initialize VideoCapture objects with the selected cameras
# if k==0:
#     cap1 = cv2.VideoCapture(selected_cameras[0], cv2.CAP_DSHOW)
#     cap2 = cv2.VideoCapture(selected_cameras[1], cv2.CAP_DSHOW)



# Initialize video capture for two cameras
cap1 = cv2.VideoCapture(r'videos\Clean Test.mp4')  # First camera
cap2 = cv2.VideoCapture(r'videos\Dirty Test.mp4')   # Second camera
#cap1 = cv2.VideoCapture(selected_cameras[0])  # First camera
#cap2 = cv2.VideoCapture(selected_cameras[1])   # Second camera

# Display resolution and aspect ratio calculations
display_width_per_camera = 1600 // 2
display_height_per_camera = 900
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

skipper=0
temptime11 = 14
temptime12 = 14
temptime13 = 14
temptime14 = 14
temptime21 = 14
temptime22 = 14
temptime23 = 14
temptime24 = 14
unpause_time = 0

# Buffers for storing data
buffer1 = collections.deque(maxlen=100)
buffer2 = collections.deque(maxlen=100)
last_change_time1=collections.deque(maxlen=100)
last_change_time2=collections.deque(maxlen=100)

'''
# Parameters for anomaly detection
window_size = 50
fluctuation_threshold = 0.2
consistent_threshold = 0.5 


# Function to detect rapid fluctuations
def detect_rapid_fluctuations(buffer, window_size, threshold):
    if len(buffer) < window_size:
        return False  # Not enough data to analyze
    
    window = np.array(buffer)[-window_size:]
    mean = np.mean(window)
    std_dev = np.std(window)
    
    # Detect if the standard deviation is high, indicating rapid fluctuations
    if std_dev > threshold:
        return True
    return False

# Function to detect consistent changes
def detect_consistent_changes(buffer, consistent_threshold):
    if len(buffer) < 2:
        return False  # Not enough data to analyze
    
    # Detect if the current value is consistently different from the mean of the buffer
    mean = np.mean(buffer)
    if abs(buffer[-1] - mean) > consistent_threshold:
        return True
    return False
'''

timestamp = time.strftime("%Y%m%d_%H%M%S")
folder_name = f"captured_videos_{timestamp}"
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

video_count = 0
count = 0

while True:
    ret, frame1 = cap1.read()
    ret1, frame2 = cap2.read()

    if not ret or not ret1:
        print("Failed to capture frames from cameras")
        break
    frame1 = cv2.rotate(frame1, cv2.ROTATE_90_COUNTERCLOCKWISE)
    frame2 = cv2.rotate(frame2, cv2.ROTATE_90_COUNTERCLOCKWISE)

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


    #Detection Section

    for detection in results.xyxy[0]:
        detected_class = detection[5].item()
        x1, y1, x2, y2, conf, cls = detection
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        if detected_class == 1:
            if detection[0].item() < frame1.shape[1]-10:
                a += 1
            else:
                b += 1
        elif detected_class == 0:
            handDetected = True

        # Draw the bounding boxes
        color = (255, 255, 0) if detected_class == 0 else (255, 0, 0)
        cv2.rectangle(combined_frame, (x1, y1), (x2, y2), color, 2)  # Adjust thickness here

        # Draw the text labels
        label = f'{combined_model.names[int(cls)]} {conf:.2f}'
        font_scale = 0.7  # Adjust font scale here
        thickness = 1  # Adjust thickness here
        cv2.putText(combined_frame, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

    image = combined_frame
       
            
    # Hand Detection and Traffic Light Section
    
    if handDetected and f == 0:
        pause_timer = time.time()
        isPaused = True
        box_color = (0, 255, 255)
        draw_colored_box(image, box_color)
        buffer1.clear()
        buffer2.clear()
        #last_change_time1.clear()
        #last_change_time2.clear()
        skipper=1
        
    elif pause_timer > 0 and time.time() - pause_timer >= PAUSE_DURATION and time.time() - pause_timer <= UNPAUSE_DURATION and f == 0:
        isPaused = False
        box_color = (0, 0, 255)
        draw_colored_box(image, box_color)
        skipper=1
        
    elif pause_timer > 0 and time.time() - pause_timer >= UNPAUSE_DURATION and f == 0:
        isPaused = True
        box_color = (0, 255, 0)
        draw_colored_box(image, box_color)
        skipper=0
    elif f == 0:
        isPaused = True
        box_color = (0, 255, 255)
        draw_colored_box(image, box_color)
        
    onscreenInTemp, onscreenOutTemp = a, b

    # Anomaly detection section
    if not handDetected:
        current_time = time.time()
        
        if onscreenInTemp != buffer1[-1] if buffer1 else None:
            last_change_time1.append(current_time)
        
        if onscreenOutTemp != buffer2[-1] if buffer2 else None:
            last_change_time2.append(current_time)
    
        buffer1.append(onscreenInTemp)
        buffer2.append(onscreenOutTemp)
        
        try:  #Two try statements so they do not affect one another
            temptime11 = current_time - last_change_time1[-1]
            temptime12 = last_change_time1[-1] - last_change_time1[-2]
            temptime13 = last_change_time1[-2] - last_change_time1[-3]
            temptime14 = last_change_time1[-3] - last_change_time1[-4]
        except:
            why=1+1

        try:
            temptime21 = current_time - last_change_time2[-1]
            temptime22 = last_change_time2[-1] - last_change_time2[-2]
            temptime23 = last_change_time2[-2] - last_change_time2[-3]
            temptime24 = last_change_time2[-3] - last_change_time2[-4]
        except:
            why=1+1
            #temptime11 = temptime12 = temptime21 = temptime22 = 13
            
        anomalytime=0.5
        
        # Detect rapid fluctuations
        if (temptime11 < anomalytime and temptime12 < anomalytime and temptime13 < anomalytime and temptime14 < anomalytime\
           ) or (temptime21 < anomalytime and temptime22 < anomalytime and temptime23 < anomalytime and temptime24 < anomalytime):
            isPaused = True
            box_color = (0, 0, 255)
            draw_colored_box(image, box_color)
            text_size = cv2.getTextSize('ANOMALY DETECTED', cv2.FONT_HERSHEY_SIMPLEX, 4, 4)[0]
            text_x = (image.shape[1] - text_size[0]) // 2
            text_y = (image.shape[0] + text_size[1]) // 2
            cv2.putText(image, 'ANOMALY DETECTED', (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 255), 4)
        
    if not handDetected and skipper == 0:
        
        if (temptime11 > 10 and temptime11 < 10.5) or (temptime21 > 10 and temptime21 < 10.5):
            isPaused = False

        elif temptime11 >= 10.5 or temptime21 >= 10.5:
            isPaused = True

   
    '''
    # Anomaly detection section
    if not handDetected:
        buffer1.append(onscreenInTemp)
        buffer2.append(onscreenOutTemp)
        
        rapid_fluctuations1 = detect_rapid_fluctuations(buffer1, window_size, fluctuation_threshold)
        rapid_fluctuations2 = detect_rapid_fluctuations(buffer2, window_size, fluctuation_threshold)
        
        consistent_changes1 = detect_consistent_changes(buffer1, consistent_threshold)
        consistent_changes2 = detect_consistent_changes(buffer2, consistent_threshold)
        
        if rapid_fluctuations1 or rapid_fluctuations2:
            isPaused = True
            box_color = (0, 0, 255)
            draw_colored_box(image, box_color)
            text_size = cv2.getTextSize('ANOMALY DETECTED', cv2.FONT_HERSHEY_SIMPLEX, 4, 4)[0]
            text_x = (image.shape[1] - text_size[0]) // 2
            text_y = (image.shape[0] + text_size[1]) // 2
            cv2.putText(image, 'ANOMALY DETECTED', (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 255), 4)
        elif consistent_changes1 or consistent_changes2:
            isPaused = False
        else:
            isPaused = True
        '''
    if not isPaused:
        onscreenIn, onscreenOut = a, b
        if onscreenIn > prevOnScreenIn:
            countIn += onscreenIn - prevOnScreenIn
            change_in_onscreenIn = onscreenIn - prevOnScreenIn
        if onscreenIn < prevOnScreenIn:
            change_in_onscreenIn = onscreenIn - prevOnScreenIn
        if onscreenOut > prevOnScreenOut:
            countOut += onscreenOut - prevOnScreenOut
            change_in_onscreenOut=  onscreenOut - prevOnScreenOut
        if onscreenOut < prevOnScreenOut:   
            change_in_onscreenOut=  onscreenOut - prevOnScreenOut
            
        countPlay = countIn - countOut - onscreenIn
#         if prev_red_frame_onscreenIn is not None:
#             change_in_onscreenIn = onscreenIn - prev_red_frame_onscreenIn
#             change_display_time = time.time()  # Update the display time
#         prev_red_frame_onscreenIn = onscreenIn

    endTime = time.time()
    fps = 1 / (endTime - startTime)
    startTime = endTime
   

    # Text Section
    
    #(image, text, position, font, font_scale, text_color, stroke_color, thickness, stroke_thickness)
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
    if countPlay == 0:
        put_stroked_text(image, f'In Play = {countPlay}', (960 - 300, display_height - 50),
                     cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), (0, 0, 0), 3, 8)
    else:
        put_stroked_text(image, f'In Play = {countPlay}', (960 - 300, display_height - 50),
                     cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), (0, 0, 0), 3, 8)
    put_stroked_text(image, f'FPS = {round(fps, 1)}', (960 - 100, display_height - 20),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), (0, 0, 0), 2, 8)


    # Exit Section
    if key == ord('q'):
        break


    #Frame Storing Section
    if key == ord('a'):
        f = 1
        print('frames storing')
        box_color = (0, 0, 255)
        draw_colored_box(image, box_color)
        isPaused = True

    # How it captures frames from prev section
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