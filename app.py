from flask import Flask, render_template, Response, request, jsonify
import cv2
from ultralytics import YOLO
import time
import random

# model
model = YOLO("yolo-Weights/yolov8n.pt")

# object classes
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

app = Flask(__name__)

# Message queue
messages = []
oldMap = {}
peopleSaved = 0

# Webcam capture
videoPath = './recreation-beach-summer.mp4'
videoCapture = cv2.VideoCapture(videoPath)
camera = cv2.VideoCapture(0)


@app.route('/')
def index():
    return render_template('index.html', messages=messages)

# Video feed route
def gen_frames():
  global oldMap
  global peopleSaved
  index = 0
  
  while True:
    success, frame = videoCapture.read()  # read the camera frame
    if not success:
      videoCapture.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Loop the video
      continue
    
    results = model(frame, stream=True)
      
    # coordinates
    newMap = {}
    
    for r in results:
      boxes = r.boxes

      for box in boxes:
        # bounding box
        x1, y1, x2, y2 = box.xyxy[0]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values

        # put box in cam
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)
        
        # class name
        cls = int(box.cls[0])
        className = classNames[cls]
        
        # object details
        org = [x1, y1]
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 1
        color = (255, 0, 0)
        thickness = 2

        cv2.putText(frame, className, org, font, fontScale, color, thickness)
        
        if className in newMap:
          newMap[className] += 1
        else:
          newMap[className] = 1
    
    if index == 2:
      index = 0
      remove_expired_messages()
      
      #compare
      newKeys = newMap.keys()
      for key in newKeys:
        if (key in oldMap and newMap[key] > oldMap[key]) or (key not in oldMap):
          send_message({
            "coords": {
              "lat": 42.684192 + random.randrange(0, 10000) / 10000000.0,
              "lng": -70.754180 + random.randrange(0, 10000) / 10000000.0
            },
            
            "text": key + " found!"
          })
        
        if key in oldMap and newMap[key] - oldMap[key] > 1 and key == 'person':
          peopleSaved += 1
          
      oldMap = newMap
    else:
      index += 1
  
    ret, buffer = cv2.imencode('.jpg', frame)
    frame = buffer.tobytes()
    yield (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route("/get_output")
def get_output():
  global oldMap
  return oldMap

@app.route("/get_saved")
def get_saved():
  global peopleSaved
  stringPeopleSaved = f"{peopleSaved}"
  return stringPeopleSaved

@app.route('/video_feed')
def video_feed():
  return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Function to add a message programmatically
def send_message(message):
  if message:
    current_time = time.time()
    messages.append({'coordinates': message["coords"], 'text': message["text"], 'time': current_time})
  return messages

@app.route('/get_messages')
def get_messages():
  # Only return the text part of each message
  return jsonify({'messages': messages})
  
# Function to remove expired messages
def remove_expired_messages():
  current_time = time.time()
  # Keep only messages that are less than 4 seconds old
  global messages
  messages = [msg for msg in messages if current_time - msg['time'] < 4]

if __name__ == '__main__':
  app.run(debug=True)
