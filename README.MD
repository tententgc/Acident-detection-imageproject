## Accident Detection and Notification using YOLOv8 and DETR

Accidents on the road can have serious consequences, and quick response times are crucial for minimizing harm. This project aims to enhance road safety by automatically detecting accidents through object detection in real-time video streams (also in static images and videos) and sending a notification to concerned authorities. This repository contains the implementation of an accident detection system utilizing the YOLOv8 object detection model.

### Step 1: Setup
- Create a virtual environment and activate it: `python -m venv venv && source venv/bin/activate`
- Install Prerequisites: `pip install -r requirements.txt`
- Clone this repository: `git clone https://github.com/tententgc/Acident-detection-imageproject.git`
- `cd` into the directory: `cd Acident-detection-imageproject`, then proceed to Step 2

### Step 2: Trainning model
- Using Dataset from Roboflow [Accident Dataset](https://universe.roboflow.com/yolovideos/accident-detection-bcc2v/browse?queryText=&pageSize=50&startingIndex=0&browseQuery=true)
- Training model with keras Model
- Training model with YOLO v8 
- Training Model with DinoDETR 


### Step 3: Detection

#### Step 3.1: Static Image/Video Detection

- To perform static image/video detection, use the file named `detecting_static.py` available in `ML Part` folder
- Run the script: `python detecting_static.py`
- The script will load the model, perform object detection on the specified image (sample input images available in `inputs` folder), and save the results in the `results` directory.

#### Step 3.2: Videostream Detection

- To perform videostream detection, use the file named `detecting_videostream.py` available in `ML Part` folder
- Run the script: `python detecting_videostream.py`
- The script will load the model, perform object detection on the input videostream (specified by the `stream_url`) by converting it to individual frames.

## Collabrator 
- 64090500404 ธัญพิสิษฐ์ บัวประคอง
- 64090500407 วรินทร สิทธิสินธุ์

