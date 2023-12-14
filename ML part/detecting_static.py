from ultralytics import YOLO

# Load a model
model = YOLO("./best-2.pt")

# Path variables
save_path = "./results/"
image_path = "./inputs/images/image1.jpg"
video_path = "./inputs/videos/video1.mp4"

# Detection
results = model.predict(source=video_path, project=save_path, save=True, show=True)

# Filter results based on a confidence threshold
threshold = 0.7
filtered_results = [result for result in results if result['confidence'] > threshold]

result = filtered_results[0] 
# Extracting data to appropriate variables
# for box in result.boxes:
#     conf = round(box.conf[0].item(), 2)
    
#     # Check if confidence is greater than or equal to 0.5
#     if conf >= 0.8:
#         class_id = result.names[box.cls[0].item()]
#         cords = box.xyxy[0].tolist()
#         cords = [round(x) for x in cords]

#         print("Object type:", class_id)
#         print("Coordinates:", cords)
#         print("Probability:", conf)
#         print("---")
