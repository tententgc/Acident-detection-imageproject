from ultralytics import YOLO

# Load a model
model = YOLO("./best.pt")

# path variables
save_path = "./results/"
image_path = "./inputs/images/image1.jpg"
video_path = "./inputs/videos/video1.mp4"

# detection
results = model.predict(source=video_path, project=save_path, save=True, show=True)
result = results[0]
box = result.boxes[0]


# extracting data to appropriate variables
for box in result.boxes:
    class_id = result.names[box.cls[0].item()]
    cords = box.xyxy[0].tolist()
    cords = [round(x) for x in cords]
    conf = round(box.conf[0].item(), 2)

    print("Object type:", class_id)
    print("Coordinates:", cords)
    print("Probability:", conf)
    print("---")
