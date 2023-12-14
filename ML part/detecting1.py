from ultralytics import YOLO

# Load a model
model = YOLO("./best-2.pt")

# Path variables
save_path = "./results/"
image_path = "./inputs/images/image1.jpg"  # Note: This is not used in this script
video_path = "./inputs/videos/video1.mp4"

# Detection with a confidence threshold
threshold = 0.7
results = model.predict(source=video_path, project=save_path, save=True, show=True, conf=threshold)


result = results[0] if results else None
