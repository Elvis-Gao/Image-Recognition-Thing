from ultralytics import YOLO

# Initialize a YOLO-World model
model = YOLO('yolov8m-world.pt')  # or choose yolov8m/l-world.pt

# Define custom classes
model.set_classes(["cat"])

# Execute prediction for specified categories on an image
results = model.predict('videoExamples/images.jpg')

# Show results
results[0].show()