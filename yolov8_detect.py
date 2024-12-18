from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import numpy as np
import math

# Load the custom-trained YOLOv8 model
model = YOLO("/workspaces/Bambeleo/yolov8_model/best.pt")  # Replace "best.pt" with the path to your model weights file if needed

# Load an image for inference
image_path = "/workspaces/Bambeleo/test_video/output_video.mp4"  # Replace with the path to your test image
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for visualization

# Perform inference on the image
results = model(image_rgb)  # Runs inference

# Initialize variables
board_center = None
detected_objects = []

# Process results to find objects and identify the board
annotated_image = image_rgb.copy()  # Create a copy for annotation

for r in results:
    boxes = r.boxes.xyxy.cpu().numpy()  # Extract bounding boxes
    classes = r.boxes.cls.cpu().numpy()  # Extract class indices
    names = model.names  # Get class names from the model

    for box, cls in zip(boxes, classes):
        x1, y1, x2, y2 = map(int, box)  # Bounding box coordinates
        center_x = (x1 + x2) // 2  # Center X
        center_y = (y1 + y2) // 2  # Center Y

        class_name = names[int(cls)]  # Get the class name

        # Save detected objects for processing later
        detected_objects.append((class_name, center_x, center_y, box))

        # Check if the object is the board and set the board center
        if class_name == "board" and board_center is None:
            board_center = (center_x, center_y)

# Annotate all objects relative to the board center
if board_center:
    distances = []
    for obj in detected_objects:
        class_name, center_x, center_y, box = obj
        relative_x = center_x - board_center[0]
        relative_y = center_y - board_center[1]
        distance = math.sqrt(relative_x**2 + relative_y**2)
        distances.append((class_name, relative_x, relative_y, round(distance, 2)))

        # Annotate relative position on the image
        text = f"({relative_x}, {relative_y})"
        cv2.putText(annotated_image, text, (center_x + 10, center_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        cv2.circle(annotated_image, (center_x, center_y), 5, (255, 0, 0), -1)  # Blue points for other objects

    # Annotate the board
    cv2.putText(annotated_image, "(0, 0)", (board_center[0] + 10, board_center[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.circle(annotated_image, board_center, 5, (0, 0, 255), -1)  # Red point for the board

    # Print distances for reference
    print("Relative positions and distances from board center (0, 0):")
    for obj in distances:
        print(f"Object: {obj[0]}, Relative Position: ({obj[1]}, {obj[2]}), Distance: {obj[3]}")

    # Find the object with the shortest relative distance
    if distances:
        closest_object = min(distances, key=lambda x: x[3])
        print("\nObject with the shortest relative distance:")
        print(f"Object: {closest_object[0]}, Relative Position: ({closest_object[1]}, {closest_object[2]}), Distance: {closest_object[3]}")
else:
    print("Board not detected in the image.")
    for obj in detected_objects:
        class_name, center_x, center_y, box = obj
        print(f"Object detected: {class_name} at ({center_x}, {center_y})")

# Display the annotated image
plt.figure(figsize=(10, 10))
plt.imshow(annotated_image)
plt.axis("off")
plt.title("Relative Positions of Objects with Respect to the Board")
plt.show()



