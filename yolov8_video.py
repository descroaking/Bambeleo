from ultralytics import YOLO
import cv2

# Load the custom-trained YOLOv8 model
model = YOLO("/workspaces/Bambeleo/yolov8_model/best.pt")  # Replace "best.pt" with the path to your model weights file if needed

# Load a video for inference
video_path = "/workspaces/Bambeleo/test_video/output_video.mp4"  # Replace with the path to your video file
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Get video details
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Optional: Save the output video with detections
output_path = "/workspaces/Bambeleo/test_video/output_video.mp4"  # Replace with the path where to write the deteced video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Finished processing the video.")
        break

    cv2.namedWindow("YOLOv8 Inference", cv2.WINDOW_NORMAL)

    # Convert BGR to RGB for inference
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_rgb = cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2RGB)

    # Perform inference on the frame
    results = model(frame_rgb)

    # Annotate the frame with detection results
    for r in results:
        annotated_frame = r.plot()  # Annotated frame with bounding boxes

    # Write the annotated frame to the output video
    out.write(annotated_frame)

    # Display the annotated frame in a window

    cv2.imshow("YOLOv8 Inference", annotated_frame)

    key = cv2.waitKey(1) & 0xFF

    # Break on 'q' key press
    if key == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()


