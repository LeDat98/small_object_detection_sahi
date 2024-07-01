from sahi.utils.yolov8 import download_yolov8s_model
from sahi import AutoDetectionModel
from sahi.predict import get_prediction, get_sliced_prediction
import cv2
import time

color_map = {
    "person": (0, 0, 255),
    #other classes
}

# Download YOLOv8 model
yolov8_model_path = "yolov8x.pt"
# download_yolov8s_model(yolov8_model_path)

# Load the YOLOv8 model
detection_model = AutoDetectionModel.from_pretrained(
    model_type='yolov8',
    model_path=yolov8_model_path,
    confidence_threshold=0.5,
    device="cuda:0",  # or 'cuda:0'
)

# Open video file
video_path = "input_video.mp4"
cap = cv2.VideoCapture(video_path)

# Get video information
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create VideoWriter object to save the output video
output_path = "output_video.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
print(total_frames)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    #% done of video
    frame_number = cap.get(cv2.CAP_PROP_POS_FRAMES)
    print(f"processing {int(frame_number / total_frames * 100)} % of video")
    x1_crop = 785
    y1_crop = 0
    x2_crop = 1180
    y2_crop = 1080
    # Crop the frame to the specified region
    cropped_frame = frame[y1_crop:y2_crop, x1_crop:x2_crop]
    cropped_frame_width = x2_crop - x1_crop
    cropped_frame_height = y2_crop - y1_crop
    # Convert frame from BGR to RGB
    frame_rgb = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)

    # Sliced Inference with YOLOv8
    start_time = time.time()
    result = get_sliced_prediction(
        frame_rgb,
        detection_model,
        slice_height=int(cropped_frame_height / 4),
        slice_width=int(cropped_frame_width / 1),
        overlap_height_ratio=0.2,
        overlap_width_ratio=0.2
    )
    end_time = time.time()
    time_detection = end_time - start_time
    print(f"detection time: {time_detection} seconds")
    start_time = time.time()

    object_prediction_list = result.object_prediction_list

    # Draw bounding boxes and labels on the frame
    for obj in object_prediction_list:
        bbox = obj.bbox
        x1, y1, x2, y2 = int(bbox.minx), int(bbox.miny), int(bbox.maxx), int(bbox.maxy)
        label = obj.category.name

        # Adjust coordinates to original frame
        x1 += x1_crop
        x2 += x1_crop

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color_map[label], 2)  # Red color in BGR

        # Draw label and score
        label_text = f"{label}"
        cv2.putText(frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_map[label], 2)  # Red color in BGR

    # Write the frame to the output video
    out.write(frame)
    end_time = time.time()
    time_draw = end_time - start_time
    print(f"draw and write time: {time_draw} seconds")
    print("--------------------------------------------------")
    print("total time: ", time_detection + time_draw)
    # Optionally, display the frame
    # Comment out the following lines if you don't need to display each frame
    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and writer
cap.release()
out.release()
cv2.destroyAllWindows()
