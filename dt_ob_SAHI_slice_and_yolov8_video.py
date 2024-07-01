from sahi.utils.yolov8 import download_yolov8s_model
from sahi import AutoDetectionModel
from sahi.predict import get_prediction, get_sliced_prediction
import cv2
import time
color_map = {
    "person" : (0, 0, 255),
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

# Calculate new width and height (reduce to half)
new_width = width 
new_height = height 

# Define the codec and create VideoWriter object to save the output video
output_path = "output_video.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (new_width, new_height))
total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
print(total_frames)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    #% done of video
    frame_number = cap.get(cv2.CAP_PROP_POS_FRAMES)
    print(f"processing {int(frame_number / total_frames * 100)} % of video")

    # Resize the frame to half size
    frame_resized = cv2.resize(frame, (new_width, new_height))

    # Convert frame from BGR to RGB
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

    # Sliced Inference with YOLOv8
    start_time = time.time()
    result = get_sliced_prediction(
        frame_rgb,
        detection_model,
        slice_height=new_height,
        slice_width=int(new_width / 2),
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
    
        # Draw bounding box
        cv2.rectangle(frame_resized, (x1, y1), (x2, y2), color_map[label], 2)  # Red color in BGR

        # Draw label and score
        label_text = f"{label}"
        cv2.putText(frame_resized, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_map[label], 2)  # Red color in BGR

    # Write the frame to the output video
    out.write(frame_resized)
    end_time = time.time()
    time_draw = end_time - start_time
    print(f"draw and write time: {time_draw} seconds")
    print("--------------------------------------------------")
    print("total time: ", time_detection + time_draw)
    # Optionally, display the frame
    # Comment out the following lines if you don't need to display each frame
    cv2.imshow('Frame', frame_resized)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and writer
cap.release()
out.release()
cv2.destroyAllWindows()
