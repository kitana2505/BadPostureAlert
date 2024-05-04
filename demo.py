import cv2
from BadPose import BadPoseAlert
import ipdb

# Read the video file
cap = cv2.VideoCapture('./assets/test_video.mp4')
SAMPLE_POSE_PATH = "./standard_pose_landmark.pkl"
THRESHOLD = 0.85
OUTPUT_VIDEO_PATH = "./output_video.mp4"  # Specify the path for the output video file

# Check if the video file was successfully opened
if not cap.isOpened():
    print("Error opening video file")

# Read and display frames from the video
badpose = BadPoseAlert()
badpose.load_sample_pose(SAMPLE_POSE_PATH)

# Get the video properties
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Create a VideoWriter object to save the annotated frames as a video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Specify the codec for the output video
out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    # Check if a frame was successfully read
    if not ret:
        break    

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    filter_pose_landmark, correlation_score = badpose.get_correlation_score(frame)
    annotated_image = frame.copy()

    cv2.line(annotated_image, (int(filter_pose_landmark["nose"].x * width), int(filter_pose_landmark["nose"].y * height)), 
        (int(filter_pose_landmark["left shoulder"].x * width), int(filter_pose_landmark["left shoulder"].y * height)), 
        (0, 255, 0), 2)
    cv2.line(annotated_image, (int(filter_pose_landmark["left shoulder"].x * width), int(filter_pose_landmark["left shoulder"].y * height)), 
        (int(filter_pose_landmark["left hip"].x * width), int(filter_pose_landmark["left hip"].y * height)), 
        (0, 255, 0), 2)    

    if correlation_score > THRESHOLD:
        cv2.putText(annotated_image, "Good Posture :)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        cv2.putText(annotated_image, "Bad Posture!!!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    
    annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)

    cv2.imshow('Annotated Image', annotated_image)

    # Write the annotated frame to the output video file
    out.write(annotated_image)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release the video file, close the window, and release the output video file
cap.release()
out.release()
cv2.destroyAllWindows()