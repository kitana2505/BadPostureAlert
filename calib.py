from BadPose import BadPoseAlert
import ipdb
import cv2
import numpy as np
import time
import pickle

if __name__ == "__main__":
    sample_video = './assets/test_video.mp4'

    # Read the video file
    cap = cv2.VideoCapture(sample_video)

    # Check if the video file was successfully opened
    if not cap.isOpened():
        print("Error opening video file")

    # Read and display frames from the video
    badpose = BadPoseAlert()

    while cap.isOpened():
        ret, frame = cap.read()
        # Check if a frame was successfully read
        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Get the angle
        angle = badpose.get_angle(frame)

        filter_pose_landmark = badpose.filtered_pose_landmarks

        # annotated_image = badpose.draw_landmarks_on_image(frame, None)

        annotated_image = frame.copy()
        height, width = annotated_image.shape[:2]

        cv2.line(annotated_image, (int(filter_pose_landmark["nose"].x * width), int(filter_pose_landmark["nose"].y * height)), 
            (int(filter_pose_landmark["left shoulder"].x * width), int(filter_pose_landmark["left shoulder"].y * height)), 
            (0, 255, 0), 2)
        cv2.line(annotated_image, (int(filter_pose_landmark["left shoulder"].x * width), int(filter_pose_landmark["left shoulder"].y * height)), 
            (int(filter_pose_landmark["left hip"].x * width), int(filter_pose_landmark["left hip"].y * height)), 
            (0, 255, 0), 2)    
        annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)

        cv2.putText(annotated_image, "Press space to save sample pose", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Pause the video if 'space' is pressed
        if cv2.waitKey(1) & 0xFF == ord(' '):
            with open('standard_pose_landmark.pkl', 'wb') as f:
                pickle.dump(filter_pose_landmark, f)
            print("Standard pose landmarks are saved as: ")

            for (key, value) in filter_pose_landmark.items():
                print(f"{key}: {value}, {value}")
            break

        cv2.imshow('Annotated Image', annotated_image)

        # Exit the loop if 'q' is pressed
        if cv2.waitKey(50) & 0xFF == 27:
            break

# Release the video file and close the window
cap.release()
cv2.destroyAllWindows()