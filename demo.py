import cv2
from BadPose import BadPoseAlert

# Read the video file
cap = cv2.VideoCapture('./assets/test_video.mp4')

# Check if the video file was successfully opened
if not cap.isOpened():
    print("Error opening video file")

# Read and display frames from the video
badpose = BadPoseAlert()

while cap.isOpened():
    ret, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Get the angle
    angle = badpose.get_angle(frame)

    annotated_image = badpose.draw_landmarks_on_image(frame, None)
    annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
    # Draw the angle on the annotated image
    cv2.putText(annotated_image, f"Angle: {angle}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    cv2.imshow('Annotated Image', annotated_image)

    # Check if a frame was successfully read
    if not ret:
        break

    # Display the frame
    # cv2.imshow('Video', frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(50) & 0xFF == 27:
        break

# Release the video file and close the window
cap.release()
cv2.destroyAllWindows()