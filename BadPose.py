import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import sys
import ipdb

def dot(vA, vB):
  return vA[0]*vB[0]+vA[1]*vB[1]

class BadPoseAlert:
  def __init__(self) -> None:
    self.detector = self.load_detector()  
    self.detection_result = None

  def load_detector(self):
    base_options = python.BaseOptions(model_asset_path='pose_est_weight\pose_landmarker_heavy.task')
    options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    output_segmentation_masks=True)
    detector = vision.PoseLandmarker.create_from_options(options)
    return detector
  
  def angle(self, filtered_pose_landmarks):
      nose = [filtered_pose_landmarks["nose"].x, filtered_pose_landmarks["nose"].y]

      left_shoulder = [filtered_pose_landmarks["left shouder"].x, filtered_pose_landmarks["left shouder"].y]

      nose_leftshoulder_line = np.array(nose + left_shoulder)

      left_shoulder = [filtered_pose_landmarks["left shouder"].x, filtered_pose_landmarks["left shouder"].y]
      left_hip = [filtered_pose_landmarks["left hip"].x, filtered_pose_landmarks["left hip"].y]
                  
      leftshoulder_lefthip_line= np.array(left_shoulder + left_hip)

      lineA = nose_leftshoulder_line
      lineB = leftshoulder_lefthip_line
      vA = [(lineA[0]-lineA[2]), (lineA[1]-lineA[3])]
      vB = [(lineB[0]-lineB[2]), (lineB[1]-lineB[3])]

      # Get dot prod
      dot_prod = dot(vA, vB)
      # Get magnitudes
      # magA = dot(vA, vA)**0.5
      # magB = dot(vB, vB)**0.5
      magA = (vA[0]**2 + vA[1]**2)**0.5
      magB = (vB[0]**2 + vB[1]**2)**0.5
      
      # Get cosine value
      cos_ = dot_prod/magA/magB
      # Get angle in radians and then convert to degrees
      angle = np.arccos(dot_prod/magB/magA)
      angle = angle*180/np.pi
      return angle

  def draw_landmarks_on_image(self, rgb_image, detection_result):
    if detection_result == None:
      pose_landmarks_list = self.detection_result.pose_landmarks
    else:
      pose_landmarks_list = detection_result.pose_landmarks

    annotated_image = np.copy(rgb_image)

    # Loop through the detected poses to visualize.
    for idx in range(len(pose_landmarks_list)):
      pose_landmarks = pose_landmarks_list[idx]

      # Draw the pose landmarks.
      pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
      pose_landmarks_proto.landmark.extend([
        landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
      ])
      solutions.drawing_utils.draw_landmarks(
        annotated_image,
        pose_landmarks_proto,
        solutions.pose.POSE_CONNECTIONS,
        solutions.drawing_styles.get_default_pose_landmarks_style())
    return annotated_image
  
  def get_angle(self, frame):
    image_frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    self.detection_result = self.detector.detect(image_frame)
    filtered_pose_landmarks = {
      "nose": self.detection_result.pose_landmarks[0][0],
      "left shouder": self.detection_result.pose_landmarks[0][11],
      "left hip": self.detection_result.pose_landmarks[0][23],
    }
    angle = self.angle(filtered_pose_landmarks)
    return angle
    
  

if __name__ == "__main__":
  # image = mp.Image.create_from_file('./assets/badpose.PNG')
  # image = mp.Image.create_from_file('./assets/goodpose.PNG')
  frame = cv2.imread('./assets/badpose.PNG')
  frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

  badpose = BadPoseAlert()
  angle = badpose.get_angle(frame)
  annotated_image = badpose.draw_landmarks_on_image(frame, None)
  cv2.imshow("pose", cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))

  cv2.waitKey(0)