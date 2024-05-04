import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import os
import sys
import ipdb
import pickle

def dot(vA, vB):
  return vA[0]*vB[0]+vA[1]*vB[1]

class BadPoseAlert:
  def __init__(self) -> None:
    self.detector = self.load_detector()  
    self.detection_result = None
    self.sample_pose_landmarks = None
    
  def load_sample_pose(self, sample_pose_path):
    if (os.path.exists(sample_pose_path) == True):
      with open(sample_pose_path, 'rb') as f:
        self.sample_pose_landmarks = pickle.load(f)
    else:
      print("Standard pose file does not exist. Please run calib.py to create a standard pose file.")
      sys.exit(1)

  def load_detector(self):
    base_options = python.BaseOptions(model_asset_path='pose_est_weight\pose_landmarker_heavy.task')
    options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    output_segmentation_masks=True)
    detector = vision.PoseLandmarker.create_from_options(options)
    return detector

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
  

  def get_correlation_score(self, frame):
      image_frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
      self.detection_result = self.detector.detect(image_frame)

      filtered_pose_landmarks = {
        "nose": self.detection_result.pose_landmarks[0][0],
        "left shoulder": self.detection_result.pose_landmarks[0][11],
        "left hip": self.detection_result.pose_landmarks[0][23],
        "ear": self.detection_result.pose_landmarks[0][3],
        "height": image_frame.height,
        "width": image_frame.width,
      }
  
      nose_obs = filtered_pose_landmarks["nose"]
      left_shoulder_obs = filtered_pose_landmarks["left shoulder"]
      left_hip_obs = filtered_pose_landmarks["left hip"]
      ear_obs = filtered_pose_landmarks["ear"]

      nose_sample = self.sample_pose_landmarks["nose"]
      left_shoulder_sample = self.sample_pose_landmarks["left shoulder"]
      left_hip_sample = self.sample_pose_landmarks["left hip"]
      ear_sample = self.sample_pose_landmarks["ear"]

      dx = left_hip_sample.x - left_hip_obs.x
      dy = left_hip_sample.y - left_hip_obs.y

      for item in [nose_obs, left_shoulder_obs, left_hip_obs, ear_obs]:
        item.x += dx
        item.y += dy

      # Calculate the correlation score
      obs_arr = np.array([[nose_obs.x, nose_obs.y], [left_shoulder_obs.x, left_shoulder_obs.y], [ear_obs.x, ear_obs.y]])
      sample_arr = np.array([[nose_sample.x, nose_sample.y], [left_shoulder_sample.x, left_shoulder_sample.y], [ear_sample.x, ear_sample.y]])
      correlation_score = np.corrcoef(obs_arr.flatten(), sample_arr.flatten())[0, 1]
    
      return filtered_pose_landmarks, correlation_score
    
  
if __name__ == "__main__":
  frame = cv2.imread('./assets/badpose.PNG')
  frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

  badpose = BadPoseAlert()
  angle = badpose.get_angle(frame)
  annotated_image = badpose.draw_landmarks_on_image(frame, None)
  cv2.imshow("pose", cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))

  cv2.waitKey(0)