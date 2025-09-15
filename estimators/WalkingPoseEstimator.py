import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import cv2
import time

PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
PoseLandmarkerResult = mp.tasks.vision.PoseLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

pose_frame = None
pose_result = None

prev_pose_result = None
delayed_update_counter = 0

prevResult = [True, True]

leftLeg = False
rightLeg = False
currentLeg = "None"

stepNum = 0

LEFT_HIP = 23
RIGHT_HIP = 24

LEFT_KNEE = 25
RIGHT_KNEE = 26

LEFT_HEEL = 29
RIGHT_HEEL = 30

LEFT_FOOT_INDEX = 31
RIGHT_FOOT_INDEX = 32


def withinTolerance(value: float, test: list, tolerance: list):
  for i in range(len(test)):
    if not ((value - tolerance[i]) < test[i] < (value + tolerance[i])):
      return False
  return True

def legsDetected(detection_result: PoseLandmarkerResult):
  # Knee and heel and hip are all aligned vertically
  pose_landmarks_list = detection_result.pose_landmarks

  if len(pose_landmarks_list) == 1:
    if pose_landmarks_list[0][LEFT_HIP].presence > 0.5 and pose_landmarks_list[0][RIGHT_HIP].presence > 0.5 and pose_landmarks_list[0][LEFT_KNEE].presence > 0.5 and pose_landmarks_list[0][RIGHT_KNEE].presence > 0.5 and pose_landmarks_list[0][LEFT_HEEL].presence > 0.5 and pose_landmarks_list[0][RIGHT_HEEL].presence > 0.5:
      return True
  return False

def alignedLeftFoot(detection_result: PoseLandmarkerResult):
  pose_landmarks_list = detection_result.pose_landmarks
  return withinTolerance(pose_landmarks_list[0][LEFT_HIP].x, [pose_landmarks_list[0][LEFT_KNEE].x, pose_landmarks_list[0][LEFT_HEEL].x], [0.05, 0.03]) # 0.05, 0.03 for walking

def alignedRightFoot(detection_result: PoseLandmarkerResult):
  pose_landmarks_list = detection_result.pose_landmarks
  return withinTolerance(pose_landmarks_list[0][RIGHT_HIP].x, [pose_landmarks_list[0][RIGHT_KNEE].x, pose_landmarks_list[0][RIGHT_HEEL].x], [0.05, 0.03]) # 0.05, 0.03 for walking

def stepDetected(detection_result: PoseLandmarkerResult):
  if legsDetected(detection_result):
    return [alignedLeftFoot(detection_result), alignedRightFoot(detection_result)]
  return [False, False]

def initiallyLeftForward(prev_detection_result: PoseLandmarkerResult):
  pose_landmarks_list = prev_detection_result.pose_landmarks
  return legsDetected(prev_detection_result) and pose_landmarks_list[0][LEFT_HIP].x < pose_landmarks_list[0][LEFT_HEEL].x

def initiallyRightForward(prev_detection_result: PoseLandmarkerResult):
  pose_landmarks_list = prev_detection_result.pose_landmarks
  return legsDetected(prev_detection_result) and pose_landmarks_list[0][RIGHT_HIP].x < pose_landmarks_list[0][RIGHT_HEEL].x


def draw_landmarks_on_image(detection_result: PoseLandmarkerResult, rgb_image: mp.Image, timestamp_ms: int):
  pose_landmarks_list = detection_result.pose_landmarks
  annotated_image = np.copy(rgb_image.numpy_view())

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
      solutions.drawing_utils.DrawingSpec(color=(255, 0, 0), thickness=4, circle_radius=2),
      solutions.drawing_utils.DrawingSpec(color=(255, 0, 0), thickness=6, circle_radius=2))

  global pose_frame
  pose_frame = annotated_image

  global pose_result
  pose_result = detection_result

base_options = python.BaseOptions(model_asset_path='models/pose_landmarker_full.task')
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=draw_landmarks_on_image,
    min_tracking_confidence=0.7, # 0.7
    min_pose_detection_confidence=0.7, # 0.7
    min_pose_presence_confidence=0.6) # 0.6
detector = vision.PoseLandmarker.create_from_options(options)

cap = cv2.VideoCapture("walking_assets/Walking.mp4")

while cap.isOpened():
  ret, frame = cap.read()

  if not ret:
    break

  frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

  timestamp = int(round(time.time()*1000))
  mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,data=frame)
  detector.detect_async(mp_image, timestamp)

  if pose_frame is not None:
    pose_frame = cv2.resize(pose_frame, (640,320))
    converted_frame = cv2.cvtColor(pose_frame, cv2.COLOR_RGB2BGR)

    if pose_result is not None:
      if (prev_pose_result is None):
        prev_pose_result = pose_result

      result = stepDetected(pose_result)

      if (not(leftLeg) and result[0] and not(prevResult[0]) and initiallyLeftForward(prev_pose_result)):
        stepNum+=1
        leftLeg = True
        rightLeg = False
        currentLeg = "Left Leg"
      if (not(rightLeg) and result[1] and not(prevResult[1]) and initiallyRightForward(prev_pose_result)):
        stepNum+=1
        rightLeg = True
        leftLeg = False
        currentLeg = "Right Leg"
      
      prevResult = result

      delayed_update_counter+=1
      if (delayed_update_counter >= 5): # 5 for walking
        prev_pose_result = pose_result
        delayed_update_counter = 0

    converted_frame = cv2.putText(converted_frame, currentLeg, (200, 50), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    converted_frame = cv2.putText(converted_frame, "Step Count: " + str(stepNum), (500, 50), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow("Step Counter", converted_frame)

  if cv2.waitKey(10) & 0xFF == ord('q'):
    break

cap.release()
cv2.destroyAllWindows()