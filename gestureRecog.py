# import packages
import cv2
import os
import time
import numpy as np
import mediapipe as mp
import tensorflow as tf
import sklearn
from matplotlib import pyplot as plt

# create variables for mediapipe model and drawing
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils


# function to detect mediapipe items
# Takes in the image to predict on and the model to use
# Return the image used and the results of the detection
def mediapipe_detection(image, model):
    # convert color from BGR base to RGB base
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # ensure image cannot be edited while processing occurs
    image.setflags(write=False)

    # use the model to make predictions of items
    results = model.process(image)

    # make image writable again for future work
    image.setflags(write=True)

    # convert color from RGB base to BGR base for future work
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results


# function to draw landmarks on the image
# Takes in the image to draw on and the results of the mediapipe detection
def draw_landmarks(image, results):
    # constant styling to be used
    landmark_spec = mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1)
    connection_spec = mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1)

    # Draw landmarks and contours for face, both hands, and pose
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
                              landmark_spec, connection_spec)
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              landmark_spec, connection_spec)
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              landmark_spec, connection_spec)
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                              landmark_spec, connection_spec)


# Key points With MP Holistic
cap = cv2.VideoCapture(1)

# Access mediapipe model
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:

    while cap.isOpened():
        # read feed
        ret, frame = cap.read()

        # make detections
        image, results = mediapipe_detection(frame, holistic)

        # draw all landmarks
        draw_landmarks(image, results)

        # show to screen
        cv2.imshow('OpenCV Feed', image)

        # break gracefully
        if cv2.waitKey(10) == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()


def extract_landmarks(results):
    # Extract pose key points, blank if nothing
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() \
        if results.pose_landmarks else np.zeros(33*4)

    # Extract hand landmarks, blank array if no detection
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() \
        if results.left_hand_landmarks else np.zeros(21*3)

    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() \
        if results.right_hand_landmarks else np.zeros(21*3)

    # Extract face landmarks, blank if no detection
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() \
        if results.face_landmarks else np.zeros(468*3)

    return np.concatenate([pose, face, lh, rh])


# collect keypoint values for training and testing
# Path for exported data, numpy array
DATA_PATH = os.path.join('MP_Data')
# Actions that we try to detect
actions = np.array(['hello', 'thanks', 'iloveyou'])
# Thirty videos worth of data
no_sequences = 30
# videos are going to be 30 frames in length
sequence_length = 30

cap = cv2.VideoCapture(1)
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:

    # Loop through actions
    for action in actions:
        for sequence in range(no_sequences):
            for frame_num in range(sequence_length):

                # read feed
                ret, frame = cap.read()

                # make detections
                image, results = mediapipe_detection(frame, holistic)

                # draw all landmarks
                draw_landmarks(image, results)

                # apply wait logic
                if frame_num == 0:
                    cv2.putText(image, 'STARTING COLLECTION', (120, 120),
                                 cv2.FONT_HERSHEY_SIMPLEX, 1, (0,225,0), 4, cv2.LINE_AA)
                    cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence) , (15, 12),
                                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    # show to screen
                    cv2.imshow('OpenCV Feed', image)
                    cv2.waitKey(2000)
                else:
                    cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15, 12),
                                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    # show to screen
                    cv2.imshow('OpenCV Feed', image)

                # export keypoint data
                keypoints = extract_landmarks(results)
                npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                np.save(npy_path, keypoints)

                # break gracefully
                if cv2.waitKey(10) == ord('q'):
                    break

cap.release()
cv2.destroyAllWindows()
