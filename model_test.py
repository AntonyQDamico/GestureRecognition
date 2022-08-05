import tensorflow as tf
import numpy as np
import os
import cv2
import mediapipe as mp

# create variables for mediapipe model and drawing
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
actions = np.array(['hello', 'thanks', 'iloveyou'])
model = tf.keras.models.load_model('action.h5')


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


sequence = []
sentence = []
threshold = 0.85

colors = [(245, 117, 16), (117, 245, 16), (16, 117, 145)]


def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0, 60 + num * 40), (int(prob * 100), 90 + num * 40), colors[num], -1)
        cv2.putText(output_frame, actions[num], (0, 85 + num * 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                    cv2.LINE_AA)
    return output_frame


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

        # Prediction logic
        keypoints = extract_landmarks(results)
        sequence.append(keypoints)
        sequence = sequence[-30:]

        if len(sequence) == 30:
            res = model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]
            # print(actions[np.argmax(res)])
        else:
            res = [0, 0, 0]

        # visualization logic
        if res[np.argmax(res)] > threshold:
            if len(sentence) > 0:
                if actions[np.argmax(res)] != sentence[-1]:
                    sentence.append(actions[np.argmax(res)])
            else:
                sentence.append(actions[np.argmax(res)])

        if len(sentence) > 5:
            sentence = sentence[-5:]

        cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
        cv2.putText(image, ' '.join(sentence), (3, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # visualize probability
        image = prob_viz(res, actions, image, colors)

        # show to screen
        cv2.imshow('OpenCV Feed', image)

        # break gracefully
        if cv2.waitKey(10) == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
