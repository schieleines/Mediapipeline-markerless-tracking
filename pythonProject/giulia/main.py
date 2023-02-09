import mediapipe as mp
import cv2 as cv
import numpy as np
import os.path
import csv
import pickle
pickle.HIGHEST_PROTOCOL = 4
import pandas as pd


def mark_video(mp_hands, bodyparts, path, cap, points):
    with mp_hands.Hands(model_complexity=1, min_tracking_confidence=0.5, min_detection_confidence=0.5) as hands:
        frame_num = -1;
        while cap.isOpened():
            ret, frame = cap.read()
            if ret == True:
                frame_num += 1
                image_height, image_width, _ = frame.shape

                image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

                # Note that handedness is determined assuming the input image is mirrored,
                # i.e., taken with a front-facing/selfie camera with images flipped horizontally.
                # If it is not the case, please swap the handedness output in the application.
                image = cv.flip(image, 1)

                results = hands.process(image)

                if results.multi_hand_landmarks:

                    if len(results.multi_hand_landmarks) == 1:
                        hand = results.multi_hand_landmarks[0]

                        if (results.multi_handedness[0].classification[0].label == "Right"):
                            # print("Right - 1")
                            for lm in mp_hands.HandLandmark:
                                points[frame_num][lm.value][0] = image_width - hand.landmark[lm.value].x * image_width
                                points[frame_num][lm.value][1] = hand.landmark[lm.value].y * image_height
                                points[frame_num][lm.value][
                                    2] = 1  # hand.landmark[lm.value].visibility ALWAYS ZERO, UNCLEAR HOW TO UNLOCK
                        else:
                            for lm in mp_hands.HandLandmark:
                                points[frame_num][lm.value][0] = np.nan
                                points[frame_num][lm.value][1] = np.nan
                                points[frame_num][lm.value][2] = np.nan
                    elif len(results.multi_hand_landmarks) == 2:
                        for num, hand in enumerate(results.multi_hand_landmarks):
                            if (results.multi_handedness[num].classification[0].label == "Right"):
                                # print("Right - 2")
                                for lm in mp_hands.HandLandmark:
                                    points[frame_num][lm.value][0] = image_width - hand.landmark[
                                        lm.value].x * image_width
                                    points[frame_num][lm.value][1] = hand.landmark[lm.value].y * image_height
                                    points[frame_num][lm.value][
                                        2] = 1  # hand.landmark[lm.value].visibility ALWAYS ZERO, UNCLEAR HOW TO UNLOCK

                else:
                    for lm in mp_hands.HandLandmark:
                        points[frame_num][lm.value][0] = np.nan
                        points[frame_num][lm.value][1] = np.nan
                        points[frame_num][lm.value][2] = np.nan

            else:
                break

        cap.release()

        orderofbpincsv = bodyparts
        n_frames, n_joints, c = points.shape
        data = points.reshape(n_frames, n_joints * c)
        frameindex = list([i for i in range(n_frames)])

        scorer = 'DLC_resnet50_3rdThumb-trackingJan12shuffle1_150000'
        index = pd.MultiIndex.from_product([[scorer], orderofbpincsv, ['x', 'y', 'likelihood']],
                                           names=['scorer', 'bodyparts', 'coords'])
        frame = pd.DataFrame(np.array(data, dtype=float), columns=index, index=frameindex)

        frame.to_hdf(path[:-4] + ".h5", key="whatever", mode='w')
        frame.to_csv(path[:-4] + ".csv")

        with open(path[:-4] + ".pickle", "wb") as file:
            pickle.dump(frame, file, protocol=4)


mp_hands = mp.solutions.hands

bodyparts = []

for lm in mp_hands.HandLandmark:
    bodyparts.append(lm.name)

root = 'C:\\Users\\Ines Sebti\\PycharmProjects\\pythonProject\\videosigns.mp4"'  # CAREFUL: change accordingly to the root folder of the videos folder
for current_dir, directories, files in os.walk(root):
    if files:
        for f in files:
            if (f[-4:] == '.mp4'):
                path = os.path.join(current_dir, f)
                cap = cv.VideoCapture(path)

                if (cap.isOpened() == False):
                    print("Error opening file")
                else:
                    print(path)
                    n_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
                    # points = np.empty((n_frames, len(bodyparts), 2))
                    if (n_frames > 0):
                        points = np.empty((n_frames, len(bodyparts), 3))
                        mark_video(mp_hands, bodyparts, path, cap, points)

