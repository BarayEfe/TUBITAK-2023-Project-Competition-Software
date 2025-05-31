import cv2
import mediapipe as mp
import time
import matplotlib.pyplot as plt
from matplotlib.patheffects import withStroke
import numpy as np
import json
import arucoReader

import os

timestamps = {}
visuals = True


def remove_all_but_last(lst):
    if len(lst) > 1:
        lst = lst[-1:]
    return lst


def if_hand_escaped(data):
    for outer_key, outer_value in data.items():
        for inner_key, inner_value in outer_value.items():
            entries = inner_value['entries']
            exits = inner_value['exits']
            while len(exits) < len(entries):
                exits.append(time.time())
    return data


def track_finger_time(timestamps, cx, cy, page, frame):
    for question in json_data[page]["q"]:
        M, width, height = arucoReader.getTransformMatrix(frame)

        if M is None:
            continue

        paper_point = cv2.perspectiveTransform(np.array([[[cx, cy]]], dtype=np.float32), M)[0][0]

        upper_left = [width / json_data[page]["w"] * question["x1"], height / json_data[page]["h"] * question["y1"]]
        lower_right = [width / json_data[page]["w"] * question["x2"], height / json_data[page]["h"] * question["y2"]]
        # img = arucoReader.warpArucoImg(frame)

        # cv2.rectangle(img, (int(upper_left[0]), int(upper_left[1])), (int(lower_right[0]), int(lower_right[1])), (0, 255, 0), 3)
        # cv2.imshow("show me what you got", img)
        if upper_left[0] <= paper_point[0] <= lower_right[0] and upper_left[1] <= paper_point[1] <= lower_right[1]:
            # finger is within question coordinates
            print("question: " + str(question["n"]))
            if page not in timestamps:
                timestamps[page] = {}
            if question["n"] not in timestamps[page]:
                timestamps[page][question["n"]] = {"entries": [time.time()], "exits": []}
            else:
                if len(timestamps[page][question["n"]]["entries"]) == len(timestamps[page][question["n"]]["exits"]):
                    timestamps[page][question["n"]]["entries"].append(time.time())
        else:
            # finger is not within question coordinates
            if page in timestamps and question["n"] in timestamps[page]:
                if len(timestamps[page][question["n"]]["entries"]) > len(timestamps[page][question["n"]]["exits"]):
                    timestamps[page][question["n"]]["exits"].append(time.time())

    return timestamps


def calculate_total_time(timestamps):
    total_time = {}
    for page, questions in timestamps.items():
        total_time[page] = {}
        for i, question in questions.items():
            entries = question["entries"]
            exits = question["exits"]
            time_spent = sum([exits[i] - entries[i] for i in range(len(exits))])
            total_time[page][i] = time_spent

    return total_time


mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

print("Current working directory:", os.getcwd())

with open('data.json') as f:
    json_data = json.load(f)
    json_data = json_data["pages"]

print(json_data)

# For webcam input:
cap = cv2.VideoCapture(0)

with mp_hands.Hands(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        success, frame = cap.read()
        # frame = cv2.flip(frame, 0)

        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue

        # frame = cv2.cvtColor(cv2.flip(frame, 0), cv2.COLOR_BGR2RGB)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # To improve performance, optionally mark the image as not writeable to
        frame.flags.writeable = False
        results = hands.process(frame)

        image_height, image_width, _ = frame.shape
        # print(image_width, image_height)
        # window height: 480
        # window width: 640

        # Draw the hand annotations on the image.
        frame.flags.writeable = True
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for ids, landmrk in enumerate(hand_landmarks.landmark):
                    cx, cy = landmrk.x * image_width, landmrk.y * image_height
                    if ids == 8:  # to track only index finger pip (Proximal Interphalangeal Joint)
                        # print(cx, cy)  # comment out to see index finger pip coords
                        curpage = arucoReader.findPage(frame)
                        if curpage is None or frame is None or curpage >= len(json_data):
                            break
                        track_finger_time(timestamps, cx, cy, curpage, frame)
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        else:
            timestamps = if_hand_escaped(timestamps)

        if visuals:
            corners, ids = arucoReader.readImage(frame)
            if ids is not None:
                for corner in corners:
                    corner = corner.astype(int)
                    cv2.polylines(frame, [corner[0].reshape((-1, 1, 2))], isClosed=True, color=(0, 255, 0), thickness=2)

            cv2.imshow("Camera", frame)

            morphed = arucoReader.warpArucoImg(frame)
            if morphed is not None:
                cv2.imshow("othercam", morphed)

            if cv2.waitKey(5) & 0xFF == ord('q'):  # press q to terminate
                break
cap.release()
cv2.destroyAllWindows()

print(timestamps)
total_time = calculate_total_time(timestamps)
print(total_time)

data = total_time
# example output below
# data =  {0: {3: 75.021275520324707,2: 60.021275520324707, 1: 200.021275520324707, 4: 26.36930775642395}}
#       2: {3: 78.432127552032470,2: 37.021275520324707, 1: 49.021275520324707, 0: 23.36930775642395}}
# Create a figure with subplots

for i, (page_number, questions) in enumerate(data.items()):
    print(questions)
    fig, ax = plt.subplots(1, 2, figsize=(20, 5))
    ax[0].set_title(f"Sayfa {page_number + 1}: Soru Başına Harcanan Toplam Süre (s)", pad=30)
    ax[0].set_ylabel("Toplam Harcanan Süre (s)")
    ax[0].set_xlabel("Soru Numarası")
    ax[0].set_xlim(-0.5, len(json_data[page_number]["q"]) - 0.5)
    ax[0].set_xticks(range(0, len(json_data[page_number]["q"])))
    ax[0].set_xticklabels(["{}. Soru".format(x + 1) for x in range(0, len(json_data[page_number]["q"]))])

    question_numbers = []
    for key in questions.keys():
        question_numbers.append(key - 1)
    # question_numbers = list(questions.keys())
    question_times = list(questions.values())
    print(question_numbers)
    print(question_times)
    ax[0].bar(question_numbers, question_times, color=plt.cm.Pastel1.colors)
    for bar in ax[0].patches:
        ax[0].annotate(format(bar.get_height(), '.2f'),
                       (bar.get_x() + bar.get_width() / 2,
                        bar.get_height() / 2), ha='center', va='center',
                       size=15, xytext=(0, 8),
                       textcoords='offset points', color='white', fontweight='bold',
                       path_effects=[withStroke(linewidth=3, foreground='black')])

    total_time_per_page = sum(question_times)
    question_percentages = [time / total_time_per_page * 100 for time in question_times]
    ax[1].set_title(f"Sayfa {page_number + 1}: Soru Başına Harcanan Toplam Süre (%)", pad=30)

    wedges, texts = ax[1].pie(question_percentages, wedgeprops=dict(width=0.5), startangle=-40,
                              colors=plt.cm.Pastel1.colors)
    ax[1].legend(["{0}. Soru".format(x + 1) for x in question_numbers], loc="upper right",
                 bbox_transform=plt.gcf().transFigure)

    bbox_props = dict(boxstyle="square,pad=0.5", fc="w", ec="k", lw=0.72)
    kw = dict(arrowprops=dict(arrowstyle="-"),
              bbox=bbox_props, zorder=0, va="center")

    for i, p in enumerate(wedges):
        ang = (p.theta2 - p.theta1) / 2. + p.theta1
        y = np.sin(np.deg2rad(ang))
        x = np.cos(np.deg2rad(ang))
        horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
        connectionstyle = "angle,angleA=0,angleB={}".format(ang)
        kw["arrowprops"].update({"connectionstyle": connectionstyle})
        ax[1].annotate("%{0}".format(format(float(question_percentages[i]), ".2f")), xy=(x, y),
                       xytext=(1.35 * np.sign(x), 1.4 * y),
                       horizontalalignment=horizontalalignment, **kw)
        centre_circle = plt.Circle((0, 0), 0.50, fc='white')
        fig_donut = plt.gcf()
        fig_donut.gca().add_artist(centre_circle)
    fig.tight_layout()
    plt.show()
