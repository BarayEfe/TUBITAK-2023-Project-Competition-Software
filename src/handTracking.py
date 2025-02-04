import cv2
import mediapipe as mp
import time
import matplotlib.pyplot as plt
from matplotlib.patheffects import withStroke
import numpy as np
import pytesseract
import threading

timestamps = {}
thread_res_list = []


def remove_all_but_last(lst):
    if len(lst) > 1:
        lst = lst[-1:]
    return lst


def detect_page_number(img, thread_res_list):
    img = cv2.flip(img, 1)
    # x,y for origin
    # h for height and w for width
    x, y, w, h = 0, 0, 640, 120
    roi = img[y:y + h, x:x + w]
    page_num = pytesseract.image_to_string(roi)
    thread_res_list.append(page_num)
    remove_all_but_last(thread_res_list)


def if_hand_escaped(data):
    for outer_key, outer_value in data.items():
        for inner_key, inner_value in outer_value.items():
            entries = inner_value['entries']
            exits = inner_value['exits']
            while len(exits) < len(entries):
                exits.append(time.time())
    return data


def track_finger_time(timestamps, coords_from_txt, cx, cy):
    for page, questions in coords_from_txt.items():
        if page == thread_res_list[-1]:
            for i, question in enumerate(questions):
                upper_left, lower_right = question
                if upper_left[0] <= cx <= lower_right[0] and upper_left[1] <= cy <= lower_right[1]:
                    # finger is within question coordinates
                    if page not in timestamps:
                        timestamps[page] = {}
                    if i not in timestamps[page]:
                        timestamps[page][i] = {"entries": [time.time()], "exits": []}
                    else:
                        if len(timestamps[page][i]["entries"]) == len(timestamps[page][i]["exits"]):
                            timestamps[page][i]["entries"].append(time.time())
                else:
                    # finger is not within question coordinates
                    if page in timestamps and i in timestamps[page]:
                        if len(timestamps[page][i]["entries"]) > len(timestamps[page][i]["exits"]):
                            timestamps[page][i]["exits"].append(time.time())

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


def read_given_txt(filename):
    with open(filename, 'r') as f:
        line = f.readline()
        total_pages = int(line.split('#')[1])
        questions_per_page = []
        for i in range(total_pages):
            line = f.readline()
            questions_per_page.append(int(line.split('*')[1]))
        patterns = []
        for i in range(total_pages):
            line = f.readline()
            patterns.append(line.split('-')[1])
    coordinates_by_page = {}

    for i, pattern in enumerate(patterns):
        page_number = i + 1
        coordinates = []
        questions = pattern.split('/')
        for question in questions:
            coords = question.split('_')
            upper_left = tuple(map(int, coords[0].split(',')))
            bottom_right = tuple(map(int, coords[1].split(',')))
            coordinates.append([upper_left, bottom_right])
        coordinates_by_page[page_number] = coordinates
    return coordinates_by_page


coords_from_txt = read_given_txt("ht.txt")

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# For webcam input:
cap = cv2.VideoCapture(0)

with mp_hands.Hands(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        success, image = cap.read()

        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue

        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        # To improve performance, optionally mark the image as not writeable to
        image.flags.writeable = False
        results = hands.process(image)

        image_height, image_width, _ = image.shape
        # window height: 480
        # window width: 640

        # Draw the hand annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # ocr in a separate thread
        ocr_thread = threading.Thread(target=detect_page_number, args=(image, thread_res_list))
        ocr_thread.start()
        # uncomment below 2 lines to see ocr result
        # if len(thread_res_list) != 0:
        #    print(thread_res_list[-1])
        ocr_thread.join()

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for ids, landmrk in enumerate(hand_landmarks.landmark):
                    cx, cy = landmrk.x * image_width, landmrk.y * image_height
                    if ids == 6:  # to track only index finger pip (Proximal Interphalangeal Joint)
                        # print(cx, cy) #comment out to see index finger pip coords
                        track_finger_time(timestamps, coords_from_txt, cx, cy)
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        else:
            timestamps = if_hand_escaped(timestamps)

        cv2.namedWindow("TED ANTALYA")
        cv2.imshow("TED ANTALYA", image)

        if cv2.waitKey(5) & 0xFF == ord('q'):  # press q to terminate
            break
cap.release()
cv2.destroyAllWindows()

print(timestamps)
total_time = calculate_total_time(timestamps)
print(total_time)

data = total_time
# example output below
# data = {1: {4: 15.021275520324707,3: 22.021275520324707,2: 16.021275520324707, 1: 12.021275520324707, 0: 16.36930775642395},
#        2: {4: 15.021275520324707,3: 32.021275520324707,2: 16.021275520324707, 1: 12.021275520324707, 0: 16.36930775642395},
#        3: {4: 75.021275520324707,3: 12.021275520324707,2: 16.021275520324707, 1: 12.021275520324707, 0: 16.36930775642395},
#        4: {4: 25.021275520324707,3: 52.021275520324707,2: 76.021275520324707, 1: 12.021275520324707, 0: 16.36930775642395},
#        5: {4: 55.021275520324707,3: 82.021275520324707,2: 77.021275520324707, 1: 12.021275520324707, 0: 16.36930775642395}}
# Create a figure with subplots

for i, (page_number, questions) in enumerate(data.items()):
    fig, ax = plt.subplots(1, 2, figsize=(20, 5))
    ax[0].set_title(f"Sayfa {page_number}: Soru Başına Harcanan Toplam Süre (s)", pad=30)
    ax[0].set_ylabel("Toplam Harcanan Süre (s)")
    ax[0].set_xlabel("Soru Numarası")
    ax[0].set_xticks(range(0, len(data[i + 1])))
    ax[0].set_xticklabels(["{0}. Soru".format(x + 1) for x in range(0, len(data[i + 1]))])
    question_numbers = list(questions.keys())
    question_times = list(questions.values())
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
    ax[1].set_title(f"Sayfa {page_number}: Soru Başına Harcanan Toplam Süre (%)", pad=30)

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
