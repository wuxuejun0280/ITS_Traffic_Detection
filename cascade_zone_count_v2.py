import cv2

pathes = []
lengthhistory = []
updatehistory = []
count = 0

font = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10, 500)
fontScale = 1
fontColor = (255, 0, 255)
lineType = 2

greenzone = (195, 205)
redzone = (205, 350)
# delete path if it is not updated in the next update_threshold frame
update_threshold = 8


# https://github.com/ParadropLabs/traffic-camera
def remove_overlaps(detections):
    """
    Remove detection results that are fully enclosed (redundant).
    """
    keep = set(range(len(detections)))

    for i in range(len(detections)):
        x1, y1, w1, h1 = detections[i]

        for j in range(i + 1, len(detections)):
            x2, y2, w2, h2 = detections[j]

            if j in keep and \
                    x2 >= x1 and y2 >= y1 and \
                    x2 + w2 <= x1 + w1 and y2 + h2 <= y1 + h1:
                keep.remove(j)
            elif i in keep and \
                    x1 >= x2 and y1 >= y2 and \
                    x1 + w1 <= x2 + w2 and y1 + h1 <= y2 + h2:
                keep.remove(i)

    return [detections[i] for i in keep]


def add_in_green_zone(cars):
    for (x, y, w, h) in cars:
        centery = int(y + h / 2)
        centerx = int(x + w / 2)
        if greenzone[0] <= centery <= greenzone[1]:
            for path in pathes:
                (xx, yy, ww, hh) = path[len(path) - 1]
                centerxx = int(xx + ww / 2)
                if abs(centerx - centerxx) < 15:
                    return
            pathes.append([(x, y, w, h)])
            lengthhistory.append(1)
            updatehistory.append(0)


def append_in_red_zone(cars):
    indexHistory = []
    for (x, y, w, h) in cars:
        centery = int(y + h / 2)
        centerx = int(x + w / 2)
        if redzone[0] <= centery <= redzone[1]:
            minx = 999
            miny = 999
            index = -1
            for (i, path) in enumerate(pathes):
                (xx, yy, ww, hh) = path[len(path) - 1]
                centeryy = int(yy + hh / 2)
                centerxx = int(xx + ww / 2)
                # skip if the position of current is blow the last point of the path
                if abs(centerx - centerxx) < minx:
                    minx = abs(centerx - centerxx)
                    miny = abs(centery - centeryy)
                    index = i
            if minx+miny < 50 and index not in indexHistory:
                pathes[index].append((x, y, w, h))
                indexHistory.append(index)


def increment_count():
    global count
    for (i, path) in enumerate(pathes):
        (xx, yy, ww, hh) = path[len(path) - 1]
        centeryy = int(yy + hh / 2)
        if abs(centeryy - redzone[1]) < 30 or len(path) > 8:
            count = count + 1
            del pathes[i]
            del lengthhistory[i]
            del updatehistory[i]


def check_update(threshold):
    for (i, length) in enumerate(lengthhistory):
        if length < len(pathes[i]):
            updatehistory[i] = 0
            lengthhistory[i] = len(pathes[i])
        else:
            updatehistory[i] += 1

    for (i, iteration_not_updated) in enumerate(updatehistory):
        if iteration_not_updated > threshold:
            del pathes[i]
            del lengthhistory[i]
            del updatehistory[i]


def detect_cars_camera(cascade_file):
    car_cascade = cv2.CascadeClassifier()
    car_cascade.load(cascade_file)

    cap = cv2.VideoCapture("traffic.mp4")
    ret = True

    while cap.isOpened():
        ret, frame = cap.read()
        if frame is None:
            continue
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_gray = cv2.equalizeHist(frame_gray)

        cars = car_cascade.detectMultiScale(image=frame_gray,
                                            scaleFactor=1.1,
                                            minNeighbors=2,
                                            minSize=(35, 35),
                                            maxSize=(250, 250))

        cars = remove_overlaps(cars)

        for (x, y, w, h) in cars:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)
            cv2.circle(frame, (int(x + w / 2), int(y + h / 2)), 3, (0, 0, 255), -1)

        add_in_green_zone(cars)
        append_in_red_zone(cars)
        increment_count()
        check_update(update_threshold)

        cv2.putText(frame, 'count: ' + str(count),
                    bottomLeftCornerOfText,
                    font,
                    fontScale,
                    fontColor,
                    lineType)

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, greenzone[0]), (2000, greenzone[1]),
                      (0, 255, 0), -1)
        cv2.rectangle(overlay, (0, redzone[0]), (2000, redzone[1]),
                      (0, 0, 255), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5,
                        0, frame)
        for path in pathes:
            for (i, point) in enumerate(path):
                if i == 0:
                    continue
                (x1, y1, w1, h1) = path[i - 1]
                (x2, y2, w2, h2) = path[i]
                centery1 = int(y1 + h1 / 2)
                centerx1 = int(x1 + w1 / 2)
                centery2 = int(y2 + h2 / 2)
                centerx2 = int(x2 + w2 / 2)
                cv2.line(frame, (centerx1, centery1), (centerx2, centery2), (255, 0, 0), 1)

        cv2.imshow("VideoFeed", frame)

        # print("time: " + str(cap.get(cv2.CAP_PROP_POS_FRAMES)) + " " + "count: " + str(len(cars)))

        if cv2.waitKey(1) & 0xff == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


detect_cars_camera("vehicle.xml")
