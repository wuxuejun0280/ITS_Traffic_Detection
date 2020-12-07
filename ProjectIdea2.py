import cv2

# https://github.com/ParadropLabs/traffic-camera
def remove_overlaps(detections):
    """
    Remove detection results that are fully enclosed (redundant).
    """
    keep = set(range(len(detections)))

    for i in range(len(detections)):
        x1, y1, w1, h1 = detections[i]

        for j in range(i+1, len(detections)):
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


def detect_faces_camera(cascade_file):
    face_cascade = cv2.CascadeClassifier()
    face_cascade.load(cascade_file)

    cap = cv2.VideoCapture("traffic.mp4")
    ret = True

    while ret:
        ret, frame = cap.read()

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_gray = cv2.equalizeHist(frame_gray)

        faces = face_cascade.detectMultiScale(image=frame_gray,
                                              scaleFactor=1.1,
                                              minNeighbors=2,
                                              minSize=(35, 35),
                                              maxSize=(250, 250))

        faces = remove_overlaps(faces)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)

        cv2.imshow("VideoFeed", frame)

        print("time: " + str(cap.get(cv2.CAP_PROP_POS_FRAMES))+ " " + "count: " + str(len(faces)))

        if cv2.waitKey(1) & 0xff == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

detect_faces_camera("vehicle.xml")
