import cv2

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
                                              minSize=(30, 30),
                                              maxSize=(200, 200))

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)

        cv2.imshow("VideoFeed", frame)

        if cv2.waitKey(1) & 0xff == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

detect_faces_camera("vehicle.xml")
