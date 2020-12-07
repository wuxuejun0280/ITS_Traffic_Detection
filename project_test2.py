import cv2
import numpy as np


video = 'traffic.mp4'
def train_bg_subtractor(inst, cap, num=800):
    '''
        BG substractor need process some amount of frames to start giving result
    '''
    print('Training BG Subtractor...')
    i = 0
    while cap.isOpened():
        ret, frame = cap.read()
        inst.apply(frame, None, 0.002)
        i += 1
        if i >= num:
            return cap

def filter_mask(img):

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))

    # Fill any small holes
    closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    # Remove noise
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)

    # Dilate to merge adjacent blobs
    dilation = cv2.dilate(opening, kernel, iterations=1)

    return dilation


def main():

    # creting MOG bg subtractor with 500 frames in cache
    # and shadow detction
    bg_subtractor = cv2.createBackgroundSubtractorKNN(
        history=500, detectShadows=True)

    # Set up image source
    # You can use also CV2, for some reason it not working for me
    cap = cv2.VideoCapture(video)

    # skipping 700 frames to train bg subtractor
    train_bg_subtractor(bg_subtractor, cap, num=800)


    # Open output video file
    out = cv2.VideoWriter('output_of_project.avi', cv2.VideoWriter_fourcc(*'MJPG'), 30.0,
                          (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                           int(cap.get(
                               cv2.CAP_PROP_FRAME_HEIGHT))))
    # Boolean for saving the first video frame
    first_frame = True
    # For calculating again the GoodFeaturesToTrack
    run = 0
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    while cap.isOpened():
        ret, frame = cap.read()

        fg_mask = bg_subtractor.apply(frame, None, 0.001)
        fg_mask[fg_mask<200]=0
        fg_mask = filter_mask(fg_mask)

        frame = fg_mask.copy()
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        contours, hierarchy = cv2.findContours(
            fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw the contours to frame. Drawing the contours after features so contours will be on top layer
        frame = cv2.drawContours(frame, contours, -1, (255, 0, 0), 1)

        # Draw a box over the contours
        for i in contours:
            if cv2.contourArea(i) < 200:  # Area is just empirically tested to be the best at my case
                continue
            x, y, w, h = cv2.boundingRect(i)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # Show the frames
        cv2.imshow('frame', frame)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

        # Save the background removed and tracked frame to output video
        out.write(frame)

        # Save the first frame to image
        if first_frame:
            cv2.imwrite('first_frame.jpg', frame)
            first_frame = False

    # Close both input and output video file
    del out
    del cap


# ============================================================================

if __name__ == "__main__":

    main()