# Initialisation
import cv2
import numpy as np
import math


# Compute absolute colour difference of two images.
# The two images must have the same size.
# Return combined absolute difference of the 3 channels
def absDiff(image1, image2):
    if image1.shape != image2.shape:
        print('image size mismatch')
        return 0
    else:
        height, width, dummy = image1.shape
        # Compute absolute difference.
        diff = cv2.absdiff(image1, image2)
        a = cv2.split(diff)
        # Sum up the differences of the 3 channels with equal weights.
        # You can change the weights to different values.
        sum = np.zeros((height, width), dtype=np.uint8)
        for i in (1, 2, 3):
            ch = a[i - 1]
            cv2.addWeighted(ch, 1.0 / i, sum, float(i - 1) / i, gamma=0.0, dst=sum)
        return sum


# Function to extract some features from background with also choosable background color
# Got some help from https://stackoverflow.com/questions/29810128/opencv-python-set-background-colour/38516242
def setBackground(image, diff, threshold, bgcolor):
    # Getting the right binary mask based on diff array
    mask = cv2.threshold(diff, threshold, 1, cv2.THRESH_BINARY)
    # Inverse mask to know what pixels to change in background
    mask_inv = cv2.bitwise_not(mask[1] * 255) // 255
    # Making full size background with given color
    background_image = np.full(image.shape, bgcolor, dtype=np.uint8)
    # Extracting previously found feature with mask from original image
    fg = cv2.bitwise_and(image, image, mask=mask[1])
    # Extracting the area which is not in the feature area
    bg = cv2.bitwise_and(background_image, background_image, mask=mask_inv)
    # Merging background and foreground together
    fg_last = cv2.bitwise_or(fg, bg)



    return fg_last, fg


def average(video, sec):
    # Get the video original frame rate
    frame_rate = video.get(cv2.CAP_PROP_FPS)

    # Calculate the needed frames to process. Subtract 1 because we take the mean already out before the for-loop
    if sec == 0:
        frames_to_read = int(video.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
    else:
        frames_to_read = int(frame_rate * sec) - 1

    # Read the first frame to mean
    ret, mean = video.read()
    # Split to three color channels
    split_mean = cv2.split(mean)

    # Until given seconds
    for i in range(0, frames_to_read):
        ret, frame = video.read()

        # Split the frame to color channels
        split_frame = cv2.split(frame)

        # Calculate mean for every color channel
        for j in (0, 1, 2):
            cv2.addWeighted(split_frame[j], 1.0 / (i + 1), split_mean[j], float(1 - 1 / (i + 1)), gamma=0.0,
                            dst=split_mean[j])

    # Merge color channels together
    cv2.merge(split_mean, dst=mean)

    # Save weighted frame
    cv2.imwrite('weighted.jpg', mean)

    return mean


# -----------------------------
# -----------------------------
# Main

video = 'traffic.mp4'
# Open video file
cap = cv2.VideoCapture(video)
# Calculate the mean frame for given seconds
mean_frame = average(cap, 3)
# Close the video feed0
cap.release()

# Re-open the video file
cap = cv2.VideoCapture(video)
# Open output video file
out = cv2.VideoWriter('output_of_project.avi', cv2.VideoWriter_fourcc(*'MJPG'), 30.0, (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                                                                int(cap.get(
                                                                                    cv2.CAP_PROP_FRAME_HEIGHT))))
# Boolean for saving the first video frame
first_frame = True
# For calculating again the GoodFeaturesToTrack
run = 0
video_fps = cap.get(cv2.CAP_PROP_FPS)
while True:
    ret, frame = cap.read()
    if not ret:  # If out of frames exit the while loop
        break

    # Calculate the difference between video frame and the mean frame
    diff = absDiff(frame, mean_frame)
    # Set the background of a video frame to green
    set_bg = setBackground(frame, diff, 49, (0, 255, 0))

    frame = set_bg[0]
    car_frame = set_bg[1]
    gray_car_frame = cv2.cvtColor(car_frame, cv2.COLOR_BGR2GRAY)

    re, car_threshold = cv2.threshold(gray_car_frame, 50, 1, cv2.THRESH_BINARY)

    # Detect contours. RETR_EXTERNAL should only keep the outer contours
    contours, hierarchy = cv2.findContours(car_threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Got some help from https://docs.opencv.org/4.4.0/d4/dee/tutorial_optical_flow.html
    if first_frame:
        old_gray_car_frame = gray_car_frame  # Copy the first frame and do not detect anything
        mask = np.zeros_like(car_frame)  # Create mask to draw features
        corners = cv2.goodFeaturesToTrack(gray_car_frame, 50, 0.07, 15.0, False)  # Use goodFeaturesToTrack to find corners
    else:
        # Update the goodFeaturesToTrack at every 90 frames
        if run < 90:
            run += 1
        else:
            corners = cv2.goodFeaturesToTrack(old_gray_car_frame, 50, 0.07, 15.0, False)
            run = 0

        # Track previously found features
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray_car_frame, gray_car_frame, corners, None, (3, 3))

        good_new = p1[st == 1]
        good_old = corners[st == 1]

        # Draw the features with line
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()

            # # Calculate the velocity to each feature based on the new and old points.
            # dist = math.sqrt(pow(a-c, 2) + pow(b-d, 2))
            # velocity = dist * video_fps
            #
            # # Use change from old position to new to predict the car movement
            # line_lenght = velocity  # Just taking velocity so the longer the line the faster the movement is
            # new_y_line = (b - d) / dist * line_lenght
            # new_x_line = math.sqrt(pow(line_lenght, 2) - pow(new_y_line, 2))
            # y = int(b + new_y_line)
            #
            # # Based on is the movement is the car going right or left
            # if c - a >= 0:
            #     x = int(a - new_x_line)
            # else:
            #     x = int(a + new_x_line)
            #
            # # Draw the expected moving trajectory and already tracked line
            # cv2.line(frame, (int(a), int(b)), (x, y), (255, 255, 255), 1)
            mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), (0, 255, 255), 1)
        frame = cv2.add(frame, mask)

        # Save the frame as old
        old_gray_car_frame = gray_car_frame.copy()
        corners = good_new.reshape(-1, 1, 2)

    # Draw the contours to frame. Drawing the contours after features so contours will be on top layer
    frame = cv2.drawContours(frame, contours, -1, (255, 0, 0), 1)

    # Draw a box over the contours
    for i in contours:
        if cv2.contourArea(i) < 90:  # Area is just empirically tested to be the best at my case
            continue
        x, y, w, h = cv2.boundingRect(i)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 0), 2)

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
