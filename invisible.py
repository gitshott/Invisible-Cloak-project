import cv2
import numpy as np
import time
# Initialize webcam can be done wuth 0 or 1
cap = cv2.VideoCapture(0)
# Give the camera time to adjust and capture the background
time.sleep(2)
ret, background = cap.read()

# Flip the background horizontally
background = np.flip(background, axis=1)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Flip the frame horizontally to avoid mirror effect
    frame = np.flip(frame, axis=1)
    
    # Convert the frame from BGR to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define the HSV range for detecting red color
    # Note: Red color can be detected at both lower and upper boundaries in HSV space
    lower_red1 = np.array([0, 120, 70])  # Lower hue range for red
    upper_red1 = np.array([10, 255, 255])  # Upper hue range for red

    lower_red2 = np.array([170, 120, 70])  # Second range for red (wrapping around the hue spectrum)
    upper_red2 = np.array([180, 255, 255])

    # Create masks for red detection in both ranges
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

    # Combine the two masks
    mask = mask1 + mask2

    # Refine the mask (remove noise)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, np.ones((3, 3), np.uint8))

    # Invert the mask to get the non-red parts of the image
    mask_inverse = cv2.bitwise_not(mask)

    # Segment the red part (cloth) and replace it with the background
    res1 = cv2.bitwise_and(background, background, mask=mask)
    
    # Segment the rest of the frame
    res2 = cv2.bitwise_and(frame, frame, mask=mask_inverse)

    # Combine the background and current frame to create the final output
    final_output = cv2.addWeighted(res1, 1, res2, 1, 0)

    # Display the result
    cv2.imshow('Invisible Cloak - Red Cloth', final_output)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# close all windows
cap.release()
cv2.destroyAllWindows()
