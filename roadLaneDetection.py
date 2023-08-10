import cv2
import numpy as np


def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    masked_img = cv2.bitwise_and(img, mask)
    return masked_img


def draw_lines(img, lines, color=[255, 0, 255], thickness=5):
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)


def lane_detection(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Perform edge detection using Canny
    edges = cv2.Canny(blur, 50, 150)

    # Define a region of interest (ROI) for lane detection
    height, width = image.shape[:2]
    vertices = np.array([[(100, height), (width // 2 - 45, height // 2 + 60),
                          (width // 2 + 45, height // 2 + 60), (width - 100, height)]], dtype=np.int32)
    roi_img = region_of_interest(edges, vertices)

    # Perform Hough Transform to detect lines in the ROI
    lines = cv2.HoughLinesP(roi_img, rho=1, theta=np.pi / 180, threshold=20, minLineLength=20, maxLineGap=300)

    # Create a blank image to draw the lane lines on
    line_img = np.zeros_like(image)

    # Draw the detected lines on the blank image
    draw_lines(line_img, lines)

    # Combine the original image with the detected lane lines
    result = cv2.addWeighted(image, 0.8, line_img, 1.0, 0)

    return result


# Example usage for processing a video
video_path = r'C:\Users\Abubakkar sithik\PycharmProjects\pythonProject\Task-1\test_video.mp4'
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    processed_frame = lane_detection(frame)

    cv2.imshow('Lane Detection', processed_frame)

    # Press 'q' to exit the loop and close the window
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
