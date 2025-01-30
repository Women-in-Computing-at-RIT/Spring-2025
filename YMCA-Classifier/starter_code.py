'''
Use a pretrained yolov11 pose model to annotate a webcam stream.
Calculate the angle of each figures' arms to determine a 'Y', 'M', 'C' or 'A' shape 
Based off of ultralytics provided solutions https://github.com/ultralytics/ultralytics/blob/main/ultralytics/solutions/ai_gym.py

Audrey Fuller 
WiC Projects - Spring 2025

Possible Improvements:
- Play around with the angle and tolerance values to be more accurate
- Add support for multiple people
- Display letters and angles reletive to the person's position
- Average keypoint values over multiple frames for more stable results
- Add support for more letters
'''

# Libraries for YOLO, OpenCV, and math
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
import cv2
import math

# Initialize angle threshold constants in degrees
TARGET_ANGLES = {
    "Y": {"left_elbow": 180, "right_elbow": 180, "left_shoulder": 135, "right_shoulder": 135},
    "M": {"left_elbow": 90, "right_elbow": 90, "left_shoulder": 135, "right_shoulder": 135},
    "C": {"left_elbow": 90, "right_elbow": 90, "left_shoulder": 0, "right_shoulder": 180},
    "A": {"left_elbow": 180, "right_elbow": 180, "left_shoulder": 180, "right_shoulder": 180},
}
TOLERANCE = 40

def calculate_angle(p1, p2, p3):
    # Calculate the angle between three points
    # p1, p2, p3 are tuples (x, y)
    a = ((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2) ** 0.5
    b = ((p3[0] - p2[0]) ** 2 + (p3[1] - p2[1]) ** 2) ** 0.5
    c = ((p3[0] - p1[0]) ** 2 + (p3[1] - p1[1]) ** 2) ** 0.5

    # Handle edge cases
    if a == 0 or b == 0:
        return 0.0
    if abs((a**2 + b**2 - c**2) / (2 * a * b)) > 1.0:
        return 0.0
    
    # Calculate angle in degrees
    angle = math.degrees(math.acos((a**2 + b**2 - c**2) / (2 * a * b)))
    return angle

def determine_letter(left_elbow, right_elbow, left_shoulder, right_shoulder):
    # Iterate through each letter and check if the angles match within tolerance
    # Return the letter if found, otherwise return "Unknown"
    for letter, angles in TARGET_ANGLES.items():
        pass

## Main Processing Loop (We'll implement this in the demo)
def main():
    pass
    # Load a model

    # Open the webcam stream

    # Process each frame from the webcam stream

        # Perform object detection with our model

        # Enumerate over results

            # Enumerate over keypoints

                # Ensure there are enough keypoints
                    # Calculate angles for each elbow

                    # Calculate angles for each shoulder

                    # Determine the letter based on the angles

                    # Annotate the frame with the calculated angles and letter

        # Visualize the results

        # Break the loop if 'q' is pressed

    # Release the webcam and close windows

if __name__ == '__main__':
    main()
