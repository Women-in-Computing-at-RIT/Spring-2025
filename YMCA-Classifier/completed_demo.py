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
    for letter, angles in TARGET_ANGLES.items():
        if all(
            abs(angle - target) <= TOLERANCE
            for angle, target in zip(
                [left_elbow, right_elbow, left_shoulder, right_shoulder],
                [angles["left_elbow"], angles["right_elbow"], angles["left_shoulder"], angles["right_shoulder"]],
            )
        ):
            return letter
    return "Unknown"

## Main Processing Loop
def main():
    # Load a model
    model = YOLO("yolo11n-pose.pt")  # load an official model (pose estimation)
    # model.fuse()  # optional (improves speed)

    # Open the webcam stream
    cap = cv2.VideoCapture(0)

    # Process each frame from the webcam stream
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Perform object detection
        results = model(frame, verbose = False)

        # Enumerate over results and their keypoints
        for result in results:
            keypoints = result.keypoints
            if keypoints is not None:
                person_keypoints = keypoints[0] # Modify for multiple people
                keypoints_xy = person_keypoints.xy[0]
                if len(keypoints_xy) > 10:  # Ensure there are enough keypoints
                    # Calculate angles for each elbow
                    left_elbow_angle = calculate_angle(keypoints_xy[5], keypoints_xy[7], keypoints_xy[9])
                    right_elbow_angle = calculate_angle(keypoints_xy[6], keypoints_xy[8], keypoints_xy[10])
                    # Calculate angles for each shoulder
                    left_shoulder_angle = calculate_angle((keypoints_xy[5][0], keypoints_xy[5][1]+1), keypoints_xy[5], keypoints_xy[7])
                    right_shoulder_angle = calculate_angle((keypoints_xy[6][0], keypoints_xy[6][1]+1), keypoints_xy[6], keypoints_xy[8])

                    # Determine the letter based on the angles
                    letter = determine_letter(left_elbow_angle, right_elbow_angle, left_shoulder_angle, right_shoulder_angle)

                    # Annotate the frame with the calculated angles and letter
                    annotator = Annotator(frame)
                    annotator.text((10, 30), f"Left Elbow Angle: {left_elbow_angle:.0f} degrees", txt_color=(0, 0, 0))
                    annotator.text((10, 60), f"Right Elbow Angle: {right_elbow_angle:.0f} degrees", txt_color=(0, 0, 0))
                    annotator.text((10, 90), f"Left Shoulder Angle: {left_shoulder_angle:.0f} degrees", txt_color=(0, 0, 0))
                    annotator.text((10, 120), f"Right Shoulder Angle: {right_shoulder_angle:.0f} degrees", txt_color=(0, 0, 0))
                    annotator.text((10, 150), f"Letter: {letter}", txt_color=(0, 0, 0))
                    frame = annotator.result()
        # Visualize the results
        annotated_frame = results[0][0].plot() # Modify for multiple people
        cv2.imshow("YMCA Detection with YOLOv11", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
