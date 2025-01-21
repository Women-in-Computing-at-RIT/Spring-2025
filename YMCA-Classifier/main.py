'''
Use a pretrained yolov11 pose model to annotate a webcam stream.
Calculate the angle of each figures' arms to determine a 'Y', 'M', 'C' or 'A' shape 
Based off of ultralytics provided solutions https://github.com/ultralytics/ultralytics/blob/main/ultralytics/solutions/ai_gym.py

Audrey Fuller 
WiC Projects - Spring 2025
'''

from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
import cv2
import keyboard

# Initialize angle threshold constants
Y_THRESHOLD = 30  # degrees
M_THRESHOLD = 30  # degrees
C_THRESHOLD = 30  # degrees
A_THRESHOLD = 30  # degrees

## Main Processing Loop
def main():
    # Load a model
    model = YOLO("yolo11n-pose.pt")  # load an official model (pose estimation)
    model.fuse()  # optional (improves speed)
    # model.info()  # display model information

    # Open the webcam stream
    cap = cv2.VideoCapture(0)

    # Process each frame from the webcam stream
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Perform object detection
        results = model(frame)

        # Visualize the results
        annotated_frame = results[0].plot()
        cv2.imshow("YOLOv11 Object Detection", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

        # Process the frame for pose estimation and YMCA classification
        #process_frame(frame, model)

    # Initialize lists of angles for each person in the frame

    # Predict poses for current frame with the model
    # results = model()

    # Extract and check keypoints 

    # Initialize annotator

    # Enumerate over keypoints

        # Get keypoints and estimate the angle

        # Determine YMCA letter logic based on angle thresholds

        # Annotate keypoints, connections, angles, and letter text


    # Display annotated frames


if __name__ == '__main__':
    main()
