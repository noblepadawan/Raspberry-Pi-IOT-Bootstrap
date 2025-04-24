# Imports
import mediapipe as mp
from picamera2 import Picamera2
import time
import cv2

# Initialize the pi camera
pi_camera = Picamera2()
# Convert the color mode to RGB
config = pi_camera.create_preview_configuration(main={"format": "RGB888"})
pi_camera.configure(config)

# Start the pi camera and give it a second to set up
pi_camera.start()
time.sleep(1)

def draw_pose(image, landmarks):
	''' 
	TODO Task 1
	
	Code to this fucntion to draw circles on the landmarks and lines
	connecting the landmarks then return the image.
	
	Use the cv2.line and cv2.circle functions.

	landmarks is a collection of 33 dictionaries with the following keys
		x: float values in the interval of [0.0,1.0]
		y: float values in the interval of [0.0,1.0]
		z: float values in the interval of [0.0,1.0]
		visibility: float values in the interval of [0.0,1.0]
		
	References:
	https://docs.opencv.org/4.x/dc/da5/tutorial_py_drawing_functions.html
	https://developers.google.com/mediapipe/solutions/vision/pose_landmarker
	'''

	# copy the image
	landmark_image = image.copy()
	
	# get the dimensions of the image
	height, width, _ = image.shape

    
	# Define drawing specifications for the landmarks (circles)
	landmark_drawing_spec = mp.solutions.drawing_utils.DrawingSpec(thickness=5, circle_radius=5, color=(149, 18, 132))
	# Define drawing specifications for the connections (lines) between the landmarks
	connection_drawing_spec = mp.solutions.drawing_utils.DrawingSpec(thickness=3, circle_radius=0, color=(185, 185, 23))

    # Use MediaPipe's drawing utility to draw the landmarks and connections on the image
	mp.solutions.drawing_utils.draw_landmarks(
		image=landmark_image,                               # The image to draw on
		landmark_list=landmarks,                            # The list of detected pose landmarks
		connections=mp.solutions.pose.POSE_CONNECTIONS,     # The predefined connections between pose landmarks
		landmark_drawing_spec=landmark_drawing_spec,        # Styling for landmarks
		connection_drawing_spec=connection_drawing_spec)    # Styling for connections
	
	return landmark_image       # Return the image with the pose landmarks drawn on it

def main():
	''' 
	TODO Task 2
		modify this fucntion to take a photo uses the pi camera instead 
		of loading an image

	TODO Task 3
		modify this function further to loop and show a video
	'''

	# Create a pose estimation model 
	mp_pose = mp.solutions.pose
	
	# start detecting the poses
	with mp_pose.Pose(
			min_detection_confidence=0.5,               # Minimum confidence threshold for pose detection to be considered successful
			min_tracking_confidence=0.5) as pose:       # Minimum confidence threshold for tracking the pose across frames
     
		while True: # Continuously capture and process frames
			# Capture a frame from the camera
			image = pi_camera.capture_array()

			# Display the raw camera feed
			cv2.imshow('Camera Feed', image)
   
			# Break the loop if 'q' key is pressed
			if cv2.waitKey(1) == ord('q'):
				break

			# To improve performance, optionally mark the image as not
            # writeable to pass by reference
			image.flags.writeable = False
   
			# Get the landmarks
			results = pose.process(image)
   
			# Draw the landmarks on the image
			if results.pose_landmarks:                                      # Check if any pose landmarks were detected
				annotated_image = draw_pose(image, results.pose_landmarks)  # Create a copy to draw on
				cv2.imshow('MediaPipe Pose', annotated_image)               # Display the image with the pose landmarks
				print("Pose Detected") # Indicate when a pose is detected   # Print to the console that a pose was detected
			else:
				print("No Pose Detected")                                   # Print to the console if no pose was detected

    # Stop the Raspberry Pi camera
	pi_camera.stop()
    # Close all OpenCV windows
	cv2.destroyAllWindows()
 
    



if __name__ == "__main__":
	main()
	print('done')
