# Import libraries
# ==================================================
import sys

import numpy as np
import cv2
import matplotlib.pyplot as plt
# ==================================================

# Global variables
# ==================================================
cascPath = 'haarcascade_frontalface_default.xml'
# ==================================================

# Functions
# ==================================================
def test1():
	# Get user supplied values
	imagePath = sys.argv[1]
	# Create the haar cascade
	faceCascade = cv2.CascadeClassifier(cascPath)
	# Read the image
	image = cv2.imread(imagePath, 0)
	gray = image #cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	# Detect faces in the image
	faces = faceCascade.detectMultiScale(
	    gray,
	    scaleFactor=1.1,
	    minNeighbors=5,
	    minSize=(30, 30),
	    #flags = cv2.cv.CV_HAAR_SCALE_IMAGE
	)
	print ("Found {0} faces!".format(len(faces)))

	# Draw a rectangle around the faces
	crop_faces = []
	for (x, y, w, h) in faces:
	    #cv2.rectangle(image, (int(x-0.2*w), int(y-0.5*h)), (int(x+1.2*w), int(y+1.5*h)), (0, 255, 0), 2)
		crop_img = image[int(y-0.5*h):int(y+1.5*h), int(x-0.2*w):int(x+1.2*w)] # Crop from x, y, w, h -> 100, 200, 300, 400
		# NOTE: its img[y: y + h, x: x + w] and *not* img[x: x + w, y: y + h]
		crop_faces.append(crop_img)
		#plt.imshow(crop_img, cmap = 'gray', interpolation = 'bicubic')
		#plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
		#plt.show()


	#cv2.imshow("Faces found" ,image)
	#k = cv2.waitKey(0)
	#if k == 27:
	#	cv2.destroyAllWindows()
	
	if len(crop_faces) > 0:
		cv2.imwrite('crop_'+imagePath, crop_faces[0])
		#plt.imshow(crop_faces[0], cmap = 'gray', interpolation = 'bicubic')
		#plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
		#plt.show()
	
def test2():
	faceCascade = cv2.CascadeClassifier(cascPath)

	video_capture = cv2.VideoCapture(0)

	while True:
		# Capture frame-by-frame
		ret, frame = video_capture.read()

		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		faces = faceCascade.detectMultiScale(
		    gray,
		    scaleFactor=1.1,
		    minNeighbors=5,
		    minSize=(30, 30),
		    #flags=cv2.cv.CV_HAAR_SCALE_IMAGE
		)

		# Draw a rectangle around the faces
		for (x, y, w, h) in faces:
		    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

		# Display the resulting frame
		cv2.imshow('Video', frame)

		if cv2.waitKey(1) & 0xFF == ord('q'):
		    break

	# When everything is done, release the capture
	video_capture.release()
	cv2.destroyAllWindows()

def test3():
	cap = cv2.VideoCapture(0)
	cap.set(cv2.CAP_FFMPEG,True)
	cap.set(cv2.CAP_PROP_FPS,30)
	print (cap)
	print (cap.isOpened())

	while(True):
		if not cap.isOpened():
			cap.open(0)
		
		# Capture frame-by-frame
		ret, frame = cap.read()
		print (ret)
		print (frame)

		# Our operations on the frame come here
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		# Display the resulting frame
		cv2.imshow('frame',gray)
		if cv2.waitKey(1) & 0xFF == ord('q'):
		    break

	# When everything done, release the capture
	cap.release()
	cv2.destroyAllWindows()

# ==================================================

# Main
# ==================================================
if __name__ == '__main__':
	test3()
# ==================================================
