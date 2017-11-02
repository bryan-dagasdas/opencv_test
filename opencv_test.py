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
orig_images_folder = 'orig_images'
face_images_folder = 'face_images'
# ==================================================

# Functions
# ==================================================
def test1():
	# Get user supplied values
	imageFile = sys.argv[1]
	imagePath = orig_images_folder + '/' + imageFile
	# Create the haar cascade
	faceCascade = cv2.CascadeClassifier(cascPath)
	# Read the image
	image = cv2.imread(imagePath, 0)
	#gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	# Detect faces in the image
	faces = faceCascade.detectMultiScale(
	    image, #gray,
	    scaleFactor=1.1,
	    minNeighbors=5,
	    minSize=(30, 30),
	    #flags = cv2.cv.CV_HAAR_SCALE_IMAGE
	)
	print ("Found {0} faces!".format(len(faces)))

	# Draw a rectangle around the faces
	crop_faces = []
	i = 0
	for (x, y, w, h) in faces:
	    #cv2.rectangle(image, x, y, w, h, (0, 255, 0), 2)
		imgW, imgH = image.shape
		x0 = int(x-0.2*w) if int(x-0.2*w) > 0 else 0
		y0 = int(y-0.5*h) if int(y-0.5*h) > 0 else 0
		x1 = min([imgH-1,int(x+1.2*w)])
		y1 = min([imgW-1,int(y+1.5*h)])
		crop_img = image[y0:y1, x0:x1]
		#crop_img = image[int(y-0.5*h):int(y+1.5*h), int(x-0.2*w):int(x+1.2*w)] # Crop from x, y, w, h -> 100, 200, 300, 400
		#print ((x, y, w, h))
		#print (image.shape)
		#print (crop_img.shape)
		# NOTE: its img[y: y + h, x: x + w] and *not* img[x: x + w, y: y + h]
		crop_faces.append(crop_img)
		i = i + 1
		face_image_path = face_images_folder + '/face_' + str(i) + '_' + imageFile
		cv2.imwrite(face_image_path, crop_img)
	
	for (x, y, w, h) in faces:
		cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
		#plt.imshow(crop_img, cmap = 'gray', interpolation = 'bicubic')
		#plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
		#plt.show()
	
	orig_image_path = face_images_folder + '/' + imageFile
	cv2.imwrite(orig_image_path, image)
		
	#cv2.imshow("Faces found" ,image)
	#k = cv2.waitKey(0)
	#if k == 27:
	#	cv2.destroyAllWindows()
	
	#if len(crop_faces) > 0:
		#cv2.imwrite('crop_'+imagePath, crop_faces[0])
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
	test1()
# ==================================================
