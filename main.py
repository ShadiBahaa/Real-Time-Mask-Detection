# importing openCV package for image processing
import cv2 as cv
# importing numpy package for mathematical operations on list
import numpy as np
# sklearn is a package that implements machine learning algorithms. for this project, Support Vector Classification
# is used for classifying images
from sklearn.svm import SVC
# For measuring the accuracy of prediction
from sklearn.metrics import accuracy_score
# Splits data set to a training one and testing one
from sklearn.model_selection import train_test_split
# Principal Component Analysis: reduces the dimensions of data
from sklearn.decomposition import PCA
# Configurations of the whole process
from Constants import *
# Objects of training algorithm, and component analyzer
svm = SVC()
pca = PCA(n_components=PCA_COMPONENTS)
# a function for taking images to build data set, it takes the file of data name that should be created


def take_images(file_name):
    # Capturing by laptop video camera
    capture = cv.VideoCapture(LAPTOP_CAMERA)
    # List that contains faces determined in each capture
    faces_list = []
    while True:
        # camera_works is a boolean variable that determines if the camera works
        # well or not while img is a numpy array that contains picture
        camera_works, img = capture.read()
        if camera_works:
            # Using the frontal_face file of HAAR features to determine faces in pictures
            haar_frontal_face = cv.CascadeClassifier(HAAR_ALGO_FILE)
            # Array of faces found in the picture
            faces = haar_frontal_face.detectMultiScale(img)
            # Iterating over each face and storing it in faces_list
            for topX, topY, width, height in faces:
                # Putting a rectangle around the face
                cv.rectangle(img, (topX, topY), (topX + width, topY + height), (COLOR_MAX, COLOR_MIN, COLOR_MAX),
                             THICKNESS)
                face = img[topY: topY + height, topX: topX + width, :]
                face = cv.resize(face, (RESIZING_WIDTH, RESIZING_HEIGHT))
                print(len(faces_list))
                if len(faces_list) < DATA_SIZE:
                    faces_list.append(face)
            # showing image on the camera with the rectangle
            cv.imshow('Working On Images...', img)
            # if a specific character pressed or the list is full, break
            if cv.waitKey(MIN_DELAY) == STOP_CHAR or len(faces_list) == DATA_SIZE:
                break
    # closing the camera and saving the data in a numpy file
    capture.release()
    cv.destroyAllWindows()
    np.save(str(file_name) + ".npy", faces_list)

# Reshaping data to two dimensions for more simplicity


def manipulate_file(file_name):
    file_data = np.load(str(file_name)+".npy")
    data_shape = file_data.shape
    file_data = file_data.reshape(data_shape[0], data_shape[1]*data_shape[2]*data_shape[3])
    return file_data

# training the model on the data collected by the camera, then comparing the predictions with the test results


def train_and_validate(data, labels):
    # dividing the dataset to training set and validation set
    training_data, test_data, training_result, test_result = train_test_split(data, labels,
                                                                              test_size=TEST_DATA_FRACTION)
    # fitting data to the dimensions of PCA
    training_data = pca.fit_transform(training_data)
    test_data = pca.transform(test_data)
    # training the data
    svm.fit(training_data, training_result)
    # predicting for test results and comparing with the actual results
    test_predictions = svm.predict(test_data)
    print(accuracy_score(test_result, test_predictions))

# testing the model with a real-life camera


def test():
    capture = cv.VideoCapture(LAPTOP_CAMERA)
    while True:
        camera_works, img = capture.read()
        if camera_works:
            haar_frontal_face = cv.CascadeClassifier(HAAR_ALGO_FILE)
            faces = haar_frontal_face.detectMultiScale(img)
            for topX, topY, width, height in faces:
                cv.rectangle(img, (topX, topY), (topX + width, topY + height), (COLOR_MAX, COLOR_MIN, COLOR_MAX),
                             THICKNESS)
                face = img[topY: topY + height, topX: topX + width, :]
                face = cv.resize(face, (RESIZING_WIDTH, RESIZING_HEIGHT))
                face = face.reshape(RESHAPING_1ST_FACTOR, RESHAPING_2ND_FACTOR)
                face = pca.transform(face)
                prediction = classification[int(svm.predict(face)[0])]
                cv.putText(img, prediction, (topX, topY), cv.FONT_HERSHEY_TRIPLEX, SCALE, (COLOR_MAX, COLOR_MAX,
                                                                                           COLOR_MAX), THICKNESS)
            cv.imshow('Result...', img)
            if cv.waitKey(MIN_DELAY) == STOP_CHAR:
                break
    capture.release()
    cv.destroyAllWindows()

# the main function


if __name__ == "__main__":
    # taking images with and without mask for creating data set
    take_images("without_masking")
    take_images("with_masking")
    # getting manipulated arrays to work on them
    without_masking = manipulate_file("without_masking")
    with_masking = manipulate_file("with_masking")
    # concatenating the two datasets into one
    all_data = np.r_[with_masking, without_masking]
    # setting the result as 0 for mask and 1 for no mask
    identifiers = np.zeros(all_data.shape[0])
    identifiers[all_data.shape[0]//2:] = 1
    # training, validation and testing of the model
    train_and_validate(all_data, identifiers)
    test()


