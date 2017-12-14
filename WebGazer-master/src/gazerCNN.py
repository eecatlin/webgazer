import os
import csv
import cv2
import numpy as np
import tensorflow as tf

######################################################################################
##################################### Parameters #####################################
######################################################################################

# https://arxiv.org/pdf/1605.05258.pdf
# https://web.stanford.edu/class/cs231a/prev_projects_2016/eye-display-gaze-2.pdf

batchSz = 8
imgW = 50
imgH = 42
learnRate = 0.001

######################################################################################
################################## CNN Architecture ##################################
######################################################################################

def cnn(batch):
	conv1 = tf.layers.conv2d(
		inputs=batch,
        filters=24,
        kernel_size=[7, 7],
        padding="valid",
        activation=tf.nn.relu)
	pool1 = tf.layers.max_pooking2d(
		inputs=conv1,
		pool_size=[2, 2],
		strides=2)
	conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=24,
        kernel_size=[5, 5],
        padding="valid",
        activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(
    	inputs=conv2, 
    	pool_size=[2, 2], 
    	strides=2)
    conv3 = tf.layers.conv2d(
        inputs=pool2,
        filters=24,
        kernel_size=[3, 3],
        padding="valid",
        activation=tf.nn.relu)
    pool3 = tf.layers.max_pooling2d(
    	inputs=conv3, 
    	pool_size=[2, 2], 
    	strides=2)
    feature_vector = tf.layers.dense(
		inputs=pool3,
		units= ,
		activation=tf.nn.relu)

	return feature_vector

logits = tf.layers.dense()
distance = 
loss = 
train = tf.train.AdamOptimizer(learnRate).minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

######################################################################################
################################## Data Processing ###################################
######################################################################################

with open('/course/cs143/datasets/webgazer/framesdataset/train_1430_1.txt') as f:
    trainPaths = f.readlines()

with open('/course/cs143/datasets/webgazer/framesdataset/test_1430_1.txt') as f:
    testPaths = f.readlines()

for i in range(65):
	pTrainPaths = list(map(lambda x: x.replace("\\", "/").replace("\n", ""), filter(lambda x: x.split("\\")[0] == 'P_' + str(1), trainPaths)))
	pTestPaths = list(map(lambda x: x.replace("\\", "/").replace("\n", ""), filter(lambda x: x.split("/")[0] == 'P_' + str(i), testPaths)))

	if len(pTrainPaths) != 0 and len(pTestPaths) != 0:
		trainREyes = []
		trainLEyes = []
		trainEyes = []
		trainRLabelsY = []
		trainRLabelsX = []
		trainLLabelsY = []
		trainLLabelsX = []
		trainLabelsY = []
		trainLabelsX = []

		testREyes = []
		testLEyes = []
		testEyes = []
		testRLabelsY = []
		testRLabelsX = []
		testLLabelsY = []
		testLLabelsX = []
		testLabelsY = []
		testLabelsX = []

		for path in pTrainPaths:
			predPath = '/course/cs143/datasets/webgazer/framesdataset/' + path + '/gazePredictions.csv'

			with open(predPath) as f:
    			readCSV = csv.reader(f, delimiter=',')
    			for row in readCSV:

			        frameFilename = row[0]
			        frameTimestamp = row[1]
			        tobiiLeftEyeGazeX = float( row[2] )
			        tobiiLeftEyeGazeY = float( row[3] )
			        tobiiRightEyeGazeX = float( row[4] )
			        tobiiRightEyeGazeY = float( row[5] )
			        webgazerX = float( row[6] )
			        webgazerY = float( row[7] )
			        clmTracker = row[8:len(row)-1]
			        clmTracker = [float(i) for i in clmTracker]
			        clmTrackerInt = [int(i) for i in clmTracker]

			        tobiiEyeGazeX = (tobiiLeftEyeGazeX + tobiiRightEyeGazeX) / 2
			        tobiiEyeGazeY = (tobiiLeftEyeGazeY + tobiiRightEyeGazeY) / 2

			        lEyeMidY = clmTrackerInt[54]
			        lEyeMidX = clmTrackerInt[55]
			        rEyeMidY = clmTrackerInt[64]
			        rEyeMidX = clmTrackerInt[65]
			        lEyeCornerY = max(lEyeMidY - (imgH // 2), 0)
			        lEyeCornerX = max(lEyeMidX - (imgW // 2), 0)
			        rEyeCornerY = max(rEyeMidY - (imgH // 2), 0)
			        rEyeCornerX = max(rEyeMidX - (imgW // 2), 0)

			        image = cv2.imread('/course/cs143/datasets/webgazer/framesdataset/' + frameFilename[2:],0) / 255

			        lEye = image[lEyeCornerY:lEyeCornerY + imgH, lEyeCornerX:lEyeCornerX + imgW]
			        rEye = image[rEyeCornerY:lEyeCornerY + imgH, rEyeCornerX:rEyeCornerX + imgW]
			        bEye = np.concatenate((lEye, rEye), axis=1)

			        trainREyes.append(rEye)
			        trainLEyes.append(lEye)
			        trainEyes.append(bEye)
			        trainRLabelsY.append(tobiiRightEyeGazeY)
			        trainRLabelsX.append(tobiiRightEyeGazeX)
			        trainLLabelsY.append(tobiiLeftEyeGazeY)
			        trainLLabelsX.append(tobiiLeftEyeGazeX)
			        trainLabelsY.append(tobiiEyeGazeY)
			        trainLabelsX.append(tobiiEyeGazeX)

		for path in pTestPaths:
			predPath = '/course/cs143/datasets/webgazer/framesdataset/' + path + '/gazePredictions.csv'

			with open(predPath) as f:
    			readCSV = csv.reader(f, delimiter=',')
    			for row in readCSV:

			        frameFilename = row[0]
			        frameTimestamp = row[1]
			        tobiiLeftEyeGazeX = float( row[2] )
			        tobiiLeftEyeGazeY = float( row[3] )
			        tobiiRightEyeGazeX = float( row[4] )
			        tobiiRightEyeGazeY = float( row[5] )
			        webgazerX = float( row[6] )
			        webgazerY = float( row[7] )
			        clmTracker = row[8:len(row)-1]
			        clmTracker = [float(i) for i in clmTracker]
			        clmTrackerInt = [int(i) for i in clmTracker]

			        tobiiEyeGazeX = (tobiiLeftEyeGazeX + tobiiRightEyeGazeX) / 2
			        tobiiEyeGazeY = (tobiiLeftEyeGazeY + tobiiRightEyeGazeY) / 2

			        lEyeMidY = clmTrackerInt[54]
			        lEyeMidX = clmTrackerInt[55]
			        rEyeMidY = clmTrackerInt[64]
			        rEyeMidX = clmTrackerInt[65]
			        lEyeCornerY = max(lEyeMidY - (imgH // 2), 0)
			        lEyeCornerX = max(lEyeMidX - (imgW // 2), 0)
			        rEyeCornerY = max(rEyeMidY - (imgH // 2), 0)
			        rEyeCornerX = max(rEyeMidX - (imgW // 2), 0)

			        image = cv2.imread('/course/cs143/datasets/webgazer/framesdataset/' + frameFilename[2:],0) / 255

			        lEye = image[lEyeCornerY:lEyeCornerY + imgH, lEyeCornerX:lEyeCornerX + imgW]
			        rEye = image[rEyeCornerY:lEyeCornerY + imgH, rEyeCornerX:rEyeCornerX + imgW]
			        bEye = np.concatenate((lEye, rEye), axis=1)

			        testREyes.append(rEye)
			        testLEyes.append(lEye)
			        testEyes.append(bEye)
			        testRLabelsY.append(tobiiRightEyeGazeY)
			        testRLabelsX.append(tobiiRightEyeGazeX)
			        testLLabelsY.append(tobiiLeftEyeGazeY)
			        testLLabelsX.append(tobiiLeftEyeGazeX)
			        testLabelsY.append(tobiiEyeGazeY)
			        testLabelsX.append(tobiiEyeGazeX)

		trainNum = len(trainLabelsY)
		testNum = len(testLabelsY)

		# Left Eye Y
		for j in range(trainNum // batchSz):

			ses.run(train, feed_dict={img: imgs, ans: anss})

		# Left Eye X
		for j in range(trainNum // batchSz):

			ses.run(train, feed_dict={img: imgs, ans: anss})

		# Right Eye Y
		for j in range(trainNum // batchSz):

			ses.run(train, feed_dict={img: imgs, ans: anss})

		# Right Eye X
		for j in range(trainNum // batchSz):

			ses.run(train, feed_dict={img: imgs, ans: anss})



		for j in range(trainNum // batchSz):

			ses.run(distance, feed_dict={img: imgs, ans: anss})