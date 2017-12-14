import os
import csv
import cv2
import numpy as np
import tensorflow as tf
from time import gmtime, strftime

with open('CNNresults/results.csv', 'a') as csvfile:
	wr = csv.writer(csvfile, dialect='excel')
	wr.writerow(['Time Stamp:', strftime("%Y-%m-%d %H:%M:%S", gmtime())])

######################################################################################
##################################### Parameters #####################################
######################################################################################

# https://arxiv.org/pdf/1605.05258.pdf
# https://web.stanford.edu/class/cs231a/prev_projects_2016/eye-display-gaze-2.pdf

batchSz = 8
imgW = 50
imgH = 42
epochNum = 1
learnRate = 0.001

######################################################################################
################################## CNN Architecture ##################################
######################################################################################

imgBatch = tf.placeholder(tf.float32, [batchSz, imgH, imgW, 1])
labels = tf.placeholder(tf.float32, [batchSz, 1])

def cnn(batch):
	conv1 = tf.layers.conv2d(
		inputs=batch,
		filters=24,
		kernel_size=[7, 7],
		padding="valid",
		activation=tf.nn.relu)
	pool1 = tf.layers.max_pooling2d(
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

	flatten = tf.reshape(pool3, [batchSz, 2 * 2 * 3 * 12])
	feature_vector = tf.layers.dense(
		inputs=flatten,
		units=4096,
		activation=tf.nn.relu)

	return feature_vector

logits = tf.layers.dense(inputs=cnn(imgBatch), units=1)
coordinates = logits
loss = tf.reduce_mean(tf.square(tf.subtract(logits, labels)))
train = tf.train.AdamOptimizer(learnRate).minimize(loss)

sess = tf.Session()
saver = tf.train.Saver()

######################################################################################
################################## Data Processing ###################################
######################################################################################

peopleProcessed = 0
framesProcessed = 0
testsProcessed = 0
framesSetup = 0

with open('/course/cs143/datasets/webgazer/framesdataset/train_1430_1.txt') as f:
	trainPaths = f.readlines()

with open('/course/cs143/datasets/webgazer/framesdataset/test_1430_1.txt') as f:
	testPaths = f.readlines()

for i in range(65):
	pTrainPaths = list(map(lambda x: x.replace("\\", "/").replace("\n", ""), filter(lambda x: x.split("\\")[0] == 'P_' + str(i), trainPaths)))
	pTestPaths = list(map(lambda x: x.replace("\\", "/").replace("\n", ""), filter(lambda x: x.split("\\")[0] == 'P_' + str(i), testPaths)))

	if len(pTrainPaths) != 0 and len(pTestPaths) != 0:
		peopleProcessed += 1
		if peopleProcessed % 1 == 0:
			print('People Processed:', peopleProcessed)

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

		testGazerY = []
		testGazerX = []

		print('Building Data')

		print('Building Training Data')

		for path in pTrainPaths:
			predPath = '/course/cs143/datasets/webgazer/framesdataset/' + path + '/gazePredictions.csv'

			with open(predPath) as f:
				readCSV = csv.reader(f, delimiter=',')
				for row in readCSV:
					framesProcessed += 1
					if framesProcessed % 5000 == 0:
						print('Collected data from %d frames' % framesProcessed)

					frameFilename = row[0]
					frameTimestamp = row[1]
					tobiiLeftEyeGazeX = float( row[2] )
					tobiiLeftEyeGazeY = float( row[3] )
					tobiiRightEyeGazeX = float( row[4] )
					tobiiRightEyeGazeY = float( row[5] )
					webgazerX = float( row[6] )
					webgazerY = float( row[7] )
					clmTracker = row[8:len(row)-1]
					clmTracker = [float(m) for m in clmTracker]
					clmTrackerInt = [int(m) for m in clmTracker]

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
					rEye = image[rEyeCornerY:rEyeCornerY + imgH, rEyeCornerX:rEyeCornerX + imgW]
					try:
						bEye = np.concatenate((lEye, rEye), axis=1)
					except:
						print('Eye Concat Error')
						print(np.shape(lEye))
						print(np.shape(rEye))
						cv2.imshow('image', lEye)
						cv2.waitKey(0)
						cv2.imshow('image', rEye)
						cv2.waitKey(0)

					trainREyes.append(rEye)
					trainLEyes.append(lEye)
					trainEyes.append(bEye)
					trainRLabelsY.append(tobiiRightEyeGazeY)
					trainRLabelsX.append(tobiiRightEyeGazeX)
					trainLLabelsY.append(tobiiLeftEyeGazeY)
					trainLLabelsX.append(tobiiLeftEyeGazeX)
					trainLabelsY.append(tobiiEyeGazeY)
					trainLabelsX.append(tobiiEyeGazeX)

		print('Building Testing Data')

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
					clmTracker = [float(m) for m in clmTracker]
					clmTrackerInt = [int(m) for m in clmTracker]

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
					rEye = image[rEyeCornerY:rEyeCornerY + imgH, rEyeCornerX:rEyeCornerX + imgW]
					try:
						bEye = np.concatenate((lEye, rEye), axis=1)
					except:
						print('Eye Concat Error')
						print(np.shape(lEye))
						print(np.shape(rEye))
						cv2.imshow('image', lEye)
						cv2.waitKey(0)
						cv2.imshow('image', rEye)
						cv2.waitKey(0)

					testREyes.append(rEye)
					testLEyes.append(lEye)
					testEyes.append(bEye)
					testRLabelsY.append(tobiiRightEyeGazeY)
					testRLabelsX.append(tobiiRightEyeGazeX)
					testLLabelsY.append(tobiiLeftEyeGazeY)
					testLLabelsX.append(tobiiLeftEyeGazeX)
					testLabelsY.append(tobiiEyeGazeY)
					testLabelsX.append(tobiiEyeGazeX)

					testGazerY.append(webgazerY)
					testGazerX.append(webgazerX)

		trainNum = len(trainLabelsY)
		testNum = len(testLabelsY)

		####################################### Training #######################################

		print('Training')

		# Left Eye Y
		print('Training Left Eye Y')
		sess.run(tf.global_variables_initializer())

		for l in range(epochNum):
			zipped = list(zip(trainREyes, trainLEyes, trainRLabelsY, trainRLabelsX, trainLabelsY, trainLLabelsX))
			rand.shuffle(zipped)
			trainREyes, trainLEyes, trainRLabelsY, trainRLabelsX, trainLabelsY, trainLLabelsX = zip(*zipped)

			for j in range(trainNum // batchSz):
				framesProcessed += 1
				if framesProcessed % 1000 == 0:
					print('Frames Batches Processed:', framesProcessed)

				x = []
				for k in range(j * batchSz, (j + 1) * batchSz):
					x += [trainLEyes[k]]
				x = np.array(x).reshape(batchSz, imgH, imgW, 1)
				y = np.array(trainLLabelsY[j * batchSz:(j + 1) * batchSz]).reshape(batchSz, 1)
				debugloss, logs, _ = sess.run([loss, logits, train], feed_dict={imgBatch: x, labels: y})
				if framesProcessed % 100 == 0:
					print('debug loss:', debugloss)
					print(logs)

		saver.save(sess, "CNNmodels/leftEyeY.ckpt")

		# Left Eye X
		print('Training Left Eye X')
		sess.run(tf.global_variables_initializer())

		for l in range(epochNum):
			zipped = list(zip(trainREyes, trainLEyes, trainRLabelsY, trainRLabelsX, trainLabelsY, trainLLabelsX))
			rand.shuffle(zipped)
			trainREyes, trainLEyes, trainRLabelsY, trainRLabelsX, trainLabelsY, trainLLabelsX = zip(*zipped)

			for j in range(trainNum // batchSz):
				framesProcessed += 1
				if framesProcessed % 1000 == 0:
					print('Frames Batches Processed:', framesProcessed)

				x = []
				for k in range(j * batchSz, (j + 1) * batchSz):
					x += [trainLEyes[k]]
				x = np.array(x).reshape(batchSz, imgH, imgW, 1)
				y = np.array(trainLLabelsX[j * batchSz:(j + 1) * batchSz]).reshape(batchSz, 1)
				sess.run(train, feed_dict={imgBatch: x, labels: y})

		saver.save(sess, "CNNmodels/leftEyeX.ckpt")

		# Right Eye Y
		print('Training Right Eye Y')
		sess.run(tf.global_variables_initializer())

		for l in range(epochNum):
			zipped = list(zip(trainREyes, trainLEyes, trainRLabelsY, trainRLabelsX, trainLabelsY, trainLLabelsX))
			rand.shuffle(zipped)
			trainREyes, trainLEyes, trainRLabelsY, trainRLabelsX, trainLabelsY, trainLLabelsX = zip(*zipped)

			for j in range(trainNum // batchSz):
				framesProcessed += 1
				if framesProcessed % 1000 == 0:
					print('Frames Batches Processed:', framesProcessed)

				x = []
				for k in range(j * batchSz, (j + 1) * batchSz):
					x += [trainREyes[k]]
				x = np.array(x).reshape(batchSz, imgH, imgW, 1)
				y = np.array(trainRLabelsY[j * batchSz:(j + 1) * batchSz]).reshape(batchSz, 1)
				sess.run(train, feed_dict={imgBatch: x, labels: y})

		saver.save(sess, "CNNmodels/rightEyeY.ckpt")

		# Right Eye X
		print('Training Right Eye X')
		sess.run(tf.global_variables_initializer())

		for l in range(epochNum):
			zipped = list(zip(trainREyes, trainLEyes, trainRLabelsY, trainRLabelsX, trainLabelsY, trainLLabelsX))
			rand.shuffle(zipped)
			trainREyes, trainLEyes, trainRLabelsY, trainRLabelsX, trainLabelsY, trainLLabelsX = zip(*zipped)
			
			for j in range(trainNum // batchSz):
				framesProcessed += 1
				if framesProcessed % 1000 == 0:
					print('Frames Batches Processed:', framesProcessed)

				x = []
				for k in range(j * batchSz, (j + 1) * batchSz):
					x += [trainREyes[k]]
				x = np.array(x).reshape(batchSz, imgH, imgW, 1)
				y = np.array(trainRLabelsX[j * batchSz:(j + 1) * batchSz]).reshape(batchSz, 1)
				sess.run(train, feed_dict={imgBatch: x, labels: y})

		saver.save(sess, "CNNmodels/rightEyeX.ckpt")

		####################################### Testing #######################################

		print('Testing')

		LeftY = []
		LeftX = []
		RightY = []
		RightX = []

		# Left Eye Y Test
		saver.restore(sess, "CNNmodels/leftEyeY.ckpt")

		for j in range(testNum // batchSz):
			testsProcessed += 1
			if testsProcessed % 1000 == 0:
				print('Frames Batches Processed:', testsProcessed)

			x = []
			for k in range(j * batchSz, (j + 1) * batchSz):
				x += [testLEyes[k]]
			x = np.array(x).reshape(batchSz, imgH, imgW, 1)
			y = np.array(testLLabelsY[j * batchSz:(j + 1) * batchSz]).reshape(batchSz, 1)
			LY = sess.run(coordinates, feed_dict={imgBatch: x, labels: y})
			LY = np.ndarray.tolist(np.reshape(LY, [batchSz]))
			LeftY += LY

		# Left Eye X Test
		saver.restore(sess, "CNNmodels/leftEyeX.ckpt")

		for j in range(testNum // batchSz):
			x = []
			for k in range(j * batchSz, (j + 1) * batchSz):
				x += [testLEyes[k]]
			x = np.array(x).reshape(batchSz, imgH, imgW, 1)
			y = np.array(testLLabelsX[j * batchSz:(j + 1) * batchSz]).reshape(batchSz, 1)
			LX = sess.run(coordinates, feed_dict={imgBatch: x, labels: y})
			LX = np.ndarray.tolist(np.reshape(LX, [batchSz]))
			LeftX += LX

		# Right Eye Y Test
		saver.restore(sess, "CNNmodels/rightEyeY.ckpt")

		for j in range(testNum // batchSz):
			x = []
			for k in range(j * batchSz, (j + 1) * batchSz):
				x += [testREyes[k]]
			x = np.array(x).reshape(batchSz, imgH, imgW, 1)
			y = np.array(testRLabelsY[j * batchSz:(j + 1) * batchSz]).reshape(batchSz, 1)
			RY = sess.run(coordinates, feed_dict={imgBatch: x, labels: y})
			RY = np.ndarray.tolist(np.reshape(RY, [batchSz]))
			RightY += RY

		# Right Eye X Test
		saver.restore(sess, "CNNmodels/rightEyeX.ckpt")

		for j in range(testNum // batchSz):
			x = []
			for k in range(j * batchSz, (j + 1) * batchSz):
				x += [testREyes[k]]
			x = np.array(x).reshape(batchSz, imgH, imgW, 1)
			y = np.array(testRLabelsX[j * batchSz:(j + 1) * batchSz]).reshape(batchSz, 1)
			RX = sess.run(coordinates, feed_dict={imgBatch: x, labels: y})
			RX = np.ndarray.tolist(np.reshape(RX, [batchSz]))
			RightX += RX


		resultNum = len(LeftY)

		trueY = testLabelsY[:resultNum]
		trueX = testLabelsX[:resultNum]

		trueLeftY = testLLabelsY[:resultNum]
		trueLeftX = testLLabelsX[:resultNum]
		trueRightY = testRLabelsY[:resultNum]
		trueRightX = testRLabelsX[:resultNum]

		gazerY = testGazerY[:resultNum]
		gazerX = testGazerX[:resultNum]

		testY = list(map(lambda x, y: float(x + y) / 2, LeftY, RightY))
		testX = list(map(lambda x, y: float(x + y) / 2, LeftX, RightX))

		gazerDists = list(map(lambda x, y, z, w: ((x - z) ** 2 + (y - w) ** 2) ** 0.5, gazerY, gazerX, trueY, trueX))

		gazerYDists = list(map(lambda x, y: abs(x - y), gazerY, trueY))
		gazerXDists = list(map(lambda x, y: abs(x - y), gazerX, trueX))

		testDists = list(map(lambda x, y, z, w: ((x - z) ** 2 + (y - w) ** 2) ** 0.5, testY, testX, trueY, trueX))

		testYDists = list(map(lambda x, y: abs(x - y), testY, trueY))
		testXDists = list(map(lambda x, y: abs(x - y), testX, trueX))

		testLeftYDists = list(map(lambda x, y: abs(x - y), LeftY, trueLeftY))
		testLeftXDists = list(map(lambda x, y: abs(x - y), LeftX, trueLeftX))
		testRightYDists = list(map(lambda x, y: abs(x - y), RightY, trueRightY))
		testRightXDists = list(map(lambda x, y: abs(x - y), RightX, trueRightX))



		avgGazerDist = sum(gazerDists) / float(resultNum)
		avgGazerYDist = sum(gazerYDists) / float(resultNum)
		avgGazerXDist = sum(gazerXDists) / float(resultNum)

		avgTestDist = sum(testDists) / float(resultNum)
		avgTestYDist = sum(testYDists) / float(resultNum)
		avgTestXDist = sum(testXDists) / float(resultNum)

		avgTestLeftYDist = sum(testLeftYDists) / float(resultNum)
		avgTestLeftXDist = sum(testLeftXDists) / float(resultNum)
		avgTestRightYDist = sum(testRightYDists) / float(resultNum)
		avgTestRightXDist = sum(testRightXDists) / float(resultNum)

		print('~')
		print('Person %d Results:' % i)
		print('Average Distance:', avgTestDist)
		print('Average Webgazer Distance:', avgGazerDist)
		print('Average X Distance:', avgTestXDist)
		print('Average Y Distance:', avgTestYDist)
		print('Average Left X Distance:', avgTestLeftXDist)
		print('Average Right X Distance:', avgTestRightXDist)
		print('Average Left Y Distance:', avgTestLeftYDist)
		print('Average Right Y Distance:', avgTestRightYDist)
		print('~')

		with open('CNNresults/results.csv', 'a') as csvfile:
			wr = csv.writer(csvfile, dialect='excel')
			wr.writerow([
				i, 
				avgTestDist, 
				avgTestYDist, 
				avgTestXDist,
				avgTestLeftYDist,
				avgTestLeftXDist,
				avgTestRightYDist,
				avgTestRightXDist,
				avgGazerDist, 
				avgGazerYDist, 
				avgGazerXDist])

