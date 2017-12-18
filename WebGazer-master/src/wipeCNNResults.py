import csv
from time import gmtime, strftime

with open('CNNresults/results.csv', 'wb') as csvfile:
	wr = csv.writer(csvfile, dialect='excel')
	wr.writerow(['Wiped at: ' + str(strftime("%Y-%m-%d %H:%M:%S", gmtime()))])

with open('CNNresults/results.csv', 'a') as csvfile:
	wr = csv.writer(csvfile, dialect='excel')
	wr.writerow([
		'person','Average Test Distance', 
		'average y test distance', 
		'average x test distance',
		'average left eye y test distance',
		'average left eye x test distance',
		'average right eye y test distance',
		'average right eye x test distance',
		'average webGazer Distance', 
		'average y webGazer Distance', 
		'average x webGazer Distance',
		'training time',
		'training size',
		'testing size'])