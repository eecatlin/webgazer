import csv
import sys
import re

#Usage: csvreader.py <path_to_file> [Only Use First 3 Participants]

def main():
	print("Calculating average accuracy...")
	argLen = len(sys.argv)
	if argLen < 2:
		print("Please include the file you want to parse as an argument")
	pattern1 = re.compile('P_1//*')
	pattern2 = re.compile('P_12//*')
	pattern3 = re.compile('P_13//*')
	undef = re.compile('undefine*')

	with open(sys.argv[1], 'r') as csvfile:
	     reader = csv.reader(csvfile, delimiter=',', quotechar='|')
	     num = 0
	     total = 0 
	     for row in reader:
	     	if row[1] != 'NaN' and not undef.match(row[0]) and (argLen < 3 or pattern1.match(row[0]) or pattern2.match(row[0]) or pattern3.match(row[0])):
	     		num += 1
	     		total += float(row[1])
	     print("Average accuracy:")
	     print(total/float(num))


if __name__ == '__main__':
    main()