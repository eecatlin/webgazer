import csv
import sys

def main():
	print("Calculating average accuracy...")
	if len(sys.argv) < 2:
		print("Please include the file you want to parse as an argument")

	with open(sys.argv[1], 'r') as csvfile:
	     reader = csv.reader(csvfile, delimiter=',', quotechar='|')
	     num = 0
	     total = 0 
	     for row in reader:
	     	if row[1] != 'NaN':
	     		num += 1
	     		total += float(row[1])
	     print("Average accuracy:")
	     print(total/float(num))


if __name__ == '__main__':
    main()