import csv
import cv2
import numpy as np

lines = []
with open('./MyData/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

images = []
measurements = []
for line in lines:
    source_path = line[0]
    filename = source_path.split('/')[-1]
    current_path = './MyData/IMG/' + filename
    image = cv2.imread(current_path)
    #Steering measurement
    measurement = float(line[3])
    measurements.append(measurement)

