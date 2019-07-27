import glob
import csv

image_names = []
for infile in glob.glob("IMG/center*.jpg"):
    image_name = infile.split('/')[-1]
    image_name = image_name.split('.')
    image_names.append(image_name[0])
    
print(image_names)

print(len(image_found), len(image_not_found), len(image_names))

with open('driving_log.csv','r') as file, open ('clean_log.csv', 'w', newline='') as outfile:
    datareader = csv.reader(file,delimiter=',')
    datawriter = csv.writer(outfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
    driving_log = []
    for row in datareader:
        image_name = row[0].split('\\')[-1]
        image_name = image_name.split('.')[0]
        if image_name not in image_not_found:
              datawriter.writerow(row)
                
        else:
            print('skip', image_name)
			
			
			
