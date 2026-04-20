import csv
import glob

file_pattern = 'sensor_data_*.csv'
files = sorted(glob.glob(file_pattern))
output_file = 'merged_sensor_data.csv'

with open(output_file, 'w', newline='') as fout:
    writer = csv.writer(fout)
    for i, fname in enumerate(files):
        with open(fname, 'r') as fin:
            reader = csv.reader(fin)
            header = next(reader)
            # Only write the header for the first file
            if i == 0:
                writer.writerow(header)
            # Write all data rows
            for row in reader:
                writer.writerow(row)
print("Done!")