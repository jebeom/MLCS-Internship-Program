import os
import csv
import numpy as np

def read_csv_and_convert_to_2d_vector(input_file_path, output_dir):
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(input_file_path, 'r') as csv_file:
        reader = csv.reader(csv_file)
        row_index = 0
        for row in reader:
            # Split the row into label and image data
            label = row[0]
            image_data = list(map(int, row[1:]))

            # Convert the image data from 1D to 2D (assuming 28x28 images)
            image_2d = np.array(image_data).reshape(28, 28)

            # Create the output file name
            output_file_path = os.path.join(output_dir, f'#{label}-#{row_index}.csv')
            row_index += 1

            # Save the 2D vector to a new CSV file
            with open(output_file_path, 'w', newline='') as output_file:
                writer = csv.writer(output_file)
                writer.writerows(image_2d)

# Paths to the datasets
path_to_train_dataset_1 = os.path.join('datasets', 'mnist_train.csv')
path_to_train_dataset_2 = os.path.join('datasets', 'mnist_train_2.csv')
path_to_test_dataset = os.path.join('datasets', 'mnist_test.csv')

# Output directories
output_train_dir = os.path.join('outputs', 'train')
output_test_dir = os.path.join('outputs', 'test')

# Process the datasets
read_csv_and_convert_to_2d_vector(path_to_train_dataset_1, output_train_dir)
read_csv_and_convert_to_2d_vector(path_to_train_dataset_2, output_train_dir)
read_csv_and_convert_to_2d_vector(path_to_test_dataset, output_test_dir)

