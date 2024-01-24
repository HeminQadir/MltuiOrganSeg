
import os
import json
from utilities.helper import split_list_into_sublists

def create_json(root_directory, k_folds=5):
    # Get a list of files in "imagesTs" and "labelsTr"
    images_directory = os.path.join(root_directory, 'imagesTr')    
    labels_directory = os.path.join(root_directory, 'labelsTr')

    files_images = [os.path.join(images_directory, file) for file in os.listdir(images_directory) if os.path.isfile(os.path.join(images_directory, file)) and file.endswith('.nii.gz') and not file.startswith('._')]
    files_labels = [os.path.join(labels_directory, file) for file in os.listdir(labels_directory) if os.path.isfile(os.path.join(labels_directory, file)) and file.endswith('.nii.gz') and not file.startswith('._')]

    # Ensure the same number of files in images and labels
    assert len(files_images) == len(files_labels), "Mismatch in the number of image and label files"

    sublists = split_list_into_sublists(files_images, k_folds=k_folds)

    # Create a list to store the training data
    training_data = []

    # Iterate through each fold
    fold = 0
    for train_index in sublists:
        print(fold)
        print(train_index)
        print("*"*40)
        
        # Select the files for the current fold
        fold_files_images = [files_image for files_image in train_index]
        fold_files_labels = [files_label for files_label in train_index]

        # Iterate through each file in the current fold
        for file_images, file_labels in zip(fold_files_images, fold_files_labels):
            example_data = {"fold": fold, "image":  file_images, "label": file_labels}
            training_data.append(example_data)

        fold+=1

    # Create a dictionary with the "training" key and the training data list
    json_data = {"training": training_data}

    # Convert the dictionary to a JSON-formatted string
    json_string = json.dumps(json_data, indent=4)


    # Write the JSON string to a file
    with open("training_data_2.json", "w") as json_file:
        json_file.write(json_string)


if __name__ == "__main__":
    root_directory = "/media/jacobo/NewDrive/Medical_Dataset/Decath_Colon/Task10_Colon"  # Replace with the actual path to your root directory
    create_json(root_directory)

