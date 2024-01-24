import os
import json
from utilities.helper import split_list_into_sublists

def create_json(root_directory, k_folds=5):
    # Get a list of subdirectories in the root directory
    subfolders = [f.path for f in os.scandir(root_directory) if f.is_dir()]

    sublists = split_list_into_sublists(subfolders, k_folds=k_folds)

    # Create a list to store the training data
    training_data = []

    # Iterate through each fold
    #for fold, (train_index, _) in enumerate(kf.split(subfolders)):

    fold = 0
    for train_index in sublists:

        # Select the subdirectories for the current fold
        fold_subfolders = [subfolder for subfolder in train_index]

        # Iterate through each subdirectory in the current fold
        for subfolder in fold_subfolders:
            image_files = [
                f"{subfolder}/{os.path.basename(subfolder)}_{seq}.nii.gz"
                for seq in ["flair", "t1ce", "t1", "t2"]
                          ]
            label_file = f"{subfolder}/{os.path.basename(subfolder)}_seg.nii.gz"

            example_data = {"fold": fold, "image": image_files, "label": label_file}
            training_data.append(example_data)

        fold+=1

    # Create a dictionary with the "training" key and the training data list
    json_data = {"training": training_data}

    # Convert the dictionary to a JSON-formatted string
    json_string = json.dumps(json_data, indent=4)

    # Write the JSON string to a file
    with open("training_data.json", "w") as json_file:
        json_file.write(json_string)


if __name__ == "__main__":
    root_directory = "/media/jacobo/NewDrive/Hemin_Collection/BraTS2021/TrainingData"  # Replace with the actual path to your root directory
    create_json(root_directory)
