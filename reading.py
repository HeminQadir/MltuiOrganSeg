import pickle

file_path = '/home/jacobo/MultiOrganSeg/data.pkl'  # Replace 'your_file.pkl' with the actual path to your pickle file

try:
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
        print("Content of the pickle file:")
        print(data)
except FileNotFoundError:
    print(f"File not found: {file_path}")
except Exception as e:
    print(f"An error occurred: {e}")
