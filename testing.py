import random

def split_list_into_sublists(input_list, num_sublists=5):
    # Randomly shuffle the input list
    random.shuffle(input_list)
    
    # Calculate the number of elements per sublist
    elements_per_sublist = len(input_list) // num_sublists
    
    # Calculate the remainder to distribute any remaining elements
    remainder = len(input_list) % num_sublists
    
    # Initialize the starting index for slicing
    start_index = 0
    
    # Initialize the list to store sublists
    sublists = []
    
    # Iterate through each sublist
    for i in range(num_sublists):
        # Calculate the ending index for slicing
        end_index = start_index + elements_per_sublist + (1 if i < remainder else 0)
        
        # Append the sublist to the result list
        sublists.append(input_list[start_index:end_index])
        
        # Update the starting index for the next iteration
        start_index = end_index
    
    return sublists

# Example usage:
my_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
result_sublists = split_list_into_sublists(my_list, 5)

print(result_sublists)
