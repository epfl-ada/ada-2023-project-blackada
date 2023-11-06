import os
import pandas as pd

def _read_beer_data(file_path, num_lines=None):
    # Initialize an empty list to store each beer's information
    beer_data = []

    # Open the text file for reading
    with open(file_path, 'r') as file:
        # Initialize an empty dictionary for the current beer
        current_beer = {}
        # Initialize line counter
        line_counter = 0
        # Iterate over each line in the file
        for line in file:
            # Increment line counter
            line_counter += 1
            # Strip the line of leading/trailing whitespace
            line = line.strip()
            # If the line is empty or not a key-value pair, it means we're between beer entries
            if line == '' or ': ' not in line:
                if current_beer:
                    # Add the current beer's dictionary to the list and reset it
                    beer_data.append(current_beer)
                    current_beer = {}
            else:
                # Split the line into key and value, if possible
                parts = line.split(': ', 1)
                if len(parts) == 2:
                    key, value = parts
                    # Special handling for 'date' field to convert it into a readable format
                    if key == 'date':
                        value = pd.to_datetime(int(value), unit='s')
                    # Convert boolean string 'True'/'False' to a Python boolean
                    elif value == 'True':
                        value = True
                    elif value == 'False':
                        value = False
                    # Add the key-value pair to the current beer's dictionary
                    current_beer[key] = value
                else:
                    print(f"Line skipped: {line}")
            # Stop reading if num_lines is reached
            if num_lines and line_counter >= num_lines:
                break

        # Make sure to add the last beer's data if the file doesn't end with a newline
        if current_beer:
            beer_data.append(current_beer)

    # Create a DataFrame from the list of dictionaries
    df = pd.DataFrame(beer_data)
    return df

def load_beer_data(file_path, num_lines=None, save=False):
    # Check the file extension
    _, file_extension = os.path.splitext(file_path)
    
    if file_extension == '.txt':
        # Read the .txt file
        df = _read_beer_data(file_path, num_lines)
        
        # Save as .feather if save flag is True
        if save:
            feather_path = file_path.replace('.txt', '.feather')
            df.to_feather(feather_path)
            print(f"Data saved as {feather_path}")
            
        return df
        
    elif file_extension == '.feather':
        # Load the .feather file
        df = pd.read_feather(file_path)
        if num_lines:
            df = df.head(num_lines)
        print(f"Data loaded from {file_path}")
        return df
    
    else:
        raise ValueError("Unsupported file format. Only .txt and .feather files are supported.")
