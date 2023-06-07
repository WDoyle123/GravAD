import pandas as pd
import json
import ast
import re
import glob

def process_files():
    file_pattern = "test_graphs/all_results_for_T*.txt"
    file_list = glob.glob(file_pattern)

    def custom_parser(data):
        pattern = r"Array\(([\d.e+-]+),.*?\)"
        matches = re.findall(pattern, data)
        for match in matches:
            data = data.replace(f'Array({match}, dtype=float32)', match)
            data = data.replace(f'Array({match}, dtype=float64)', match)
            data = data.replace(f'Array({match}, dtype=float64, weak_type=True)', match)
        return ast.literal_eval(data)

    def remove_device(data):
        # Use regex to replace 'Device' with nothing
        data = re.sub(r'Device', '', data)
        return data

    for file_name in file_list:
        with open(file_name, "r") as file:
            content = file.read()
            # Replace single quotes with double quotes
            content = content.replace("'", '"')

        # Remove 'Device' from the content
        content = remove_device(content)

        # Load the data
        data = custom_parser(content)

        # Create a pandas DataFrame from the list of dictionaries
        df = pd.DataFrame(data)

        # Save the DataFrame to a CSV file
        df.to_csv("results.csv", index=False)

