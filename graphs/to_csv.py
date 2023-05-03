import pandas as pd
import json

with open("all_results_for_T_1.00_AR_0.990_MI_500_5.5_1.5_SEED1.txt", "r") as file:
    content = file.read()
    # Replace single quotes with double quotes
    content = content.replace("'", '"')

# Load the JSON data
data = json.loads(content)

# Create a pandas DataFrame from the list of dictionaries
df = pd.DataFrame(data)

# Save the DataFrame to a CSV file
df.to_csv("all_results.csv", index=False)

