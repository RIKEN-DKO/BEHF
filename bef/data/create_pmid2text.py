import os
import json
import sys

def concatenate_json_files(directory_path):
    combined_data = {}

    # Iterate through all files in the directory
    for filename in os.listdir(directory_path):
        if filename.endswith(".json"):
            file_path = os.path.join(directory_path, filename)
            with open(file_path) as file:
                data = json.load(file)

            # Concatenate the values for each key
            for key, value in data.items():
                if key in combined_data:
                    combined_data[key] += " ".join(value)
                else:
                    combined_data[key] = " ".join(value)

    # Write the combined data to a new JSON file
    output_file = "pmid2text.json"
    with open(output_file, "w") as file:
        json.dump(combined_data, file, indent=2)

    print(f"Concatenated JSON data written to {output_file}")

# Check if the directory path is provided as a command-line argument
if len(sys.argv) < 2:
    print("Please provide the directory path as a command-line argument.")
else:
    directory_path = sys.argv[1]
    concatenate_json_files(directory_path)
