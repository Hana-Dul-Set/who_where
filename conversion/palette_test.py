import json

# Specify the path to your JSON file
json_file_path = 'color_palette.json'

# Open and read the JSON file
with open(json_file_path, 'r') as json_file:
    json_data = json.load(json_file)

# Ensure the loaded data is a list
if isinstance(json_data, list):
    print("JSON data as list:", json_data)
    print(len(json_data))
else:
    print("JSON data is not a list.")
