import os
import json

# Path to the directory with files
directory_path = "/Users/shreyash/Downloads/privacy_stories_1.1/Annotations"  # Replace with your local directory path

# Function to categorize files based on names
def categorize_file(file_name: str) -> str:
    """Categorize the file based on its name."""
    if "inotification" in file_name.lower() or "database" in file_name.lower() or "data-struct" in file_name.lower() or "architecture" in file_name.lower():
        return "Architecture and Database Design Documents"
    elif "password" in file_name.lower() or "account" in file_name.lower() or "analytics" in file_name.lower() or "configuration" in file_name.lower() or "threepids" in file_name.lower():
        return "Code Specification Documents"
    elif "readme" in file_name.lower():
        return "README Files"
    else:
        return "User and Developer Guides"

# Dictionary to store classified files
file_dict = {}

# Iterate through files in the directory
for root, _, file_list in os.walk(directory_path):
    for file in file_list:
        if file.endswith(".txt"):  # Process only .txt files
            file_path = os.path.join(root, file)
            classification = categorize_file(file)
            
            # Create an entry in the dictionary for the classification
            if classification not in file_dict:
                file_dict[classification] = []
            
            # Add file details to the classification
            file_dict[classification].append({
                "filename": file,
                "filepath": file_path,
                "classification": classification
            })

# Save the dictionary to a JSON file
output_path = "./classified_files.json"
with open(output_path, "w") as f:
    json.dump(file_dict, f, indent=4)

print(f"Classification completed and saved to {output_path}!")
