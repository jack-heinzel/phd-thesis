import os
import sys


def check_files_in_directory(directory):

    bad_names = False
    # List all files in the directory
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)

        # Skip directories, only process files
        if os.path.isdir(file_path):
            continue

        # Check if filename ends with json
        if not filename.endswith(".json"):
            continue

        # Check if the filename contains non-alphabetic characters
        if not filename.split(".json")[
            0
        ].isalpha():  # .isalpha() checks for only letters
            print(f"Offending file: {filename}")
            bad_names = True
    return bad_names


if __name__ == "__main__":
    # Specify the directory you want to process (current directory in this case)
    directory = "macro_data"
    bad_names = check_files_in_directory(directory)
    if bad_names:
        sys.exit(1)
