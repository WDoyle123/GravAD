import os
import pickle

def clear_folder(folder_path):
    # Check if the folder exists
    if not os.path.exists(folder_path):
        print(f"The folder '{folder_path}' does not exist.")
        return

    # Iterate through files and delete them
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)

    # Iterate through subfolders and delete them (optional)
    for subfolder in os.listdir(folder_path):
        subfolder_path = os.path.join(folder_path, subfolder)
        if os.path.isdir(subfolder_path):
            clear_folder(subfolder_path)


def save_txt(file_path, filename, data):
    os.makedirs(file_path, exist_ok=True)
    path = os.path.join(file_path, filename)
    
    with open(path, "w") as f:
        f.write(str(data))


def save_pickle(file_path, filename, data):
    os.makedirs(file_path, exist_ok=True)
    path = os.path.join(file_path, filename)

    with open(path, "wb") as f:
        pickle.dump(data, f)

