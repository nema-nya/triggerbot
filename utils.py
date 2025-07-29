import os
import json

def get_label_file(frame_file_name):
    frame_stamp = frame_file_name[6:-4]
    return f"label_{frame_stamp}.json" 

def get_label_file_if_exists(frame_file_name):
    label_file = get_label_file(frame_file_name)
    return (
        label_file
        if os.path.isfile(os.path.join("outputs", label_file))
        else None
    )

def get_frame_label(frame_filename):
    label_filename = get_label_file_if_exists(frame_filename)
    if label_filename is None:
        return None
    with open(os.path.join("outputs", label_filename), "r") as file:
        return json.loads(file.read())
    
def save_frame_label(frame_filename, label):
    label_filename = get_label_file(frame_filename)
    with open(os.path.join("outputs", label_filename), "w") as file:
        file.write(json.dumps(label))