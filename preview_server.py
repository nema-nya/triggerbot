import flask
import json
import os
from config import *
from utils import *

app = flask.Flask(__name__, static_folder="outputs")


def get_preview_html(body):
    return f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>preview</title>
</head>
<body>
    {body}
</body>
</html>
"""


@app.route("/<int:id>")
def preview(id):
    with open("training_samples.json", "r") as file:
        training_samples = json.loads(file.read())

    training_samples = [
        sample for sample in training_samples if sample["info"] is not None
    ]

    if "label_hit" in flask.request.args:
        label_id = int(flask.request.args["label_hit"])
        sample = training_samples[label_id]
        last_frame_filename = sample["frames"][-1]
        label_contents = get_frame_label(last_frame_filename)
        if label_contents is None:
            label_contents = {}
        label_contents["is_hit"] = True
        save_frame_label(last_frame_filename, label_contents)

    if "label_miss" in flask.request.args:
        label_id = int(flask.request.args["label_miss"])
        sample = training_samples[label_id]
        last_frame_filename = sample["frames"][-1]
        label_contents = get_frame_label(last_frame_filename)
        if label_contents is None:
            label_contents = {}
        label_contents["is_hit"] = False
        save_frame_label(last_frame_filename, label_contents)


    sample = training_samples[id]
    body = ""
    for i in range(sample_window):
        frame_location = flask.url_for(endpoint="static", filename=sample["frames"][i])
        body += f'<img src="{frame_location}" width="600" height="600">'
    body += "<br>"
    if sample["info"] is not None:
        info_text_location = os.path.join("outputs", sample["info"])
        with open(info_text_location, "r") as f:
            info_json = json.loads(f.read())
        body += f"<p> {info_json}</p>"
    else:
        body += "<p> Miss </p>"
    body += "<br>"
    label_contents = get_frame_label(sample["frames"][-1])
    if label_contents is not None:
        body += f"<p> {label_contents}</p>"
    else:
        body += "<p> unlabelled </p>"
    body += "<br>"
    prev_id = max(id - 1, 0)
    next_id = min(id + 1, len(training_samples) - 1)

    button_l = f'<a href="/{prev_id}"><button> <--- </button></a>'
    button_r_with_hit = (
        f'<a href="/{next_id}?label_hit={id}"><button> HITT </button></a>'
    )
    button_r_with_miss = (
        f'<a href="/{next_id}?label_miss={id}"><button> MISS </button></a>'
    )
    button_r = f'<a href="/{next_id}"><button> ---> </button></a>'

    body += button_l + button_r_with_hit + button_r_with_miss + button_r
    return get_preview_html(body)
#1104