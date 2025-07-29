import datetime
import os
import json
from config import *
from utils import *

def parse_stamp(text_stamp):
    return datetime.datetime.strptime(text_stamp, "%Y-%m-%d-%H-%M-%S-%f").timestamp()





def main():
    frame_names = []
    info_names = []
    delay = 4 / 3 / capture_rate
    frame_time_shift = -1 / capture_rate / 2

    for file_name in os.listdir("outputs"):
        if file_name.startswith("frame"):
            frame_names.append(file_name)
        elif file_name.startswith("info"):
            info_names.append(file_name)

    events = []

    for frame_name in frame_names:
        stamp = parse_stamp(frame_name[6:-4])
        events.append((stamp + frame_time_shift, "frame", frame_name))

    for info_name in info_names:
        stamp = parse_stamp(info_name[5:-5])
        events.append((stamp, "info", info_name))

    events = sorted(events)
    frame_buffer = []
    training_samples = []
    frames_from_last_sample = 0

    for stamp, kind, file_name in events:
        if kind == "frame":
            if (
                len(frame_buffer) >= sample_window
                and frames_from_last_sample >= sample_window
            ):
                frames_from_last_sample = 0
                sample_frames = frame_buffer[-sample_window:]
                training_samples.append(
                    {
                        "frames": [tf[0] for tf in sample_frames],
                        "info": None,
                    }
                )
            if frame_buffer and frame_buffer[-1][1] + delay < stamp:
                frame_buffer = []
            frame_buffer.append((file_name, stamp))
            frames_from_last_sample += 1
            if len(frame_buffer) > 4 * sample_window:
                frame_buffer = frame_buffer[-2 * sample_window :]
        elif kind == "info":
            if frame_buffer and frame_buffer[-1][1] + delay < stamp:
                frame_buffer = []
            if len(frame_buffer) >= sample_window:
                sample_frames = frame_buffer[-sample_window:]
                training_samples.append(
                    {
                        "frames": [tf[0] for tf in sample_frames],
                        "info": file_name,
                    }
                )
            frames_from_last_sample = 0
    with open("training_samples.json", "w") as file:
        file.write(json.dumps(training_samples, indent=2))
    print(len(training_samples))


if __name__ == "__main__":
    main()
