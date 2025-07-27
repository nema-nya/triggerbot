import threading
import time
import datetime
import numpy as np
import PIL.Image
import os
import json


class Storage:

    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.frames = []
        self.infos = []
        self.thread = threading.Thread(target=self.loop)
        self.thread.start()

    def loop(self):
        while True:
            if not self.frames and not self.infos:
                time.sleep(0.1)
                continue
            if self.frames:
                frame, stamp = self.frames.pop()
                stamp = datetime.datetime.fromtimestamp(stamp).strftime(
                    "%Y-%m-%d-%H-%M-%S-%f"
                )
                frame_file = f"frame_{stamp}.png"
                image_array = np.frombuffer(frame, dtype=np.uint8).reshape(
                    (
                        self.width,
                        self.height,
                        4,
                    )
                )
                image_array = np.stack(
                    [
                        image_array[:, :, 2],
                        image_array[:, :, 1],
                        image_array[:, :, 0],
                        image_array[:, :, 3],
                    ],
                    axis=-1,
                )
                image = PIL.Image.fromarray(image_array, mode="RGBA")
                image.save(os.path.join("outputs", frame_file))

            if self.infos:
                info, stamp = self.infos.pop()
                stamp = datetime.datetime.fromtimestamp(stamp).strftime(
                    "%Y-%m-%d-%H-%M-%S-%f"
                )
                info_file = f"info_{stamp}.json"
                with open(os.path.join("outputs", info_file), "w") as file:
                    file.write(json.dumps(info, indent=2))
