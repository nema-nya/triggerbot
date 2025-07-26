import subprocess
import os
import asyncio
import pynput
import json
import PIL.Image
import numpy as np
import threading
import time
import datetime


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
                info_file = f"info_{stamp}.json"
                with open(os.path.join("outputs", info_file), "w") as file:
                    file.write(json.dumps(info, indent=2))


class ServerHandler:

    async def start(self):

        server_argv = [
            os.path.join("server", "srcds.exe"),
            "-game",
            "cstrike",
            "+map",
            "de_dust2",
            "-console",
        ]
        self.server_process = await asyncio.subprocess.create_subprocess_exec(
            *server_argv,
            cwd="server",
            stdout=subprocess.PIPE,
        )

    async def read(self):
        while True:
            line = await self.server_process.stdout.readline()
            line = line.decode("utf-8")
            if not line.startswith("[HIT]"):
                continue
            line = line[5:]
            event = json.loads(line)
            return event


class CaptureHandler:

    def __init__(self):
        self.capture_width = 512
        self.capture_height = 512

    async def start(self):
        monitor_width = 1920
        monitor_height = 1080

        ffmpeg_argv = [
            "ffmpeg",
            "-fflags",
            "+genpts",
            "-init_hw_device",
            "d3d11va",
            "-threads:v",
            "1",
            "-filter_complex",
            f"ddagrab=0:output_fmt=87:draw_mouse=0:framerate=60:video_size={self.capture_width}x{self.capture_height}:offset_x={int(monitor_width / 2 - self.capture_width / 2)}:offset_y={int(monitor_height / 2 - self.capture_height / 2)},hwdownload,format=bgra",
            "-f",
            "rawvideo",
            "-",
        ]

        self.ffmpeg_process = await asyncio.subprocess.create_subprocess_exec(
            *ffmpeg_argv, stdout=subprocess.PIPE, limit=2**19
        )

    async def read(self):
        buf = b""
        while len(buf) < self.capture_width * self.capture_height * 4:
            buf += await self.ffmpeg_process.stdout.read(
                self.capture_width * self.capture_height * 4 - len(buf)
            )
        return buf


class InputHandler:

    def __init__(self):
        self.capturing = False

    def on_press(self, key):
        if key == pynput.keyboard.Key.f7:
            if not self.capturing:
                print("starting capturing")
            self.capturing = True
        if key == pynput.keyboard.Key.f8:
            if self.capturing:
                print("stopping capturing")
            self.capturing = False


async def main():
    capture_window = 5
    capture_window_open = 0
    frame_buffer = []

    unconditional_capture_delay = 5.0
    last_unconditional_capture = 0.0

    server_handler = ServerHandler()
    capture_handler = CaptureHandler()
    input_handler = InputHandler()
    storage = Storage(capture_handler.capture_width, capture_handler.capture_height)

    await server_handler.start()
    await capture_handler.start()

    with pynput.keyboard.Listener(on_press=input_handler.on_press) as listener:
        server_task = asyncio.create_task(server_handler.read(), name="server")
        capture_task = asyncio.create_task(capture_handler.read(), name="capture")

        tasks = {server_task, capture_task}

        while True:
            done, tasks = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for t in done:
                if t.get_name() == "server":
                    stamp = time.time()
                    tasks.add(asyncio.create_task(server_handler.read(), name="server"))
                    if input_handler.capturing:
                        info = t.result()
                        storage.infos.append((info, stamp))
                        if capture_window_open == 0:
                            capture_window_open = capture_window
                            previous_frames = frame_buffer[-capture_window:]
                            frame_buffer = frame_buffer[:-capture_window]
                            for timed_frame in previous_frames:
                                storage.frames.append(timed_frame)
                elif t.get_name() == "capture":
                    stamp = time.time()
                    tasks.add(
                        asyncio.create_task(capture_handler.read(), name="capture")
                    )
                    if input_handler.capturing:
                        frame = t.result()
                        if last_unconditional_capture + unconditional_capture_delay < stamp and capture_window_open == 0:
                            capture_window_open = capture_window
                            previous_frames = frame_buffer[-capture_window:]
                            frame_buffer = frame_buffer[:-capture_window]
                            last_unconditional_capture = stamp
                            for timed_frame in previous_frames:
                                storage.frames.append(timed_frame)
                        if capture_window_open > 0:
                            storage.frames.append((frame, stamp))
                            capture_window_open -= 1
                        else:
                            frame_buffer = frame_buffer[-capture_window:] + [(frame, stamp)]
        listener.join()


if __name__ == "__main__":
    asyncio.run(main())
