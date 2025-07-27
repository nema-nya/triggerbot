import asyncio
import pynput
import time
from capture_handler import CaptureHandler
from storage import Storage
from server_handler import ServerHandler
from input_handler import InputHandler
from config import *


async def main():
    capture_window_open = 0
    frame_buffer = []

    last_unconditional_capture = 0.0

    server_handler = ServerHandler()
    capture_handler = CaptureHandler()
    input_handler = InputHandler()
    storage = Storage(capture_width, capture_height)

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
                        if (
                            last_unconditional_capture + unconditional_capture_delay
                            < stamp
                            and capture_window_open == 0
                        ):
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
                            frame_buffer = frame_buffer[-capture_window:] + [
                                (frame, stamp)
                            ]
        listener.join()


if __name__ == "__main__":
    asyncio.run(main())
