import subprocess
import asyncio
from config import *


class CaptureHandler:

    async def start(self):

        ffmpeg_argv = [
            "ffmpeg",
            "-fflags",
            "+genpts",
            "-init_hw_device",
            "d3d11va",
            "-threads:v",
            "1",
            "-filter_complex",
            f"ddagrab=0:output_fmt=87:draw_mouse=0:framerate={capture_rate}:video_size={capture_width}x{capture_height}:offset_x={int(monitor_width / 2 - capture_width / 2)}:offset_y={int(monitor_height / 2 - capture_height / 2)},hwdownload,format=bgra",
            "-f",
            "rawvideo",
            "-",
        ]

        self.ffmpeg_process = await asyncio.subprocess.create_subprocess_exec(
            *ffmpeg_argv,
            stdout=subprocess.PIPE,
            limit=capture_width * capture_height * 4 * 2,
        )

    async def read(self):
        buf = b""
        while len(buf) < capture_width * capture_height * 4:
            buf += await self.ffmpeg_process.stdout.read(
                capture_width * capture_height * 4 - len(buf)
            )
        return buf
