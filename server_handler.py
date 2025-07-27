import os
import asyncio
import subprocess
import json


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
