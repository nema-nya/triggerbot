import pynput
import torch
import asyncio
import numpy as np
from trigger_bot_classifier_model import TriggerBotClassifierModel
from trigger_bot_autoencoder_model import TriggerBotAutoencoderModel
from capture_handler import CaptureHandler
from input_handler import InputHandler
from config import *


def frame_to_tensor(frame):
    image = np.frombuffer(frame, dtype=np.uint8)
    image = torch.tensor(image).reshape((capture_width, capture_height, 4))
    image = image[:, :, :3].float() / 255
    return image


async def main():
    autoencoder_model = TriggerBotAutoencoderModel()
    classifier_model = TriggerBotClassifierModel()

    autoencoder_model.load_state_dict(
        torch.load("autoencoder_checkpoint.pth", weights_only=True)
    )
    classifier_model.load_state_dict(
        torch.load("classifier_checkpoint.pth", weights_only=True)
    )

    autoencoder_model = autoencoder_model.cuda()
    classifier_model = classifier_model.cuda()

    capture_handler = CaptureHandler()
    input_handler = InputHandler()
    frame_buffer = []

    await capture_handler.start()
    autoencoder_model.eval()
    classifier_model.eval()
    with pynput.keyboard.Listener(on_press=input_handler.on_press) as listener:
        while True:
            frame = await capture_handler.read()
            frame_buffer.append(frame_to_tensor(frame))
            if len(frame_buffer) >= sample_window:
                frame_stack = frame_buffer[-sample_window:]
                frame_stack = (
                    torch.stack(frame_stack)
                    .permute((0, 3, 1, 2))
                    .reshape((1, -1, capture_width, capture_height))
                )
                frame_stack = frame_stack.cuda()
                if input_handler.capturing:
                    with torch.no_grad():
                        y_ = classifier_model(autoencoder_model.encode(frame_stack))

                    print(y_.cpu())
                    if y_.cpu().argmax(-1)[0] == 1:
                        input_handler.maybe_click()

            if len(frame_buffer) >= 4 * sample_window:
                frame_buffer = frame_buffer[-2 * sample_window :]
        listener.join()


if __name__ == "__main__":
    asyncio.run(main())