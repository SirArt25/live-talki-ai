"""LiveTalkAI: Real-Time Speech-to-Speech AI with GUI and Streaming

This module implements a real-time speech-to-speech system using Whisper for transcription,
OpenAI's GPT for text generation, and Coqui TTS for speech synthesis. It includes an asyncio-based
architecture and a Tkinter GUI.
"""

import asyncio
import threading
import tkinter as tk
from tkinter import scrolledtext
import whisper
import openai
import numpy as np
import sounddevice as sd
from TTS.api import TTS
import queue
import tempfile
import scipy.io.wavfile as wavfile
import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
SAMPLE_RATE = 16000
CHUNK_DURATION = 2


class AsyncAudioStream:
    """Handles asynchronous microphone input as audio chunks."""

    def __init__(self, sample_rate=SAMPLE_RATE, chunk_duration=CHUNK_DURATION):
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.chunk_size = int(sample_rate * chunk_duration)
        self.queue = queue.Queue()
        self.stream = None

    def callback(self, indata, frames, time, status):
        self.queue.put(indata.copy())

    def start(self):
        self.stream = sd.InputStream(samplerate=self.sample_rate, channels=1, callback=self.callback)
        self.stream.start()

    def stop(self):
        if self.stream:
            self.stream.stop()
            self.stream.close()

    async def get_chunk(self):
        while True:
            if not self.queue.empty():
                return self.queue.get().flatten()
            await asyncio.sleep(0.01)


class SpeechRecognizer:
    """Wraps the Whisper model for transcription."""

    def __init__(self):
        self.model = whisper.load_model("base")

    async def transcribe(self, audio_data):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            wavfile.write(f.name, SAMPLE_RATE, audio_data)
            result = self.model.transcribe(f.name)
            os.remove(f.name)
        return result['text']


class LanguageProcessor:
    """Uses OpenAI GPT to generate responses from input text."""

    def __init__(self, model="gpt-3.5-turbo"):
        self.model = model

    async def generate_reply(self, text):
        response = await asyncio.to_thread(
            openai.ChatCompletion.create,
            model=self.model,
            messages=[{"role": "user", "content": text}],
            max_tokens=100
        )
        return response['choices'][0]['message']['content']


class SpeechSynthesizer:
    """Uses Coqui TTS to convert text to speech."""

    def __init__(self):
        self.tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False, gpu=False)

    async def speak(self, text):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            self.tts.tts_to_file(text=text, file_path=f.name)
            os.system(f"aplay {f.name}")
            os.remove(f.name)


class LiveTalkAI:
    """Coordinates the full pipeline: audio input, processing, and output."""

    def __init__(self, gui):
        self.audio_stream = AsyncAudioStream()
        self.recognizer = SpeechRecognizer()
        self.processor = LanguageProcessor()
        self.synthesizer = SpeechSynthesizer()
        self.gui = gui

    async def run(self):
        self.audio_stream.start()
        self.gui.log("🟢 Listening started...")
        try:
            while True:
                chunk = await self.audio_stream.get_chunk()
                text = await self.recognizer.transcribe(chunk)
                self.gui.log(f"🗣️ You: {text}")
                response = await self.processor.generate_reply(text)
                self.gui.log(f"🤖 Bot: {response}")
                await self.synthesizer.speak(response)
        except asyncio.CancelledError:
            self.audio_stream.stop()
            self.gui.log("🛑 Stopped listening.")


class LiveTalkAIGUI:
    """Tkinter-based GUI for controlling and displaying the application."""

    def __init__(self, root):
        self.root = root
        self.root.title("LiveTalkAI")
        self.text_area = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=60, height=20, font=("Arial", 12))
        self.text_area.pack(padx=10, pady=10)
        self.text_area.config(state=tk.DISABLED)

        self.start_button = tk.Button(root, text="Start", command=self.start)
        self.start_button.pack(side=tk.LEFT, padx=10, pady=10)

        self.stop_button = tk.Button(root, text="Stop", command=self.stop, state=tk.DISABLED)
        self.stop_button.pack(side=tk.RIGHT, padx=10, pady=10)

        self.loop = asyncio.get_event_loop()
        self.task = None
        self.app = None

    def log(self, message):
        self.text_area.config(state=tk.NORMAL)
        self.text_area.insert(tk.END, message + "\n")
        self.text_area.config(state=tk.DISABLED)
        self.text_area.yview(tk.END)

    def start(self):
        self.app = LiveTalkAI(self)
        self.task = self.loop.create_task(self.app.run())
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)

    def stop(self):
        if self.task:
            self.task.cancel()
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)


class LiveTalkAIEntryPoint:
    """Initializes and starts the Tkinter GUI and asyncio loop."""

    def __init__(self):
        self.root = tk.Tk()
        self.gui = LiveTalkAIGUI(self.root)

    def run(self):
        def start_loop():
            asyncio.set_event_loop(self.gui.loop)
            self.gui.loop.run_forever()

        threading.Thread(target=start_loop, daemon=True).start()
        self.root.mainloop()


if __name__ == "__main__":
    LiveTalkAIEntryPoint().run()
