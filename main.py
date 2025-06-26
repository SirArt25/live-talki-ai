"""LiveTalkAI: Real-Time Speech-to-Speech AI with GUI and Streaming

This module implements a real-time speech-to-speech system using Whisper for transcription,
OpenAI's GPT for text generation, and Coqui TTS for speech synthesis. It includes an asyncio-based
architecture with a Tkinter GUI, preloads models with a splash screen, and logs to a file.
"""

import asyncio
import threading
import tkinter as tk
from tkinter import scrolledtext, ttk
import logging
import whisper
import openai
import sounddevice as sd
from TTS.api import TTS
import queue
import tempfile
import scipy.io.wavfile as wavfile
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Configure logging to file only
logger = logging.getLogger()
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler('livelog.log')
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

SAMPLE_RATE = 16000
CHUNK_DURATION = 2


class AsyncAudioStream:
    """Handles asynchronous microphone input as audio chunks."""

    def __init__(self, sample_rate=SAMPLE_RATE, chunk_duration=CHUNK_DURATION):
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.queue = queue.Queue()
        self.stream = None

    def list_microphones(self):
        devices = sd.query_devices()
        return [dev for dev in devices if dev['max_input_channels'] > 0]

    def start(self):
        # Log and verify available microphones
        logging.info("🔍 Checking available microphones before starting audio stream:")
        mics = self.list_microphones()
        for mic in mics:
            logging.info(f" | {mic['name']}")
        if not mics:
            logging.error("❌ No microphone devices found on system.")
            raise RuntimeError("No microphone devices found.")
        # Start audio stream on default device
        self.stream = sd.InputStream(samplerate=self.sample_rate,
                                     channels=1,
                                     callback=self._callback)
        self.stream.start()
        logging.info("🔊 Microphone stream started.") 
        sd.InputStream(samplerate=self.sample_rate,
                                     channels=1,
                                     callback=self._callback)
        self.stream.start()

    def stop(self):
        if self.stream:
            self.stream.stop()
            self.stream.close()

    def _callback(self, indata, frames, time, status):
        logging.info("🔔 Audio callback: chunk received")
        self.queue.put(indata.copy().flatten())

    async def get_chunk(self):
        while True:
            if not self.queue.empty():
                return self.queue.get()
            await asyncio.sleep(0.01)


class SpeechRecognizer:
    """Wraps Whisper for speech-to-text."""

    def __init__(self):
        logging.info("🧠 Loading Whisper model...")
        self.model = whisper.load_model("base")
        logging.info("✅ Whisper model loaded.")

    async def transcribe(self, audio_data):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            wavfile.write(f.name, SAMPLE_RATE, audio_data)
            result = self.model.transcribe(f.name)
            os.remove(f.name)
        return result.get('text', '')


class LanguageProcessor:
    """Uses OpenAI GPT to generate a reply."""

    def __init__(self, model="gpt-3.5-turbo"):
        self.model = model

    async def generate_reply(self, text):
        logging.info(f"🤖 Generating reply for: {text}")
        response = await asyncio.to_thread(
            openai.ChatCompletion.create,
            model=self.model,
            messages=[{"role": "user", "content": text}],
            max_tokens=100
        )
        reply = response['choices'][0]['message']['content']
        logging.info(f"✅ Reply: {reply}")
        return reply


class SpeechSynthesizer:
    """Wraps Coqui TTS for text-to-speech."""

    def __init__(self, model_name="tts_models/en/ljspeech/tacotron2-DDC"):
        self.model_name = model_name
        self.tts = None

    async def init_model(self):
        logging.info("🐢 Loading TTS model...")
        self.tts = await asyncio.to_thread(TTS, model_name=self.model_name, progress_bar=False, gpu=False)
        logging.info("✅ TTS model loaded.")

    async def speak(self, text):
        if self.tts is None:
            await self.init_model()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            self.tts.tts_to_file(text=text, file_path=f.name)
            os.system(f"aplay {f.name}")
            os.remove(f.name)


class LiveTalkAI:
    """Pipeline: capture -> transcribe -> generate -> speak."""

    def __init__(self, gui):
        self.audio = AsyncAudioStream()
        self.recognizer = SpeechRecognizer()
        self.processor = LanguageProcessor()
        self.synthesizer = SpeechSynthesizer()
        self.gui = gui

    async def run(self):
        logging.info("📢 LiveTalkAI run started")
        asyncio.create_task(self.synthesizer.init_model())
        self.audio.start()
        self.gui.log("🟢 Listening...")
        try:
            while True:
                chunk = await self.audio.get_chunk()
                text = await self.recognizer.transcribe(chunk)
                self.gui.log(f"🗣️ You: {text}")
                resp = await self.processor.generate_reply(text)
                self.gui.log(f"🤖 Bot: {resp}")
                await self.synthesizer.speak(resp)
        except asyncio.CancelledError:
            self.audio.stop()
            self.gui.log("🛑 Stopped.")


class LiveTalkAIGUI:
    """Tkinter GUI controller."""

    def __init__(self, root):
        self.root = root
        self.root.title("LiveTalkAI")
        self.text = scrolledtext.ScrolledText(root, width=60, height=20)
        self.text.pack(padx=10, pady=10)
        self.text.config(state=tk.DISABLED)

        self.start_btn = tk.Button(root, text="Start", command=self.start)
        self.start_btn.pack(side=tk.LEFT, padx=10, pady=10)
        self.stop_btn = tk.Button(root, text="Stop", command=self.stop, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.RIGHT, padx=10, pady=10)

        self.loop = asyncio.new_event_loop()
        self.task = None

    def log(self, msg):
        logging.info(msg)
        self.text.config(state=tk.NORMAL)
        self.text.insert(tk.END, msg + "\n")
        self.text.config(state=tk.DISABLED)
        self.text.yview(tk.END)

    def start(self):
        # Check mics
        mics = AsyncAudioStream().list_microphones()
        if not mics:
            self.log("❌ No microphones detected.")
            return
        names = [d['name'] for d in mics]
        self.log(f"✅ Devices: {names}")
        self.task = self.loop.create_task(LiveTalkAI(self).run())
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)

        def run_loop():
            asyncio.set_event_loop(self.loop)
            self.loop.run_forever()

        threading.Thread(target=run_loop, daemon=True).start()

    def stop(self):
        if self.task:
            self.task.cancel()
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)


class LiveTalkAIEntryPoint:
    """Splash + preload + launch GUI."""

    def run(self):
        splash = tk.Tk()
        splash.title("Loading LiveTalkAI")
        splash.geometry("300x100")
        pb = ttk.Progressbar(splash, mode="indeterminate")
        pb.pack(expand=True, fill=tk.BOTH, padx=20, pady=20)
        pb.start(10)
        splash.update()

        logging.info("🚀 Preloading Whisper and TTS...")
        whisper.load_model("base")
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(SpeechSynthesizer().init_model())
        loop.close()

        pb.stop()
        splash.destroy()

        root = tk.Tk()
        gui = LiveTalkAIGUI(root)
        root.mainloop()


if __name__ == "__main__":
    LiveTalkAIEntryPoint().run()
