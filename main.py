"""LiveTalkAI: Real-Time Speech-to-Speech AI with GUI and Streaming"""

import asyncio
import threading
import tkinter as tk
from tkinter import scrolledtext
import whisper
import openai
import sounddevice as sd
from TTS.api import TTS
import queue
import tempfile
import scipy.io.wavfile as wavfile
import os
import logging
from dotenv import load_dotenv

# —————————————————————————————————————————————
# Setup logging to file only
logging.basicConfig(
    filename="livelog.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Load env
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

SAMPLE_RATE = 16000
CHUNK_DURATION = 2
# —————————————————————————————————————————————


class AsyncAudioStream:
    """Handles asynchronous microphone input as audio chunks."""

    def __init__(self, sample_rate=SAMPLE_RATE, chunk_duration=CHUNK_DURATION, device=None):
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.chunk_size = int(sample_rate * chunk_duration)
        self.device = device
        self.queue = queue.Queue()
        self.stream = None

    def callback(self, indata, frames, time, status):
        self.queue.put(indata.copy())
        logging.info("🔔 Audio callback received a chunk")

    def start(self):
        logging.info("🔍 Checking for available microphones…")
        if self.device is None:
            raise RuntimeError("❌ No device index set for AsyncAudioStream")
        logging.info(f"✅ Opening InputStream on device #{self.device}")
        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            callback=self.callback,
            device=self.device
        )
        self.stream.start()
        logging.info("🔊 Microphone stream started.")

    def stop(self):
        logging.info("🛑 Stopping LiveTalkAI…")
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

    def __init__(self, model_name="base"):
        logging.info("📥 Loading Whisper model…")
        self.model = whisper.load_model(model_name)
        logging.info("✅ Whisper model loaded.")

    async def transcribe(self, audio_data):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            wavfile.write(f.name, SAMPLE_RATE, audio_data)
            result = self.model.transcribe(f.name)
            os.remove(f.name)
        return result["text"]


class LanguageProcessor:
    """Uses OpenAI GPT to generate responses from input text."""

    def __init__(self, model="gpt-3.5-turbo"):
        self.model = model

    async def generate_reply(self, text):
        resp = await asyncio.to_thread(
            openai.ChatCompletion.create,
            model=self.model,
            messages=[{"role": "user", "content": text}],
            max_tokens=100
        )
        return resp.choices[0].message.content


class SpeechSynthesizer:
    """Uses Coqui TTS to convert text to speech."""

    def __init__(self, tts_model_name="tts_models/en/ljspeech/tacotron2-DDC"):
        logging.info("🔊 Loading TTS model…")
        self.tts = TTS(model_name=tts_model_name, progress_bar=False, gpu=False)
        logging.info("✅ TTS model loaded.")

    async def speak(self, text):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            self.tts.tts_to_file(text=text, file_path=f.name)
            os.system(f"aplay {f.name}")
            os.remove(f.name)


class LiveTalkAI:
    """Coordinates the full pipeline: audio input, processing, and output."""

    def __init__(self, gui, device_index, recognizer, synthesizer):
        self.audio_stream = AsyncAudioStream(device=device_index)
        self.recognizer = recognizer
        self.processor = LanguageProcessor()
        self.synthesizer = synthesizer
        self.gui = gui

    async def run(self):
        logging.info("📢 Starting LiveTalkAI.run()")
        self.audio_stream.start()
        self.gui.log("🟢 Listening started…")
        try:
            while True:
                chunk = await self.audio_stream.get_chunk()
                text = await self.recognizer.transcribe(chunk)
                self.gui.log(f"🗣️ You: {text}")
                reply = await self.processor.generate_reply(text)
                self.gui.log(f"🤖 Bot: {reply}")
                await self.synthesizer.speak(reply)
        except asyncio.CancelledError:
            self.audio_stream.stop()
            self.gui.log("🛑 Stopped listening.")


class LiveTalkAIGUI:
    """Tkinter-based GUI for controlling and displaying the application."""

    def __init__(self, root):
        self.root = root
        self.root.title("LiveTalkAI")

        # --- Chat area ---
        self.text_area = scrolledtext.ScrolledText(
            root, wrap=tk.WORD, width=60, height=20, font=("Arial", 12)
        )
        self.text_area.pack(padx=10, pady=10)
        self.text_area.config(state=tk.DISABLED)

        # Preload models
        self.log("⏳ Loading models…")
        self.recognizer = SpeechRecognizer()
        self.synthesizer = SpeechSynthesizer()
        self.log("✅ Models loaded.")

        # ——— Device listing with host API names ———
        hostapis = sd.query_hostapis()
        hostapi_names = {i: h["name"] for i, h in enumerate(hostapis)}

        self.input_devices = []
        for idx, dev in enumerate(sd.query_devices()):
            if dev["max_input_channels"] > 0:
                display = f"{dev['name']}  [{hostapi_names.get(dev['hostapi'], 'Unknown')}]"
                self.input_devices.append((display, idx))

        if not self.input_devices:
            self.log("❌ No input-capable devices found.")
            mic_names = []
        else:
            mic_names = [name for name, _ in self.input_devices]
            self.log(f"✅ Found mics: {mic_names}")

        # --- Device selection dropdown ---
        default = mic_names[0] if mic_names else ""
        self.device_var = tk.StringVar(value=default)
        tk.Label(root, text="Select Mic:").pack(pady=(0, 0))
        tk.OptionMenu(root, self.device_var, default, *mic_names).pack(padx=10, pady=(0, 10))

        # --- Start / Stop buttons ---
        btn_frame = tk.Frame(root)
        btn_frame.pack(pady=(0, 10))
        self.start_button = tk.Button(
            btn_frame,
            text="Start",
            command=self.start,
            state=tk.NORMAL if self.input_devices else tk.DISABLED
        )
        self.start_button.pack(side=tk.LEFT, padx=5)
        self.stop_button = tk.Button(btn_frame, text="Stop", command=self.stop, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=5)

        # AsyncIO
        self.loop = asyncio.new_event_loop()
        self.task = None
        self.app = None

    def log(self, message):
        """Log to file and show in the text area."""
        logging.info(message)
        self.text_area.config(state=tk.NORMAL)
        self.text_area.insert(tk.END, message + "\n")
        self.text_area.config(state=tk.DISABLED)
        self.text_area.yview(tk.END)

    def start(self):
        """Start the speech loop using the selected microphone."""
        sel_name = self.device_var.get()
        idx = next((i for n, i in self.input_devices if n == sel_name), None)
        self.log(f"🎤 Using microphone: {sel_name}")
        self.app = LiveTalkAI(
            gui=self,
            device_index=idx,
            recognizer=self.recognizer,
            synthesizer=self.synthesizer
        )
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
        threading.Thread(target=self._start_loop, daemon=True).start()
        self.root.mainloop()

    def _start_loop(self):
        asyncio.set_event_loop(self.gui.loop)
        self.gui.loop.run_forever()


if __name__ == "__main__":
    LiveTalkAIEntryPoint().run()
