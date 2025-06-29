import sys
import time
import asyncio
import threading
import tkinter as tk
from tkinter import scrolledtext, filedialog
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
from datetime import datetime
import textwrap
import numpy as np

# Create a new log file each run
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
logging.basicConfig(
    filename=f"livelog_{timestamp}.log",
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filemode="w",
)

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

SAMPLE_RATE = 16000
CHUNK_DURATION = 2  # seconds
MAX_TEXT_CHUNK = 200  # characters per chunk


def split_text_chunks(text, max_length=MAX_TEXT_CHUNK):
    """Split a long text into chunks of approximately max_length characters."""
    return textwrap.wrap(text, max_length)


class AsyncAudioStream:
    """Handles asynchronous microphone input as audio chunks."""

    def __init__(self, sample_rate=SAMPLE_RATE, chunk_duration=CHUNK_DURATION, device=None):
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.device = device
        self.queue = queue.Queue()
        self.stream = None

    def callback(self, indata, frames, time_info, status):
        self.queue.put(indata.copy())
        logging.info("🔔 Audio callback received")

    def start(self, blocksize=None):
        if blocksize is None:
            blocksize = int(self.sample_rate * self.chunk_duration)
        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            callback=self.callback,
            device=self.device,
            blocksize=blocksize,
        )
        self.stream.start()
        logging.info(f"🔊 Mic stream started with blocksize={blocksize} frames.")

    def stop(self):
        if self.stream:
            self.stream.stop()
            self.stream.close()
            logging.info("🛑 Mic stream stopped.")

    async def get_chunk(self):
        frames_needed = int(self.sample_rate * self.chunk_duration)
        buffer = []
        total = 0
        while total < frames_needed:
            if not self.queue.empty():
                data = self.queue.get().flatten()
                buffer.append(data)
                total += len(data)
            else:
                await asyncio.sleep(0.01)
        chunk = np.concatenate(buffer)
        logging.info(f"🔔 Retrieved full chunk: {len(chunk)} frames.")
        return chunk


class SpeechRecognizer:
    """Wraps Whisper models for transcription."""

    def __init__(self, model_name="base", use_openai_whisper=False):
        self.use_openai = use_openai_whisper
        if not self.use_openai:
            logging.info(f"📥 Loading Whisper model: {model_name}")
            self.model = whisper.load_model(model_name)
            logging.info("✅ Whisper model loaded.")
        else:
            logging.info("✅ Using OpenAI Whisper API for transcription.")

    async def transcribe(self, audio_data):
        # Write audio to temp file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            wavfile.write(f.name, SAMPLE_RATE, audio_data)
            file_path = f.name

        if self.use_openai:
            with open(file_path, "rb") as audio_file:
                transcript = openai.Audio.transcribe("whisper-1", audio_file)
                text = transcript.get("text", "").strip()
        else:
            result = self.model.transcribe(file_path)
            text = result.get("text", "").strip()

        os.remove(file_path)
        logging.info(f"📝 Transcribed text: '{text}'")
        return split_text_chunks(text) if text else []


class LiveTalkAI:
    """Coordinates processing—bot responses and TTS disabled."""

    def __init__(self, gui, audio_stream, recognizer):
        self.audio_stream = audio_stream
        self.recognizer = recognizer
        self.gui = gui

    async def run(self):
        self.gui.log("▶️ Processing loop started.")
        try:
            while True:
                chunk = await self.audio_stream.get_chunk()
                text_chunks = await self.recognizer.transcribe(chunk)
                for idx, text in enumerate(text_chunks, 1):
                    if not text.strip():
                        continue
                    self.gui.log(f"You (chunk {idx}/{len(text_chunks)}): {text}", tag="user")
        except asyncio.CancelledError:
            self.audio_stream.stop()
            self.gui.log("🛑 Listening stopped.")


class LiveTalkAIGUI:
    """Tkinter-based GUI for controlling and displaying the application."""

    def __init__(self, root):
        self.root = root
        self.root.title("LiveTalkAI")

        # Text display area
        self.text_area = scrolledtext.ScrolledText(
            root, wrap=tk.WORD, width=60, height=20, font=("Arial", 12)
        )
        self.text_area.pack(padx=10, pady=10)
        self.text_area.config(state=tk.DISABLED)
        self.text_area.tag_config("user", foreground="blue")

        # Controls frame
        control_frame = tk.Frame(root)
        control_frame.pack(pady=5)

        # Whisper model selection
        tk.Label(control_frame, text="Whisper Model:").grid(row=0, column=0)
        whisper_models = ["tiny", "base", "small", "medium", "large"]
        self.whisper_var = tk.StringVar(value="base")
        tk.OptionMenu(control_frame, self.whisper_var, *whisper_models).grid(row=0, column=1)

        # OpenAI Whisper toggle
        tk.Label(control_frame, text="Use OpenAI Whisper:").grid(row=0, column=2)
        self.openai_var = tk.BooleanVar(value=False)
        tk.Checkbutton(control_frame, variable=self.openai_var).grid(row=0, column=3)

        # TTS model selection
        tk.Label(control_frame, text="TTS Model:").grid(row=1, column=0)
        tts_models = ["tts_models/en/ljspeech/tacotron2-DDC", "tts_models/en/ljspeech/glow-tts"]
        self.tts_var = tk.StringVar(value=tts_models[0])
        tk.OptionMenu(control_frame, self.tts_var, *tts_models).grid(row=1, column=1)

        # Buttons frame
        btn_frame = tk.Frame(root)
        btn_frame.pack(pady=5)
        self.start_button = tk.Button(btn_frame, text="Start", command=self.start)
        self.start_button.pack(side=tk.LEFT, padx=5)
        self.stop_button = tk.Button(btn_frame, text="Stop", command=self.stop, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=5)
        self.test_button = tk.Button(btn_frame, text="Test Mic", command=self.test_mic)
        self.test_button.pack(side=tk.LEFT, padx=5)
        self.clear_button = tk.Button(btn_frame, text="Clear", command=self.clear)
        self.clear_button.pack(side=tk.LEFT, padx=5)
        self.save_button = tk.Button(btn_frame, text="Save", command=self.save)
        self.save_button.pack(side=tk.LEFT, padx=5)

        # Status label
        self.status_label = tk.Label(root, text="Ready")
        self.status_label.pack(pady=(5, 10))

        # Audio stream and device list
        self.audio_stream = AsyncAudioStream()
        self.input_devices = [
            (dev['name'], idx) for idx, dev in enumerate(sd.query_devices()) if dev['max_input_channels'] > 0
        ]
        mic_names = [name for name, _ in self.input_devices]
        status = f"✅ {len(mic_names)} mic(s) found" if mic_names else "❌ No mics found"
        self.log(status)

        # Mic selection dropdown
        tk.Label(root, text="Select Mic:").pack()
        self.device_var = tk.StringVar(value=mic_names[0] if mic_names else "")
        tk.OptionMenu(root, self.device_var, *(mic_names or [""])).pack()

        # Event loop
        self.loop = asyncio.new_event_loop()
        self.task = None

    def log(self, msg, tag=None):
        logging.info(msg)
        self.text_area.config(state=tk.NORMAL)
        if tag:
            self.text_area.insert(tk.END, msg + "\n", tag)
        else:
            self.text_area.insert(tk.END, msg + "\n")
        self.text_area.config(state=tk.DISABLED)
        self.text_area.yview(tk.END)

    def start(self):
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        threading.Thread(target=self._load_and_start, daemon=True).start()

    def _load_and_start(self):
        self.log("⏳ Reloading models…")
        self.recognizer = SpeechRecognizer(
            model_name=self.whisper_var.get(), use_openai_whisper=self.openai_var.get()
        )
        self.log("✅ Models loaded.")

        sel = self.device_var.get()
        idx = next((i for n, i in self.input_devices if n == sel), None)
        self.log(f"🎤 Mic selected: {sel}")
        self.audio_stream.device = idx
        self.audio_stream.start()
        self.log("🟢 Listening started…")
        self.task = self.loop.create_task(
            LiveTalkAI(self, self.audio_stream, self.recognizer).run()
        )
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def stop(self):
        if self.task:
            self.task.cancel()
        self.audio_stream.stop()
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.log("🛑 Stop requested.")

    def test_mic(self):
        sel = self.device_var.get()
        idx = next((i for n, i in self.input_devices if n == sel), None)
        threading.Thread(target=self._test_thread, args=(idx,), daemon=True).start()

    def _test_thread(self, device_index):
        from scipy.io.wavfile import write
        duration = 3
        self.log(f"🎤 Testing mic for {duration}s on device #{device_index}...")
        try:
            audio = sd.rec(
                int(duration * SAMPLE_RATE),
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="int16",
                device=device_index,
            )
            sd.wait()
            write("mic_test.wav", SAMPLE_RATE, audio)
            self.log("💾 mic_test.wav saved.")
            os.system("aplay mic_test.wav")
            self.log("▶️ mic_test.wav played back.")
        except Exception as e:
            self.log(f"❌ Test failed: {e}")

    def clear(self):
        self.text_area.config(state=tk.NORMAL)
        self.text_area.delete("1.0", tk.END)
        self.text_area.config(state=tk.DISABLED)
        self.log("🧹 Chat cleared.")

    def save(self):
        try:
            filepath = filedialog.asksaveasfilename(
                defaultextension=".txt",
                filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")],
            )
        except tk.TclError:
            self.log("❌ Save dialog failed: UI already closed.")
            return
        if not filepath:
            return
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(self.text_area.get("1.0", tk.END))
            self.log(f"💾 Chat saved to {filepath}")
        except Exception as e:
            self.log(f"❌ Save failed: {e}")


class LiveTalkAIEntryPoint:
    """Initializes and starts the GUI."""

    def run(self):
        root = tk.Tk()
        gui = LiveTalkAIGUI(root)
        root.mainloop()


if __name__ == "__main__":
    LiveTalkAIEntryPoint().run()
