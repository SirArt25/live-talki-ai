import sys
import time
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
from datetime import datetime

# Create a new log file each run
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
logging.basicConfig(
    filename=f"livelog_{timestamp}.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filemode="w",
)

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

SAMPLE_RATE = 16000
CHUNK_DURATION = 2  # seconds


class AsyncAudioStream:
    """Handles asynchronous microphone input as audio chunks."""

    def __init__(self, sample_rate=SAMPLE_RATE, chunk_duration=CHUNK_DURATION, device=None):
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.device = device
        self.queue = queue.Queue()
        self.stream = None

    def callback(self, indata, frames, time_info, status):
        """Called by SoundDevice when a new chunk is ready."""
        self.queue.put(indata.copy())
        logging.info("🔔 Audio callback received")

    def start(self, blocksize=None):
        """Open the input stream; must be called from the GUI thread."""
        logging.info("🔍 Attempting to start mic input...")
        if self.device is None:
            raise RuntimeError("❌ No mic device index provided.")
        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            callback=self.callback,
            device=self.device,
            blocksize=blocksize,
        )
        self.stream.start()
        logging.info("🔊 Mic stream started.")

    def stop(self):
        """Stop and close the input stream."""
        if self.stream:
            self.stream.stop()
            self.stream.close()
            logging.info("🛑 Mic stream stopped.")

    async def get_chunk(self):
        """Wait for next chunk from the queue."""
        while True:
            if not self.queue.empty():
                return self.queue.get().flatten()
            await asyncio.sleep(0.01)


class SpeechRecognizer:
    """Wraps the Whisper model for transcription."""

    def __init__(self, model_name="base"):
        logging.info("📥 Loading Whisper model…")
        self.model = whisper.load_model(model_name)
        logging.info("✅ Whisper loaded.")

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
        response = await asyncio.to_thread(
            openai.ChatCompletion.create,
            model=self.model,
            messages=[{"role": "user", "content": text}],
            max_tokens=100,
        )
        return response.choices[0].message.content


class SpeechSynthesizer:
    """Uses Coqui TTS to convert text to speech."""

    def __init__(self, tts_model_name="tts_models/en/ljspeech/tacotron2-DDC"):
        logging.info("🔊 Loading Coqui TTS…")
        self.tts = TTS(model_name=tts_model_name, progress_bar=False, gpu=False)
        logging.info("✅ Coqui TTS loaded.")

    async def speak(self, text):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            self.tts.tts_to_file(text=text, file_path=f.name)
            os.system(f"aplay {f.name}")
            os.remove(f.name)


class LiveTalkAI:
    """Coordinates processing and output—assumes mic stream already started."""

    def __init__(self, gui, audio_stream, recognizer, synthesizer):
        self.audio_stream = audio_stream
        self.recognizer = recognizer
        self.processor = LanguageProcessor()
        self.synthesizer = synthesizer
        self.gui = gui

    async def run(self):
        self.gui.log("▶️ Processing loop started.")
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
            self.gui.log("🛑 Listening stopped.")


class LiveTalkAIGUI:
    """Tkinter-based GUI for controlling and displaying the application."""

    def __init__(self, root):
        self.root = root
        self.root.title("LiveTalkAI")

        # Text log
        self.text_area = scrolledtext.ScrolledText(
            root, wrap=tk.WORD, width=60, height=20, font=("Arial", 12)
        )
        self.text_area.pack(padx=10, pady=10)
        self.text_area.config(state=tk.DISABLED)

        # Load models
        self.log("⏳ Loading models…")
        self.recognizer = SpeechRecognizer()
        self.synthesizer = SpeechSynthesizer()
        self.log("✅ Models loaded.")

        # Prepare audio stream (device set later)
        self.audio_stream = AsyncAudioStream()

        # Detect mics
        self.input_devices = [
            (dev["name"], idx)
            for idx, dev in enumerate(sd.query_devices())
            if dev["max_input_channels"] > 0
        ]
        mic_names = [name for name, _ in self.input_devices]
        status = f"✅ {len(mic_names)} mic(s) found" if mic_names else "❌ No mics found"
        self.log(status)

        # Mic selector
        self.device_var = tk.StringVar(value=mic_names[0] if mic_names else "")
        tk.Label(root, text="Select Mic:").pack()
        tk.OptionMenu(root, self.device_var, *(mic_names or [""])).pack()

        # Status label
        self.status_label = tk.Label(root, text=status)
        self.status_label.pack(pady=(5, 10))

        # Buttons: Start / Stop / Test Mic
        btn_frame = tk.Frame(root)
        btn_frame.pack(pady=5)
        self.start_button = tk.Button(
            btn_frame,
            text="Start",
            command=self.start,
            state=tk.NORMAL if mic_names else tk.DISABLED,
        )
        self.start_button.pack(side=tk.LEFT, padx=5)
        self.stop_button = tk.Button(
            btn_frame, text="Stop", command=self.stop, state=tk.DISABLED
        )
        self.stop_button.pack(side=tk.LEFT, padx=5)
        self.test_button = tk.Button(
            btn_frame,
            text="Test Mic",
            command=self.test_mic,
            state=tk.NORMAL if mic_names else tk.DISABLED,
        )
        self.test_button.pack(side=tk.LEFT, padx=5)

        # Async setup
        self.loop = asyncio.new_event_loop()
        self.task = None

    def log(self, msg):
        """Log to file and show in GUI."""
        logging.info(msg)
        self.text_area.config(state=tk.NORMAL)
        self.text_area.insert(tk.END, msg + "\n")
        self.text_area.config(state=tk.DISABLED)
        self.text_area.yview(tk.END)

    def start(self):
        """Start mic stream then launch async transcription loop."""
        sel = self.device_var.get()
        idx = next((i for n, i in self.input_devices if n == sel), None)
        self.log(f"🎤 Mic selected: {sel}")

        # Start mic input immediately
        try:
            blocksize = int(self.audio_stream.sample_rate // 10)
            self.audio_stream.device = idx
            self.audio_stream.start(blocksize=blocksize)
            self.log("🟢 Listening started…")
        except Exception as e:
            self.log(f"❌ Failed to open mic: {e}")
            return

        # Start async processing
        self.task = self.loop.create_task(
            LiveTalkAI(self, self.audio_stream, self.recognizer, self.synthesizer).run()
        )
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)

    def stop(self):
        """Stop transcription loop and mic stream."""
        if self.task:
            self.task.cancel()
        self.audio_stream.stop()
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.log("🛑 Stop requested.")

    def test_mic(self):
        """Record 3 seconds from the selected mic, save & play back."""
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


class LiveTalkAIEntryPoint:
    """Initializes and starts the GUI and asyncio loop."""

    def __init__(self):
        self.gui = None

    def run(self):
        root = tk.Tk()
        self.gui = LiveTalkAIGUI(root)
        threading.Thread(target=self._start_loop, daemon=True).start()
        root.mainloop()

    def _start_loop(self):
        asyncio.set_event_loop(self.gui.loop)
        self.gui.loop.run_forever()


if __name__ == "__main__":
    LiveTalkAIEntryPoint().run()
