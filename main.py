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

# Setup fresh log per run
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f"livelog_{timestamp}.log"
logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filemode="w"
)

# Load .env
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

SAMPLE_RATE = 16000
CHUNK_DURATION = 2


class AsyncAudioStream:
    def __init__(self, sample_rate=SAMPLE_RATE, chunk_duration=CHUNK_DURATION, device=None):
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.device = device
        self.queue = queue.Queue()
        self.stream = None

    def callback(self, indata, frames, time_info, status):
        self.queue.put(indata.copy())
        logging.info("🔔 Audio callback received")

    def start(self):
        logging.info("🔍 Starting mic input...")
        if self.device is None:
            raise RuntimeError("❌ No mic device index provided.")
        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            callback=self.callback,
            device=self.device
        )
        self.stream.start()
        logging.info("🔊 Mic stream started.")

    def stop(self):
        if self.stream:
            self.stream.stop()
            self.stream.close()
            logging.info("🛑 Mic stream stopped.")

    async def get_chunk(self):
        while True:
            if not self.queue.empty():
                return self.queue.get().flatten()
            await asyncio.sleep(0.01)


class SpeechRecognizer:
    def __init__(self):
        logging.info("📥 Loading Whisper model…")
        self.model = whisper.load_model("base")
        logging.info("✅ Whisper loaded.")

    async def transcribe(self, audio_data):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            wavfile.write(f.name, SAMPLE_RATE, audio_data)
            result = self.model.transcribe(f.name)
            os.remove(f.name)
        return result["text"]


class LanguageProcessor:
    def __init__(self, model="gpt-3.5-turbo"):
        self.model = model

    async def generate_reply(self, text):
        response = await asyncio.to_thread(
            openai.ChatCompletion.create,
            model=self.model,
            messages=[{"role": "user", "content": text}],
            max_tokens=100
        )
        return response.choices[0].message.content


class SpeechSynthesizer:
    def __init__(self):
        logging.info("🔊 Loading Coqui TTS…")
        self.tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False, gpu=False)
        logging.info("✅ Coqui TTS loaded.")

    async def speak(self, text):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            self.tts.tts_to_file(text=text, file_path=f.name)
            os.system(f"aplay {f.name}")
            os.remove(f.name)


class LiveTalkAI:
    def __init__(self, gui, device_index, recognizer, synthesizer):
        self.audio_stream = AsyncAudioStream(device=device_index)
        self.recognizer = recognizer
        self.processor = LanguageProcessor()
        self.synthesizer = synthesizer
        self.gui = gui

    async def run(self):
        self.audio_stream.start()
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
        self.status_label.pack()

        # Buttons: Start / Stop / Test Mic
        btns = tk.Frame(root)
        btns.pack(pady=5)
        self.start_button = tk.Button(
            btns, text="Start", command=self.start,
            state=tk.NORMAL if mic_names else tk.DISABLED
        )
        self.start_button.pack(side=tk.LEFT, padx=5)
        self.stop_button = tk.Button(
            btns, text="Stop", command=self.stop, state=tk.DISABLED
        )
        self.stop_button.pack(side=tk.LEFT, padx=5)
        self.test_button = tk.Button(
            btns, text="Test Mic", command=self.test_mic,
            state=tk.NORMAL if mic_names else tk.DISABLED
        )
        self.test_button.pack(side=tk.LEFT, padx=5)

        # Async setup
        self.loop = asyncio.new_event_loop()
        self.task = None
        self.app = None

    def log(self, msg):
        logging.info(msg)
        self.text_area.config(state=tk.NORMAL)
        self.text_area.insert(tk.END, msg + "\n")
        self.text_area.config(state=tk.DISABLED)
        self.text_area.yview(tk.END)

    def start(self):
        sel = self.device_var.get()
        idx = next((i for n, i in self.input_devices if n == sel), None)
        self.log(f"🎤 Mic selected: {sel}")
        self.log("🟢 Listening started…")

        self.app = LiveTalkAI(self, idx, self.recognizer, self.synthesizer)
        self.task = self.loop.create_task(self.app.run())
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)

    def stop(self):
        if self.task:
            self.task.cancel()
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)

    def test_mic(self):
        """Record 3 seconds from the selected mic, save & play back."""
        sel = self.device_var.get()
        idx = next((i for n, i in self.input_devices if n == sel), None)
        threading.Thread(target=self._test_thread, args=(idx,), daemon=True).start()

    def _test_thread(self, device_index):
        from scipy.io.wavfile import write
        duration = 3  # seconds
        self.log(f"🎤 Testing mic for {duration}s on device #{device_index}...")
        try:
            audio = sd.rec(
                int(duration * SAMPLE_RATE),
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype='int16',
                device=device_index
            )
            sd.wait()
            write("mic_test.wav", SAMPLE_RATE, audio)
            self.log("💾 mic_test.wav saved.")
            os.system("aplay mic_test.wav")
            self.log("▶️ mic_test.wav played back.")
        except Exception as e:
            self.log(f"❌ Test failed: {e}")



def test_recorder():
    devs = sd.query_devices()
    inputs = [i for i, d in enumerate(devs) if d["max_input_channels"] > 0]
    if not inputs:
        logging.error("❌ No mic devices found.")
        return
    device = inputs[0]
    logging.info(f"🎧 test_recorder using device #{device} ({devs[device]['name']})")
    stream = AsyncAudioStream(device=device)
    stream.start()
    time.sleep(3)
    stream.stop()
    logging.info("✅ test_recorder completed 3 seconds.")


def test_microphone_recording(output_path="test_recording.wav", duration=3, sample_rate=SAMPLE_RATE):
    """Record audio from the default mic and save to a WAV file."""
    import numpy as np
    mic_list = sd.query_devices()
    inputs = [i for i, d in enumerate(mic_list) if d["max_input_channels"] > 0]
    if not inputs:
        logging.error("❌ No input-capable device found for test_microphone_recording.")
        return
    device = inputs[0]
    logging.info(f"🔍 test_microphone_recording: using device #{device} ({mic_list[device]['name']})")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='int16', device=device)
    sd.wait()
    wavfile.write(output_path, sample_rate, audio)
    logging.info(f"💾 Saved recording to {output_path}")


class LiveTalkAIEntryPoint:
    def __init__(self):
        self.gui = None

    def run(self):
        if len(sys.argv) > 1:
            arg = sys.argv[1]
            if arg == "--test-recorder":
                test_recorder()
                return
            elif arg == "--test-recording":
                test_microphone_recording()
                return

        root = tk.Tk()
        self.gui = LiveTalkAIGUI(root)
        threading.Thread(target=self._start_loop, daemon=True).start()
        root.mainloop()

    def _start_loop(self):
        asyncio.set_event_loop(self.gui.loop)
        self.gui.loop.run_forever()


if __name__ == "__main__":
    LiveTalkAIEntryPoint().run()
