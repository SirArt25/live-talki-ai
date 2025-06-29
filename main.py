import sys
import time
import asyncio
import threading
import tkinter as tk
from tkinter import scrolledtext, filedialog
import whisper
from openai import OpenAI
import sounddevice as sd
import queue
import tempfile
import scipy.io.wavfile as wavfile
import os
import logging
from dotenv import load_dotenv
from datetime import datetime
import textwrap
import numpy as np
from TTS.api import TTS

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
# Init OpenAI client
openai_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_api_key)

SAMPLE_RATE = 16000
CHUNK_DURATION = 2  # seconds
MAX_TEXT_CHUNK = 200  # characters per chunk


def split_text_chunks(text, max_length=MAX_TEXT_CHUNK):
    return textwrap.wrap(text, max_length)


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
    def __init__(self, model_name="base", use_openai=False):
        self.use_openai = use_openai
        if not self.use_openai:
            logging.info(f"📥 Loading Whisper model: {model_name}")
            self.model = whisper.load_model(model_name)
        else:
            logging.info("✅ Using OpenAI Whisper API.")

    async def transcribe(self, audio_data):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            wavfile.write(f.name, SAMPLE_RATE, audio_data)
            path = f.name
        if self.use_openai:
            resp = client.audio.transcriptions.create(model="whisper-1", file=open(path, "rb"))
            text = resp.text.strip()
        else:
            result = self.model.transcribe(path)
            text = result.get("text", "").strip()
        os.remove(path)
        logging.info(f"📝 Transcribed text: '{text}'")
        return split_text_chunks(text) if text else []


class LanguageProcessor:
    def __init__(self, model_name="gpt-3.5-turbo"):
        self.model = model_name
        self.conversation = []

    def generate_reply(self, user_text):
        self.conversation.append({"role": "user", "content": user_text})
        resp = client.chat.completions.create(
            model=self.model,
            messages=self.conversation,
            max_tokens=150,
        )
        reply = resp.choices[0].message.content.strip()
        self.conversation.append({"role": "assistant", "content": reply})
        logging.info(f"💬 GPT reply: '{reply}'")
        return reply


class SpeechSynthesizer:
    def __init__(self, tts_model):
        logging.info(f"🔊 Loading TTS model: {tts_model}")
        self.tts = TTS(model_name=tts_model, progress_bar=False, gpu=False)

    def speak(self, text):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            self.tts.tts_to_file(text=text, file_path=f.name)
            os.system(f"aplay {f.name}")
            time.sleep(0.1)
            os.remove(f.name)


class LiveTalkAIThread(threading.Thread):
    """Background thread to run asyncio event loop for live transcribe and chat."""
    def __init__(self, gui, recognizer, processor, synthesizer, device_idx):
        super().__init__(daemon=True)
        self.gui = gui
        self.stream = AsyncAudioStream(device=device_idx)
        self.recognizer = recognizer
        self.processor = processor
        self.synthesizer = synthesizer
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.task = None

    def run(self):
        self.stream.start()
        self.gui.log(f"🎤 Mic started (idx={self.stream.device})")
        self.task = self.loop.create_task(self._process_loop())
        self.loop.run_forever()

    async def _process_loop(self):
        self.gui.log("▶️ Live mode started.")
        try:
            while True:
                audio = await self.stream.get_chunk()
                chunks = await self.recognizer.transcribe(audio)
                for i, txt in enumerate(chunks, 1):
                    self.gui.log(f"You ({i}/{len(chunks)}): {txt}", tag="user")
                    reply = self.processor.generate_reply(txt)
                    self.gui.log(f"Bot: {reply}", tag="bot")
                    self.synthesizer.speak(reply)
        except asyncio.CancelledError:
            self.stream.stop()
            self.gui.log("🛑 Live mode stopped.")

    def stop(self):
        if self.task:
            self.task.cancel()
        if self.loop.is_running():
            self.loop.call_soon_threadsafe(self.loop.stop)
        self.stream.stop()


class LiveTalkAIGUI:
    def __init__(self, root):
        self.root = root
        root.title("LiveTalkAI")

        # Chat display
        self.text_area = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=60, height=20, font=("Arial",12))
        self.text_area.pack(padx=10, pady=10)
        self.text_area.config(state=tk.DISABLED)
        self.text_area.tag_config("user", foreground="blue")
        self.text_area.tag_config("bot", foreground="green")

        # Controls
        ctrl = tk.Frame(root); ctrl.pack(pady=5)
        tk.Label(ctrl, text="Whisper Model:").grid(row=0,column=0)
        wm=["tiny","base","small","medium","large"]
        self.whisper_var=tk.StringVar(value="base")
        tk.OptionMenu(ctrl, self.whisper_var,*wm).grid(row=0,column=1)
        tk.Label(ctrl, text="Use OpenAI:").grid(row=0,column=2)
        self.openai_var=tk.BooleanVar(value=False)
        tk.Checkbutton(ctrl,variable=self.openai_var).grid(row=0,column=3)
        tk.Label(ctrl, text="TTS Model:").grid(row=1,column=0)
        tm=["tts_models/en/ljspeech/tacotron2-DDC","tts_models/en/ljspeech/glow-tts"]
        self.tts_var=tk.StringVar(value=tm[0])
        tk.OptionMenu(ctrl,self.tts_var,*tm).grid(row=1,column=1)

        # Buttons
        btnf=tk.Frame(root);btnf.pack(pady=5)
        self.start_btn=tk.Button(btnf,text="Start",command=self.start_live)
        self.start_btn.pack(side=tk.LEFT,padx=5)
        self.stop_btn=tk.Button(btnf,text="Stop",command=self.stop_live,state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT,padx=5)
        # Entry & Ask mode - initially disabled
        self.ask_entry=tk.Entry(btnf,width=40,state=tk.DISABLED)
        self.ask_entry.pack(side=tk.LEFT,padx=5)
        self.ask_btn=tk.Button(btnf,text="Ask",command=self.ask_question,state=tk.DISABLED)
        self.ask_btn.pack(side=tk.LEFT,padx=5)

        # Mic select
        tk.Label(root,text="Select Mic:").pack()
        devices=[d['name'] for d in sd.query_devices() if d['max_input_channels']>0]
        self.dev_var=tk.StringVar(value=devices[0] if devices else "")
        tk.OptionMenu(root,self.dev_var,*devices).pack()
        self.log(f"✅ {len(devices)} mic(s) found")

        self.live_thread=None
        self.processor=None
        self.synthesizer=None
        root.protocol("WM_DELETE_WINDOW",self.on_close)

    def log(self,msg,tag=None):
        self.text_area.config(state=tk.NORMAL)
        if tag: self.text_area.insert(tk.END,msg+"\n",tag)
        else:   self.text_area.insert(tk.END,msg+"\n")
        self.text_area.config(state=tk.DISABLED)
        self.text_area.yview(tk.END)
        logging.info(msg)

    def start_live(self):
        # disable start, enable stop
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        # disable ask UI
        self.ask_entry.config(state=tk.DISABLED)
        self.ask_btn.config(state=tk.DISABLED)
        # init modules
        rec=SpeechRecognizer(self.whisper_var.get(), self.openai_var.get())
        proc=LanguageProcessor()
        synth=SpeechSynthesizer(self.tts_var.get())
        self.processor=proc; self.synthesizer=synth
        idx= [i for i,d in enumerate(sd.query_devices()) if d['name']==self.dev_var.get()][0]
        # start background thread
        self.live_thread=LiveTalkAIThread(self, rec, proc, synth, idx)
        self.live_thread.start()

    def stop_live(self):
        # stop background live thread
        if self.live_thread:
            self.live_thread.stop()
        # enable note-taking UI
        self.ask_entry.config(state=tk.NORMAL)
        self.ask_btn.config(state=tk.NORMAL)
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.log("🛑 Live mode stopped. Note-taking enabled.")

    def ask_question(self):
        question=self.ask_entry.get().strip()
        if not question: return
        self.log(f"You: {question}", tag="user")
        def thread_fn():
            reply=self.processor.generate_reply(question)
            self.log(f"Bot: {reply}", tag="bot")
            self.synthesizer.speak(reply)
        threading.Thread(target=thread_fn,daemon=True).start()
        self.ask_entry.delete(0,tk.END)

    def on_close(self):
        # ensure live thread stopped
        if self.live_thread:
            self.live_thread.stop()
        self.root.destroy()

    def test_mic(self):pass
    def clear(self):pass
    def save(self):pass

class LiveTalkAIEntryPoint:
    def run(self):
        root=tk.Tk(); LiveTalkAIGUI(root); root.mainloop()

if __name__=='__main__':
    LiveTalkAIEntryPoint().run()
