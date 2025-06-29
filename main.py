import sys
import time
import asyncio
import threading
import tkinter as tk
from tkinter import scrolledtext
import whisper
from openai import OpenAI
import sounddevice as sd
import tempfile
import scipy.io.wavfile as wavfile
import os
import logging
from dotenv import load_dotenv
from datetime import datetime
import textwrap
import numpy as np
import subprocess
from TTS.api import TTS

# ——— Setup logging ———
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
logging.basicConfig(
    filename=f"livelog_{timestamp}.log",
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filemode="w",
)

# ——— Load API key ———
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ——— Constants ———
SAMPLE_RATE = 16000
CHUNK_DURATION = 2       # seconds; only used to size the buffer
MAX_TEXT_CHUNK = 200     # chars per split


def split_text_chunks(text, max_length=MAX_TEXT_CHUNK):
    return textwrap.wrap(text, max_length)

# ——— Audio buffer ———
class AsyncAudioStream:
    def __init__(self, sample_rate=SAMPLE_RATE, device=None):
        self.sample_rate = sample_rate
        self.device = device
        self.stream = None
        self.record_buffer = []

    def callback(self, indata, frames, time_info, status):
        data = indata.copy().flatten()
        self.record_buffer.append(data)
        logging.info("🔔 Audio callback received")

    def start(self):
        self.record_buffer = []
        blocksize = int(self.sample_rate * CHUNK_DURATION)
        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            callback=self.callback,
            device=self.device,
            blocksize=blocksize
        )
        self.stream.start()
        logging.info("🔊 Recording started.")

    def stop(self):
        if self.stream:
            self.stream.stop()
            self.stream.close()
            logging.info("🛑 Recording stopped.")

    def get_full_recording(self):
        return np.concatenate(self.record_buffer) if self.record_buffer else np.array([], dtype=np.float32)

# ——— Whisper wrapper ———
class SpeechRecognizer:
    def __init__(self, model_name="base", use_openai=False):
        self.use_openai = use_openai
        if not use_openai:
            self.model = whisper.load_model(model_name)
            logging.info(f"📥 Loaded Whisper model: {model_name}")
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

# ——— GPT conversation ———
class LanguageProcessor:
    def __init__(self, model_name="gpt-3.5-turbo"):
        self.model = model_name
        self.conversation = []

    def generate_reply(self, user_text):
        self.conversation.append({"role": "user", "content": user_text})
        resp = client.chat.completions.create(model=self.model, messages=self.conversation, max_tokens=150)
        reply = resp.choices[0].message.content.strip()
        self.conversation.append({"role": "assistant", "content": reply})
        logging.info(f"💬 GPT reply: '{reply}'")
        return reply

# ——— TTS ———
class SpeechSynthesizer:
    def __init__(self, tts_model):
        logging.info(f"🔊 Loading TTS model: {tts_model}")
        self.tts = TTS(model_name=tts_model, progress_bar=False, gpu=False)
        self.processes = []

    def speak(self, text, audio_on=True):
        if not audio_on:
            return
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            self.tts.tts_to_file(text=text, file_path=f.name)
            p = subprocess.Popen(["aplay", f.name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            self.processes.append((p, f.name))

    def stop_all(self):
        for p, path in self.processes:
            try: p.terminate()
            except: pass
            try: os.remove(path)
            except: pass
        self.processes.clear()

# ——— Main GUI ———
class LiveTalkAIGUI:
    def __init__(self, root):
        self.root = root
        root.title("LiveTalkAI")
        self.text_area = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=60, height=20, font=("Arial",12))
        self.text_area.pack(padx=10, pady=10)
        self.text_area.config(state=tk.DISABLED)
        self.text_area.tag_config("user", foreground="blue")
        self.text_area.tag_config("bot", foreground="green")
        ctrl = tk.Frame(root); ctrl.pack(pady=5)
        tk.Label(ctrl, text="Whisper Model:").grid(row=0,column=0)
        wm=["tiny","base","small","medium","large"]
        self.whisper_var=tk.StringVar(value="base")
        tk.OptionMenu(ctrl,self.whisper_var,*wm).grid(row=0,column=1)
        tk.Label(ctrl,text="Use OpenAI:").grid(row=0,column=2)
        self.openai_var=tk.BooleanVar(value=False)
        tk.Checkbutton(ctrl,variable=self.openai_var).grid(row=0,column=3)
        tk.Label(ctrl,text="TTS Model:").grid(row=1,column=0)
        tm=["tts_models/en/ljspeech/tacotron2-DDC","tts_models/en/ljspeech/glow-tts"]
        self.tts_var=tk.StringVar(value=tm[0])
        tk.OptionMenu(ctrl,self.tts_var,*tm).grid(row=1,column=1)
        tk.Label(ctrl,text="Audio On:").grid(row=1,column=2)
        self.audio_var=tk.BooleanVar(value=True)
        tk.Checkbutton(ctrl,variable=self.audio_var).grid(row=1,column=3)
        btnf=tk.Frame(root); btnf.pack(pady=5)
        self.start_btn=tk.Button(btnf,text="Start",command=self.start_live)
        self.start_btn.pack(side=tk.LEFT,padx=5)
        self.stop_btn=tk.Button(btnf,text="Stop",command=self.stop_live,state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT,padx=5)
        self.ask_entry=tk.Entry(btnf,width=40,state=tk.DISABLED)
        self.ask_entry.pack(side=tk.LEFT,padx=5)
        self.ask_btn=tk.Button(btnf,text="Ask",command=self.ask_question,state=tk.DISABLED)
        self.ask_btn.pack(side=tk.LEFT,padx=5)
        tk.Label(root,text="Select Mic:").pack()
        devices=[d['name'] for d in sd.query_devices() if d['max_input_channels']>0]
        self.dev_var=tk.StringVar(value=devices[0] if devices else "")
        tk.OptionMenu(root,self.dev_var,*devices).pack()
        self.log(f"✅ {len(devices)} mic(s) found")
        self.stream=self.recognizer=self.processor=self.synthesizer=None
        root.protocol("WM_DELETE_WINDOW",self.on_close)

    def log(self,msg,tag=None):
        self.text_area.config(state=tk.NORMAL)
        if tag: self.text_area.insert(tk.END,msg+"\n",tag)
        else: self.text_area.insert(tk.END,msg+"\n")
        self.text_area.config(state=tk.DISABLED)
        self.text_area.yview(tk.END)
        logging.info(msg)

    def start_live(self):
        # Stop any ongoing speech
        if self.synthesizer:
            self.synthesizer.stop_all()
        # UI state
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.ask_entry.config(state=tk.DISABLED)
        self.ask_btn.config(state=tk.DISABLED)
        # init modules
        self.recognizer=SpeechRecognizer(self.whisper_var.get(),self.openai_var.get())
        self.processor=LanguageProcessor()
        self.synthesizer=SpeechSynthesizer(self.tts_var.get())
        idx=[i for i,d in enumerate(sd.query_devices()) if d['name']==self.dev_var.get()][0]
        self.stream=AsyncAudioStream(device=idx)
        self.stream.start()
        self.log("🎤 Recording…")

    def stop_live(self):
        self.stream.stop()
        self.log("🛑 Stopped recording; processing…")
        audio=self.stream.get_full_recording()
        if audio.size==0:
            self.log("⚠️ No audio captured.")
        else:
            chunks=asyncio.run(self.recognizer.transcribe(audio))
            for txt in chunks:
                self.log(f"User: {txt}",tag="user")
                reply=self.processor.generate_reply(txt)
                self.log(f"Bot: {reply}",tag="bot")
                threading.Thread(target=lambda r=reply,a=self.audio_var.get(): self.synthesizer.speak(r,a),daemon=True).start()
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.ask_entry.config(state=tk.NORMAL)
        self.ask_btn.config(state=tk.NORMAL)

    def ask_question(self):
        q=self.ask_entry.get().strip()
        if not q: return
        self.log(f"User: {q}",tag="user")
        def worker():
            reply=self.processor.generate_reply(q)
            self.root.after(0,lambda: self.log(f"Bot: {reply}",tag="bot"))
            threading.Thread(target=lambda r=reply,a=self.audio_var.get(): self.synthesizer.speak(r,a),daemon=True).start()
        threading.Thread(target=worker,daemon=True).start()
        self.ask_entry.delete(0,tk.END)

    def on_close(self):
        if self.stream: self.stream.stop()
        if self.synthesizer: self.synthesizer.stop_all()
        self.root.destroy()

class LiveTalkAIEntryPoint:
    def run(self):
        root=tk.Tk()
        LiveTalkAIGUI(root)
        root.mainloop()

if __name__=='__main__':
    LiveTalkAIEntryPoint().run()
