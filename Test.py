import os
import pyttsx3
from gtts import gTTS
import subprocess
from playsound import playsound

audio = 'speech.mp3'
language = 'en-uk'


speech = gTTS(text = "I will speak this text", lang = language, slow = False)
speech.save(audio)
playsound(audio)

def play_stream(audio_stream, use_ffmpeg=True):
    player = "ffplay"
    if not is_installed(player):
        raise ValueError(f"{player} not found, necessary to stream audio.")
    
    player_command = ["ffplay", "-autoexit", "-", "-nodisp"]
    player_process = subprocess.Popen(
        player_command,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    for chunk in audio_stream:
        if chunk:
            player_process.stdin.write(chunk)  # type: ignore
            player_process.stdin.flush()  # type: ignore
    
    if player_process.stdin:
        player_process.stdin.close()
    player_process.wait()