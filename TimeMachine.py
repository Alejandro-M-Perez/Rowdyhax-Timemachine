import asyncio
from dotenv import load_dotenv
import shutil
import subprocess
import requests
import time
import os
import json
from PIL import Image
from groq import Groq

from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chains import LLMChain

from deepgram import (
    DeepgramClient,
    DeepgramClientOptions,
    LiveTranscriptionEvents,
    LiveOptions,
    Microphone,
    SpeakOptions
)

import pyaudio as pa
import wave
#from playsound import playsound

load_dotenv()

client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)

name = input("Enter the name of the historical figure you want to talk to: ")

print('Was ' + name + ' a man, answer with one word "True" or "False"')

setup_question_1 = [
        {
            "role": "system",
            "content": 'Is ' + name + ' male, answer "True" or "False"',
        }
]

setup_question_completion = client.chat.completions.create(
    model="llama-3.2-3b-preview",
    messages=setup_question_1,
    temperature=0,
    max_tokens=1024,
    top_p=1,
    stream=False,
)

print("The model says: " + setup_question_completion.choices[0].message.content)
output = setup_question_completion.choices[0].message.content

if output == "True":
    model = "aura-helios-en"
else:
    model = "aura-athena-en"

print('The model is ' + model)
class Image_Finder():
    my_folder = 'wiki_images'
    def __init__(self, name):
        self.name = name
    def delete_files_on_start(wiki_images):
        """Deletes the specified files upon program start."""

        for file_path in wiki_images:
            try:
                os.remove(file_path)
                #print(f"File '{file_path}' deleted successfully.")
            except FileNotFoundError:
                print(f"File '{file_path}' not found.")
            except Exception as e:
                print(f"Error deleting '{file_path}': {e}")
    # set the folder name where images will be stored
    wiki_images = [os.path.join('wiki_images', f) for f in os.listdir('wiki_images') if os.path.isfile(os.path.join('wiki_images', f))]
    delete_files_on_start(wiki_images)

    # create the folder in the current working directory
    # in which to store the downloaded images
    os.makedirs(my_folder, exist_ok=True)

    # front part of each Wikipedia URL
    base_url = 'https://en.wikipedia.org/wiki/'

    # partial URLs for each desired Wikipedia page

    # Wikipedia API query string to get the main image on a page
    # (partial URL will be added to the end)
    query = 'http://en.wikipedia.org/w/api.php?action=query&prop=pageimages&format=json&piprop=original&titles='

    # get JSON data w/ API and extract image URL
    def get_image_url(partial_url):
        try:
            api_res = requests.get('http://en.wikipedia.org/w/api.php?action=query&prop=pageimages&format=json&piprop=original&titles=' + partial_url).json()
            first_part = api_res['query']['pages']
            # this is a way around not knowing the article id number
            for key, value in first_part.items():
                if (value['original']['source']):
                    data = value['original']['source']
                    return data
        except Exception as exc:
            print(exc)
            print("Partial URL: " + partial_url)
            data = None
        return data

    # download one image with URL obtained from API
    def download_image(the_url, the_page, my_folder):
        headers = {'User-Agent': 'RowdyHack_2024  Alejandromperez714@gmail.com)'}
        res = requests.get(the_url, headers=headers)
        res.raise_for_status()

        # get original file extension for image
        # by splitting on . and getting the final segment
        file_ext = '.' + the_url.split('.')[-1].lower()

        image_file = open(os.path.join(my_folder, os.path.basename("Input_Face" + file_ext)), 'wb')
        

        

        # download the image file 
        # HT to Automate the Boring Stuff with Python, chapter 12 
        for chunk in res.iter_content(100000):
            image_file.write(chunk)
        image_file.close()
        image=Image.open('wiki_images/Input_Face.jpg')
        resized_image = image.resize((1080, 1080), Image.LANCZOS)
        resized_image.save("wiki_images/Input_Face.jpg")
    # loop to download main image for each page in list

    # get JSON data and extract image URL
    the_url = get_image_url(name)
    # if the URL is not None ...
    if (the_url):
        # tell us where we are for the heck of it
        
        # download that image
        download_image(the_url, name, my_folder)
    else:
        print("No image file for " + name)

system_prompt = f"You are {name}. You are a historical figure. a time traveler just arrived to your time in a silver car. keep your respose short and simple. keep the conversation going. keep your responses short"

class LanguageModelProcessor:
    def __init__(self):
        self.llm = ChatGroq(temperature=0, model_name="mixtral-8x7b-32768", groq_api_key=os.getenv("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"))
        # self.llm = ChatOpenAI(temperature=0, model_name="gpt-4-0125-preview", openai_api_key=os.getenv("OPENAI_API_KEY"))
        # self.llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-0125", openai_api_key=os.getenv("OPENAI_API_KEY"))

        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        
        # Create a prompt template for the conversation
        self.prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{text}")
        ])

        self.conversation = LLMChain(
            llm=self.llm,
            prompt=self.prompt,
            memory=self.memory
        )

    def process(self, text):
        self.memory.chat_memory.add_user_message(text)  # Add user message to memory

        start_time = time.time()

        # Go get the response from the LLM
        response = self.conversation.invoke({"text": text})
        end_time = time.time()

        self.memory.chat_memory.add_ai_message(response['text'])  # Add AI response to memory

        elapsed_time = int((end_time - start_time) * 1000)
        print(f"LLM ({elapsed_time}ms): {response['text']}")
        return response['text']

class TextToSpeech:
    # Set your Deepgram API Key and desired voice model
    DG_API_KEY = os.getenv("DEEPGRAM_API_KEY")
    MODEL_NAME = "aura-helios-en"  # Example model name, change as needed

    @staticmethod
    def is_installed(lib_name: str) -> bool:
        lib = shutil.which(lib_name)
        return lib is not None

    def speak(self, text):
       

        DEEPGRAM_URL = f"https://api.deepgram.com/v1/speak?model={self.MODEL_NAME}&performance=some&encoding=linear16&sample_rate=24000"
        headers = {
            "Authorization": f"Token {self.DG_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "text": text
        }

        try:
            SPEAK_OPTIONS = {"text": text}
            filename = "output.wav"

            # STEP 1: Create a Deepgram client using the API key from environment variables
            deepgram = DeepgramClient(api_key=os.getenv("DG_API_KEY"))

            # STEP 2: Configure the options (such as model choice, audio configuration, etc.)
            options = SpeakOptions(
                model=model,
                encoding="linear16",
                container="wav"
            )

            # STEP 3: Call the save method on the speak property
            response = deepgram.speak.v("1").save(filename, SPEAK_OPTIONS, options)
            # Initialize PyAudio
            p = pa.PyAudio()

            # Open the WAV file
            wf = wave.open(filename, 'rb')

            # Define the callback function to play the audio
            def callback(in_data, frame_count, time_info, status):
                data = wf.readframes(frame_count)
                return (data, pa.paContinue)

            # Open a stream with the appropriate settings
            stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                            channels=wf.getnchannels(),
                            rate=wf.getframerate(),
                            output=True,
                            stream_callback=callback)

            # Start the stream
            stream.start_stream()

            # Wait for the stream to finish
            while stream.is_active():
                time.sleep(0.01)

            # Stop and close the stream
            stream.stop_stream()
            stream.close()

            # Close PyAudio
            p.terminate()

            # Close the WAV file
            wf.close()
            #os.remove("output.wav")
            #print(response.to_json(indent=4))

            player_command = ["ffplay", "-autoexit", "-", "-nodisp"]
            player_process = subprocess.Popen(
                player_command,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

        except Exception as e:
            print(f"Exception: {e}")

        start_time = time.time()  # Record the time before sending the request
        first_byte_time = None  # Initialize a variable to store the time when the first byte is received

        '''with requests.post(DEEPGRAM_URL, stream=True, headers=headers, json=payload) as r:
            for chunk in r.iter_content(chunk_size=1024):
                if chunk:
                    if first_byte_time is None:  # Check if this is the first chunk received
                        first_byte_time = time.time()  # Record the time when the first byte is received
                        ttfb = int((first_byte_time - start_time)*1000)  # Calculate the time to first byte
                        print(f"TTS Time to First Byte (TTFB): {ttfb}ms\n")
                    player_process.stdin.write(chunk)
                    player_process.stdin.flush()

        if player_process.stdin:
            player_process.stdin.close()
        player_process.wait()'''

class TranscriptCollector:
    def __init__(self):
        self.reset()

    def reset(self):
        self.transcript_parts = []

    def add_part(self, part):
        self.transcript_parts.append(part)

    def get_full_transcript(self):
        return ' '.join(self.transcript_parts)

transcript_collector = TranscriptCollector()

async def get_transcript(callback):
    transcription_complete = asyncio.Event()  # Event to signal transcription completion

    try:
        # example of setting up a client config. logging values: WARNING, VERBOSE, DEBUG, SPAM
        config = DeepgramClientOptions(options={"keepalive": "true"})
        deepgram: DeepgramClient = DeepgramClient("", config)

        dg_connection = deepgram.listen.asynclive.v("1")
        print ("Listening...")

        async def on_message(self, result, **kwargs):
            sentence = result.channel.alternatives[0].transcript
            
            if not result.speech_final:
                transcript_collector.add_part(sentence)
            else:
                # This is the final part of the current sentence
                transcript_collector.add_part(sentence)
                full_sentence = transcript_collector.get_full_transcript()
                # Check if the full_sentence is not empty before printing
                if len(full_sentence.strip()) > 0:
                    full_sentence = full_sentence.strip()
                    print(f"Human: {full_sentence}")
                    callback(full_sentence)  # Call the callback with the full_sentence
                    transcript_collector.reset()
                    transcription_complete.set()  # Signal to stop transcription and exit

        dg_connection.on(LiveTranscriptionEvents.Transcript, on_message)

        options = LiveOptions(
            model="nova-2",
            punctuate=True,
            language="en-US",
            encoding="linear16",
            channels=1,
            sample_rate=16000,
            endpointing=300,
            smart_format=True,
        )

        await dg_connection.start(options)

        # Open a microphone stream on the default input device
        microphone = Microphone(dg_connection.send)
        microphone.start()

        await transcription_complete.wait()  # Wait for the transcription to complete instead of looping indefinitely

        # Wait for the microphone to close
        microphone.finish()

        # Indicate that we've finished
        await dg_connection.finish()

    except Exception as e:
        print(f"Could not open socket: {e}")
        return

class ConversationManager:
    def __init__(self):
        self.transcription_response = ""
        self.llm = LanguageModelProcessor()

    async def main(self):
        def handle_full_sentence(full_sentence):
            self.transcription_response = full_sentence

        # Loop indefinitely until "goodbye" is detected
        while True:
            await get_transcript(handle_full_sentence)
            
            # Check for "goodbye" to exit the loop
            if "goodbye" in self.transcription_response.lower():
                break
            
            llm_response = self.llm.process(self.transcription_response)

            tts = TextToSpeech()
            tts.speak(llm_response)

            # Reset transcription_response for the next loop iteration
            self.transcription_response = ""

if __name__ == "__main__":
    manager = ConversationManager()
    asyncio.run(manager.main())
class FaceMaker: 
    files = [
        ("input_face", open("wiki_images\Input_Face.jpg", "rb")),
        ("input_audio", open("output.wav", "rb")),
    ]
    payload = {}

    response = requests.post(
        "https://api.gooey.ai/v2/Lipsync/form/?run_id=fecsii61rs6e&uid=fm165fOmucZlpa5YHupPBdcvDR02",
        headers={
            "Authorization": "Bearer " + os.environ["GOOEY_API_KEY"],
        },
        files=files,
        data={"json": json.dumps(payload)},
    )
    assert respossnse.ok, response.content

    video_url = "https://api.gooey.ai/v2/Lipsync/form/?run_id=fecsii61rs6e&uid=fm165fOmucZlpa5YHupPBdcvDR02"

    # Send a GET request to the URL
    response = requests.get(video_url, headers={
        "Authorization": "Bearer " + os.environ["GOOEY_API_KEY"],
    }, stream=True)

    if response.status_code == 200:
        # Save the content to a file in chunks
        with open("downloaded_video.mp4", "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        print("Video downloaded successfully.")
    else:
        print(f"Failed to download video. Status code: {response.status_code}")