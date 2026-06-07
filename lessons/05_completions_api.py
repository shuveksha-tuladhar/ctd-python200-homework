from dotenv import load_dotenv
from openai import OpenAI
from pprint import pprint

if load_dotenv():
    print("Successfully loaded api key")
    
client = OpenAI()

response = client.chat.completions.create(model="gpt-4o-mini",
                                          messages = [{"role": "user",
                                                      "content": "Hello World"}],
                                          n=1,
                                          temperature=1.3)
print(response.choices[0].message.content)
print(response.usage)
print(response.model)


from pathlib import Path
from IPython.display import Audio

voice = "alloy"  #shimmer is higher pitch, onyx lower pitch

speech_file_path = Path("speech.mp3")

input_text1 = "Python is an amazing programming language."
input_text2 = "But it can't take my dog for a walk.... Or feed my fish."
input_text3 = "पाइथन एकदम राम्रो प्रोग्रामिङ भाषा हो। तर यसले मेरो कुकुरलाई घुमाउन वा माछालाई खाना खुवाउन सक्दैन।"
input_text = input_text1 + input_text2 + input_text3

try:
    with client.audio.speech.with_streaming_response.create(
        model="tts-1",
        voice=voice,
        input=input_text
    ) as speech_response:
        speech_response.stream_to_file(speech_file_path)
except openai.BadRequestError:
    print("Invalid voice selected: {voice}")

print(f"Audio saved as {speech_file_path} with voice of {voice}")

Audio(str(speech_file_path))


