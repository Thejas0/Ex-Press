import speech_recognition as sr 
import os 
from pydub import AudioSegment
from pydub.silence import split_on_silence

filename = "temp_audio.wav"

r = sr.Recognizer()

# open the file
with sr.AudioFile(filename) as source:
    # listen for the data (load audio to memory)
    audio_data = r.record(source)
    # recognize (convert from speech to text)
    text = r.recognize_google(audio_data)
    
    print("The text recognized was : ",text)
    audio_data_string = text



def word_count(audio_data_string):
    counts = {}
    words = audio_data_string.split()

    for word in words:
        if word in counts:
            counts[word] += 1
        else:
            counts[word] = 1

    return counts

#print( word_count('the quick brown fox jumps over the lazy dog.'))

print(word_count(audio_data_string))

   