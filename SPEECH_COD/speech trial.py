from __future__ import division

import re
import sys

from google.cloud import speech
from google.cloud.speech import enums
from google.cloud.speech import types
import pyaudio
from six.moves import queue
import speech_recognition as sr


import numpy as np
import cv2


import re
import sys
import threading,time,sys
from google.cloud import speech
from google.cloud.speech import enums
from google.cloud.speech import types
import pyaudio
from six.moves import queue

# Audio recording parameters
RATE = 16000
CHUNK = int(RATE / 40)  # 100ms


class MicrophoneStream(object):
    """Opens a recording stream as a generator yielding the audio chunks."""
    def __init__(self, rate, chunk):
        self._rate = rate
        self._chunk = chunk

        # Create a thread-safe buffer of audio data
        self._buff = queue.Queue()
        self.closed = True

    def __enter__(self):
        self._audio_interface = pyaudio.PyAudio()
        self._audio_stream = self._audio_interface.open(
            format=pyaudio.paInt16,
            # The API currently only supports 1-channel (mono) audio
            # https://goo.gl/z757pE
            channels=1, rate=self._rate,
            input=True, frames_per_buffer=self._chunk,
            # Run the audio stream asynchronously to fill the buffer object.
            # This is necessary so that the input device's buffer doesn't
            # overflow while the calling thread makes network requests, etc.
            stream_callback=self._fill_buffer,
        )

        self.closed = False

        return self

    def __exit__(self, type, value, traceback):
        self._audio_stream.stop_stream()
        self._audio_stream.close()
        self.closed = True
        # Signal the generator to terminate so that the client's
        # streaming_recognize method will not block the process termination.
        self._buff.put(None)
        self._audio_interface.terminate()

    def _fill_buffer(self, in_data, frame_count, time_info, status_flags):
        """Continuously collect data from the audio stream, into the buffer."""
        self._buff.put(in_data)
        return None, pyaudio.paContinue

    def generator(self):
        while not self.closed:
            # Use a blocking get() to ensure there's at least one chunk of
            # data, and stop iteration if the chunk is None, indicating the
            # end of the audio stream.
            chunk = self._buff.get()
            if chunk is None:
                return
            data = [chunk]

            # Now consume whatever other data's still buffered.
            while True:
                try:
                    chunk = self._buff.get(block=False)
                    if chunk is None:
                        return
                    data.append(chunk)
                except queue.Empty:
                    break

            yield b''.join(data)


def listen_print_loop(responses):


    words_key=dict({
#                        "bump":"G",
#                        "bob":"G",
#                        "mom":"G",
                        "pro":"G",
#                        "true":"G",
#                        "bomb":"G",
                        "throw":"G",
#                        "turn":"G",
#                        "through":"G",
#                        "row":"G",
#                        "set":"C",
                        "cross":"C",
#                        "seat":"C",
#                        "downset":"C",
#                        "sit":"C",
                        "down":"C",
#                        "aim":"right click",
#                        "in":"right click",
                        "inside":"right click",
#                        "insight":"right click",
#                        "real":"R",
#                        "ral":"R",
#                        "or":"R",
                        "lord":"R",
#                        "trailer":"R",
#                        "halo":"R",
#                        "allure":"R",
#                        "alone":"R",
#                        "lorde":"R",
#                        "lordure":"R",
                        "change":"2",
#                        "swap":"2",
#                        "swept":"2"
                        "rifle":"3"
                    })

    num_chars_printed = 0
    for response in responses:
        if not response.results:
            continue

        result = response.results[0]
        if not result.alternatives:
            continue

        transcript = result.alternatives[0].transcript
        overwrite_chars = ' ' * (num_chars_printed - len(transcript))


        transcript_list=set(transcript.lower().split(" "))
        print(transcript_list)
        for i in transcript_list:
            if (i in words_key):
                print(words_key[i])
                time.sleep(1)

        if not result.is_final:

#            transcript_list=set(transcript.split(" "))
#            print(transcript_list)
#            for i in transcript_list:
#                if (i in words_key):
#                    print(words_key[i])

#            sys.stdout.write(transcript + overwrite_chars + '\r')
            sys.stdout.flush()
#            print(type(transcript))
            num_chars_printed = len(transcript)

        else:

#            print(type(transcript))
#            print(transcript + overwrite_chars)

            if re.search(r'\b(exit|quit)\b', transcript, re.I):
                print('Exiting..')
                break

            num_chars_printed = 0





def main2():

    language_code = 'en-US'  # a BCP-47 language tag

    client = speech.SpeechClient()
    config = types.RecognitionConfig(
        encoding=enums.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=RATE,
        language_code=language_code,)
    streaming_config = types.StreamingRecognitionConfig(
        config=config,
        interim_results=True)

    with MicrophoneStream(RATE, CHUNK) as stream:
        audio_generator = stream.generator()
        requests = (types.StreamingRecognizeRequest(audio_content=content)
                    for content in audio_generator)

        responses = client.streaming_recognize(streaming_config, requests)

        # Now, put the transcription responses to use.

        print("here")

        listen_print_loop(responses)
        time.sleep(0.01)
        print("here2")


main2()