from __future__ import division

import re
import sys

from google.cloud import speech
from google.cloud.speech import enums
from google.cloud.speech import types
import pyaudio
from six.moves import queue
import numpy as np
import cv2
import threading,time
import win32api, win32con

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
        interim_results=False)

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





class ProcessMain:
    play=1
    z_index_area=500
    sensitivity=2

    def get_masks(self,frame):
        yellow_frame=frame.copy()
        hand_yellow_lower=np.array([14,73,61])
        hand_yellow_upper=np.array([32,255,255])
        yellow_blur_filter=23
        frame_yellow_blurred=cv2.blur(yellow_frame,(yellow_blur_filter,yellow_blur_filter))
        yellow_hsv_blurred=cv2.cvtColor(frame_yellow_blurred,cv2.COLOR_BGR2HSV)
        mask_yellow=cv2.inRange(yellow_hsv_blurred,hand_yellow_lower,hand_yellow_upper)

        red_frame=frame.copy()
        hand_red_lower=np.array([171,67,0])
        hand_red_upper=np.array([179,255,255])
        red_brightness_filter=64
        red_blur_filter=16
        red_brightness_kernel=np.ones(red_frame.shape,dtype="uint8")*red_brightness_filter
        red_frame_brightness=cv2.add(red_frame,red_brightness_kernel)
        frame_red_blurred=cv2.blur(red_frame_brightness,(red_blur_filter,red_blur_filter))
        red_hsv_blurred=cv2.cvtColor(frame_red_blurred,cv2.COLOR_BGR2HSV)
        mask_red=cv2.inRange(red_hsv_blurred,hand_red_lower,hand_red_upper)
        
        
        cap_frame=frame.copy()
        hand_cap_lower=np.array([30,33,38])
        hand_cap_upper=np.array([56,105,255])
        cap_blur_filter=20
        cap_brightness_filter=44
        cap_brightness_kernel=np.ones(cap_frame.shape,dtype="uint8")*cap_brightness_filter
        cap_frame_brightness=cv2.add(cap_frame,cap_brightness_kernel)
        frame_cap_blurred=cv2.blur(cap_frame_brightness,(cap_blur_filter,cap_blur_filter))
        cap_hsv_blurred=cv2.cvtColor(frame_cap_blurred,cv2.COLOR_BGR2HSV)
        mask_cap=cv2.inRange(cap_hsv_blurred,hand_cap_lower,hand_cap_upper)
        

        return mask_yellow,mask_red,mask_cap




    def region_of_interest(self,mask,vertices):
        mask_temp=np.zeros_like(mask)
        cv2.fillPoly(mask_temp,np.array([vertices],dtype=np.int32),(255,255,255))
        return cv2.bitwise_and(mask,mask_temp)




    def get_contour_details(self,mask,frame):
        all_cnt,_=cv2.findContours(mask,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
        if(len(all_cnt)>0):

            cnt=max(all_cnt,key=cv2.contourArea)
            if(cv2.contourArea(cnt)>500):
                x,y,w,h = cv2.boundingRect(cnt)
                cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        return cv2.bitwise_and(frame,frame,mask),all_cnt



    def get_color_contour_details(self,mask,region_mask,frame,color):
        final_mask=cv2.bitwise_and(mask,mask,mask=region_mask)
        all_cnt,_=cv2.findContours(final_mask,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

        if(len(all_cnt)>0):

            cnt=max(all_cnt,key=cv2.contourArea)
            if(color=="red"):
                if(cv2.contourArea(cnt)>200):
                    x,y,w,h = cv2.boundingRect(cnt)
                    cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
                return cv2.bitwise_and(frame,frame,final_mask),all_cnt

            elif(color=="yellow"):
                if(cv2.contourArea(cnt)>5000):
                    x,y,w,h = cv2.boundingRect(cnt)
                    cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
                return cv2.bitwise_and(frame,frame,final_mask),all_cnt
            
            elif(color=="cap"):
                if(cv2.contourArea(cnt)>5000):
                    x,y,w,h = cv2.boundingRect(cnt)
                    cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
                return cv2.bitwise_and(frame,frame,final_mask),all_cnt
            
        return cv2.bitwise_and(frame,frame,final_mask),all_cnt
    
    
    def click(x,y):
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN,x,y,0,0)
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP,x,y,0,0)


    def hand_process(self,contours,image,color=None):

        if(len(contours)>0):

            cnt=max(contours,key=cv2.contourArea)

            if(color=="left_red"):
                if(cv2.contourArea(cnt)>200):
                    if(self.play==1):
                        print("Shooting")

            elif(color=="left_cap"):
                if(cv2.contourArea(cnt)>200):
                    if(self.play==1):
                        M=cv2.moments(cnt)
                        cx = int(M['m10']/M['m00'])
                        cy = int(M['m01']/M['m00'])
                        cv2.circle(image, (cx,cy), 50, (255,0,0), thickness=-1)
                        win32api.SetCursorPos((cx*6,int(cy*2.5)))
                        print(cx,cy,cx*6,cy*2.5)
#                        win32api.mouse_event(win32con.MOUSEEVENTF_MOVE | win32con.MOUSEEVENTF_ABSOLUTE, int(cx/320*65535.0), int(cy/320*65535.0))
                        

            elif(color=="right_red"):
                if(cv2.contourArea(cnt)>500):
                    print("Releasing all Keys",cv2.contourArea(cnt)*3)
                    self.z_index_area=cv2.contourArea(cnt)*3

            elif(color=="right_yellow"):
                if(cv2.contourArea(cnt)>4000):
                    if(cv2.contourArea(cnt)>self.z_index_area*1.25):
                        print("Move Forward")
                    elif(cv2.contourArea(cnt)<self.z_index_area*0.75):
                        print("Move Backward")
                    else:
                        print("releasing yellow all")
                    print("yellow area:",cv2.contourArea(cnt))

    


    def main_process(self):
        webcam=cv2.VideoCapture(0)
        count=0
        while(True):
            count+=1
            _,frame=webcam.read()
            frame=cv2.flip(frame,1)

            all_masks=self.get_masks(frame)
            mask=all_masks[0]+all_masks[1]+all_masks[2]
            x=frame.shape[1]
            y=frame.shape[0]
            left_vertices=np.array([[0,0],[x/2,0],[x/2,y],[0,y]])
            right_vertices=np.array([[x/2,0],[x,0],[x,y],[x/2,y]])

            left_roi_mask=self.region_of_interest(mask,left_vertices)
            right_roi_mask=self.region_of_interest(mask,right_vertices)
            time.sleep(0.001)
            
            left_red_contours,left_red_contours_details=self.get_color_contour_details(all_masks[1],left_roi_mask,frame.copy(),"red")
            left_yellow_contours,left_yellow_contours_details=self.get_color_contour_details(all_masks[0],left_roi_mask,frame.copy(),"yellow")
            right_red_contours,right_red_contours_details=self.get_color_contour_details(all_masks[1],right_roi_mask,frame.copy(),"red")
            right_yellow_contours,right_yellow_contours_details=self.get_color_contour_details(all_masks[0],right_roi_mask,frame.copy(),"yellow")
            
            cap_contours,cap_contours_details=self.get_color_contour_details(all_masks[2],left_roi_mask,frame.copy(),"cap")
            time.sleep(0.001)
            
            self.hand_process(left_red_contours_details,left_red_contours,"left_red")
            self.hand_process(right_red_contours_details,right_red_contours,"right_red")
            self.hand_process(cap_contours_details,cap_contours,"left_cap")
            if(len(right_red_contours_details)==0):
                self.hand_process(right_yellow_contours_details,right_yellow_contours,"right_yellow")
            time.sleep(0.001)
            output=cv2.bitwise_and(frame,frame,mask=mask)
            cv2.imshow("left roi",left_roi_mask)
            cv2.imshow("right roi",right_roi_mask)
            cv2.imshow("left red",left_red_contours)
            cv2.imshow("left yellow",left_yellow_contours)
            cv2.imshow("right red",right_red_contours)
            cv2.imshow("left_cap",cap_contours)
            cv2.imshow("mask cap",all_masks[2])
            cv2.imshow("og frame",frame)
            cv2.imshow("output",output)

            if(cv2.waitKey(1)==13):
                webcam.release()
                cv2.destroyAllWindows()
                break




def main3():

    pm=ProcessMain()
    pm.main_process()
    time.sleep(0.01)


def main():
    th1 = threading.Thread(target = main3).start()
#    th2 = threading.Thread(target = main2).start()

if __name__=="__main__":
    main()