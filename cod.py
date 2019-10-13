# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 22:45:29 2019

@author: abhis
"""
import numpy as np
import cv2
from directkeys import ReleaseKey, PressKey, W,A,S,D

class ProcessMain:

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

        return mask_yellow,mask_red


    def region_of_interest(self,mask,vertices):
        mask_temp=np.zeros_like(mask)
        cv2.fillPoly(mask_temp,np.array([vertices],dtype=np.int32),(0,0,255))
        return cv2.bitwise_and(mask,mask_temp)


    def left_hand_process(self,mask,frame):
        pass

    def get_contour_details(self,mask,frame):
        all_cnt,_=cv2.findContours(mask,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

        for cnt in all_cnt:
            if(cv2.contourArea(cnt)>1000):
                M=cv2.moments(cnt)
                cx = int(M['m10']/M['m00'])
                cy = int(M['m01']/M['m00'])

                x,y,w,h = cv2.boundingRect(cnt)
                cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

        return all_cnt

    def main_process(self):

        webcam=cv2.VideoCapture(0)
        play=0
        count=0

        while(True):
            count+=1
            _,frame=webcam.read()
            frame=cv2.flip(frame,1)


            masks=self.get_masks(frame)
            mask=masks[0]+masks[1]
            x=frame.shape[1]
            y=frame.shape[0]
            left_vertices=np.array([[0,0],[x/2,0],[x/2,y],[0,y]])
            right_vertices=np.array([[x/2,0],[x,0],[x,y],[x/2,y]])

#            yellow_contours=self.get_contour_details(masks[0],frame)
#            red_contours=self.get_contour_details(masks[1],frame)


            output=cv2.bitwise_and(frame,frame,mask=mask)
            cv2.imshow("roi",self.region_of_interest(frame,left_vertices))
            cv2.imshow("roi2",self.region_of_interest(frame,right_vertices))
            cv2.imshow("mask",mask)
            cv2.imshow("og frame",frame)
            cv2.imshow("output",output)

            if(cv2.waitKey(1)==13):
                webcam.release()
                cv2.destroyAllWindows()
                break


def main():
    pm=ProcessMain()
    pm.main_process()



if __name__=="__main__":
    main()