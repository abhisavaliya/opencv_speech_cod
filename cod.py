# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 22:45:29 2019

@author: abhis
"""
import numpy as np
import cv2
from directkeys import ReleaseKey, PressKey, W,A,S,D

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

        return mask_yellow,mask_red




    def region_of_interest(self,mask,vertices):
        mask_temp=np.zeros_like(mask)
        cv2.fillPoly(mask_temp,np.array([vertices],dtype=np.int32),(255,255,255))
        return cv2.bitwise_and(mask,mask_temp)




    def get_contour_details(self,mask,frame):
        all_cnt,_=cv2.findContours(mask,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
        cnt=max(all_cnt,key=cv2.contourArea)
        if(cv2.contourArea(cnt)>500):
            x,y,w,h = cv2.boundingRect(cnt)
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        return cv2.bitwise_and(frame,frame,mask),all_cnt



    def get_color_contour_details(self,mask,region_mask,frame,color):
        final_mask=cv2.bitwise_and(mask,mask,mask=region_mask)
        all_cnt,_=cv2.findContours(final_mask,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
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



    def hand_process(self,contours,image,color=None):
        cnt=max(contours,key=cv2.contourArea)

        if(color=="left_red"):
            if(cv2.contourArea(cnt)>200):
                if(self.play==1):
                    print("Shooting")

        elif(color=="left_yellow"):
            if(cv2.contourArea(cnt)>5000):
                if(self.play==1):
                    M=cv2.moments(cnt)
                    cx = int(M['m10']/M['m00'])
                    cy = int(M['m01']/M['m00'])
                    cv2.circle(image, (cx,cy), 50, (255,0,0), thickness=-1)
                    print(cx,cy)
                    print(cx*6,int(cy*2.25))

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
            mask=all_masks[0]+all_masks[1]
            x=frame.shape[1]
            y=frame.shape[0]
            left_vertices=np.array([[0,0],[x/2,0],[x/2,y],[0,y]])
            right_vertices=np.array([[x/2,0],[x,0],[x,y],[x/2,y]])

            left_roi_mask=self.region_of_interest(mask,left_vertices)
            right_roi_mask=self.region_of_interest(mask,right_vertices)

            left_hand_contours,left_hand_contour_details=self.get_contour_details(left_roi_mask,frame)
            right_hand_contours,right_hand_contour_details=self.get_contour_details(right_roi_mask,frame)

            left_red_contours,left_red_contours_details=self.get_color_contour_details(all_masks[1],left_roi_mask,frame.copy(),"red")
            left_yellow_contours,left_yellow_contours_details=self.get_color_contour_details(all_masks[0],left_roi_mask,frame.copy(),"yellow")
            right_red_contours,right_red_contours_details=self.get_color_contour_details(all_masks[1],right_roi_mask,frame.copy(),"red")
            right_yellow_contours,right_yellow_contours_details=self.get_color_contour_details(all_masks[0],right_roi_mask,frame.copy(),"yellow")

            self.hand_process(left_yellow_contours_details,left_yellow_contours,"left_yellow")
            self.hand_process(left_red_contours_details,left_red_contours,"left_red")
            self.hand_process(right_red_contours_details,right_red_contours,"right_red")
            print("lennn:",len(right_red_contours_details))
            if(len(right_red_contours_details)==0):
                self.hand_process(right_yellow_contours_details,right_yellow_contours,"right_yellow")

            output=cv2.bitwise_and(frame,frame,mask=mask)
            cv2.imshow("left roi",left_roi_mask)
            cv2.imshow("right roi",right_roi_mask)
            cv2.imshow("left red",left_red_contours)
            cv2.imshow("left hand contor",left_hand_contours)
            cv2.imshow("left yellow",left_yellow_contours)
            cv2.imshow("right red",right_red_contours)
            cv2.imshow("right yellow",right_yellow_contours)
            cv2.imshow("mask",all_masks[1])
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