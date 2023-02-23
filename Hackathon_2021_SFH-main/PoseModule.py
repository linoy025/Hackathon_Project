import cv2
import numpy as np
import time
import mediapipe as mp
class poseDetector():
   def __init__(self,mode=False,complexity=1,smooth=True,detectionCon=0.5,trackCon=0.5):
        self.mode=mode
        self.complexity=complexity
        self.smooth=smooth
        self.detectionCon=detectionCon
        self.trackCon=trackCon
        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose=mp.solutions.pose
        self.pose=self.mpPose.Pose(self.mode,self.smooth,self.complexity,self.detectionCon,self.detectionCon)

   def findPose(self,img,draw=True):
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    self.results = self.pose.process(imgRGB)
    if self.results.pose_landmarks:
        if draw:
         self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
    return img





   def findPostion(self, img ,draw=True):
    lmList = []
    if  self.results.pose_landmarks:
        for id,lm in enumerate(self.results.pose_landmarks.landmark):
                h,w,c=img.shape
                # print(id,lm)
                cx,cy=int(lm.x*w),int(lm.y*h)
                lmList.append([id,cx,cy])
                if draw:
                 cv2.circle(img,(cx,cy),5,(255,0,0),cv2.FILLED)

    return lmList

def main():
    cap = cv2.VideoCapture(0)
    detector = poseDetector()

    while True:
        success, img = cap.read()
        img = detector.findPose(img)
        lmList = detector.findPostion(img)
        print(lmList)
        cv2.waitKey(1)
        cv2.imshow("Image", img)


if __name__ =="__main__":
    main()