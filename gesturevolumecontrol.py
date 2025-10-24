import cv2
import Modules.HandDetectorModule as htm
import math
from pycaw.pycaw import AudioUtilities,IAudioEndpointVolume
from ctypes import cast,POINTER
from comtypes import CLSCTX_ALL
import numpy as np

cap = cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)
hands = htm.HandDetection(DetectionConf=0.8,TrackingConf=0.8,model=0)

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_,CLSCTX_ALL,None)
volume = cast(interface,POINTER(IAudioEndpointVolume))
current_vol = volume.GetMasterVolumeLevel()
volrange = volume.GetVolumeRange()
minvol = volrange[0]
maxvol = volrange[1]
volper = np.interp(current_vol,[minvol,maxvol],[0,100])
volbar = np.interp(current_vol,[minvol,maxvol],[350,110])
while True:
    success,frame  = cap.read()
    if not success:
        break
    frame = cv2.flip(frame,1)
    frame = hands.findhands(frame)
    ltlist = hands.findposition(frame)
    
    if len(ltlist) != 0:

        x1,y1 = ltlist[4][1],ltlist[4][2]
        x2,y2 = ltlist[8][1],ltlist[8][2]
        cx,cy = (x1+x2)//2,(y1+y2)//2

        cv2.circle(frame,(x1,y1),15,(255,0,255),-1)
        cv2.circle(frame,(x2,y2),15,(255,0,255),-1)
        cv2.circle(frame,(cx,cy),15,(255,0,255),-1)
        cv2.line(frame,(x1,y1),(x2,y2),(255,0,255),3)

        length = math.hypot(x2-x1,y2-y1)
        vol = np.interp(length,[30,300],[minvol,maxvol])
        volbar = np.interp(length,[30,300],[350,110])
        volper = np.interp(length,[30,300],[0,100])
        volume.SetMasterVolumeLevel(vol, None)

        if length < 25:
            cv2.circle(frame,(cx,cy),15,(0,255,0),-1)

    cv2.rectangle(frame,(50,110),(80,350),(255,0,0),2)
    cv2.rectangle(frame,(50,int(volbar)),(80,350),(255,0,0),-1)
    cv2.putText(frame,f"{int(volper)} %",(45,390),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),3)
    cv2.imshow("Your Frame",frame)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()