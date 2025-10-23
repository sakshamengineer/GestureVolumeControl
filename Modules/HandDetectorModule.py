import mediapipe as mp
import cv2

class HandDetection():

    def __init__(self ,maxHands = 2,TrackingConf = 0.5,DetectionConf = 0.5,model = 1):
        self.mp_hands = mp.solutions.hands
        self.Hands = self.mp_hands.Hands(max_num_hands = maxHands,min_tracking_confidence = TrackingConf,min_detection_confidence = DetectionConf,model_complexity=model)
        self.mp_drawing = mp.solutions.drawing_utils

    def findhands(self,frame,draw = True):
        rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        self.results = self.Hands.process(rgb) 
        if self.results.multi_hand_landmarks:
            for landmarks in self.results.multi_hand_landmarks:
                if draw:
                    self.mp_drawing.draw_landmarks(frame,landmarks,self.mp_hands.HAND_CONNECTIONS)
        return frame
    
    def findposition(self,frame,handno = 0,draw = False):
        lms = []
        if self.results.multi_hand_landmarks:
            myhand = self.results.multi_hand_landmarks[handno]
            for id,lm in enumerate(myhand.landmark):
                h,w = frame.shape[:2]
                px,py = int(lm.x * w),int(lm.y * h)
                lms.append([id,px,py])
                if draw:
                    if id == 4:
                        cv2.circle(frame,(px,py),15,(255,0,255),-1)
        return lms


def main():

    cap = cv2.VideoCapture(0)
    hand = HandDetection()
    while True:
        success,frame = cap.read()
        if not success:
            break
        frame = hand.findhands(frame)
        lmlist = hand.findposition(frame,draw=True)
        if len(lmlist) != 0:
            print(lmlist[4])
        cv2.imshow("webcam",frame)
        if cv2.waitKey(1) & 255 == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()