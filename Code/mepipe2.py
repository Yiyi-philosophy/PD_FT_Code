import cv2
import mediapipe as mp
import math
from os import listdir
import xlrd
import xlwt

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.65,
        min_tracking_confidence=0.65)

#cap = cv2.VideoCapture(0)


#fourcc = cv2.VideoWriter_fourcc(*'XVID')
#out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))

#for k in range(11):
k=0
    #cap = cv2.VideoCapture('D:\\07Lab_CV\\03运动功能检测\帕金森实验\手指拍打视频\\0\l_W.mp4')
cap = cv2.VideoCapture('D:\\07Lab_CV\\03运动功能检测\帕金森3\手指拍打视频2\\'+ str(k)+'\\'+str(k) +'-l.mp4')
##奇数为l，偶数为r

#file_hand = open('D:\\07Lab_CV\\03运动功能检测\帕金森实验\\result_search\\'+ str(i) +'-r.txt', mode='w')

print("task"+str(k)+"\n");

#open xlsx
data = xlrd.open_workbook('D:\\07Lab_CV\\03运动功能检测\\帕金森3\\test.xlsx')
table = data.sheets()[0]  # 选取要读的sheet表单，表单0
nrows = table.nrows  # 获取行数
ncols = table.ncols  # 获取表的列数
#print(ncols)
print(nrows,'*',ncols)


while True:
    ret, frame = cap.read()
    if not ret:
        #file_hand.close()
        break
    i = 1 #0--left;1--right;
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # frame= cv2.flip(frame,1)
    results = hands.process(frame)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    if results.multi_handedness:
        for hand_label in results.multi_handedness:
            # print(hand_label)
            label = hand_label.classification[0].label
            #print(label)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # print(hand_landmarks)
            h, w, c = frame.shape

            #print(i)
            if i==1:
                x4=hand_landmarks.landmark[4].x*w
                y4=hand_landmarks.landmark[4].y*h
                z4=hand_landmarks.landmark[4].z*100
                x8=hand_landmarks.landmark[8].x*w
                y8=hand_landmarks.landmark[8].y*h
                z8=hand_landmarks.landmark[8].z*100
                #print(i,x8,y8,z8)
                x=float('%.2f' %(x4-x8))
                y=float('%.2f' %(y4-y8))
                z=float('%.2f' %(z4-z8))
                dis = math.sqrt(x*x+y*y)
                dis='%.2f' % dis
                #'x:'+str(x)+' y:'+str(y)+' z:'
                #file_hand.write(str(dis)+'\n')
                data.write(2*k+1,i,dis)
                #2*k+1  str(dis)
                #奇数为l，偶数为r
                
            i += 1
            #draw[:, :, :] = 255
            #draw = frame
            #mp_drawing.draw_landmarks(draw, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    #out.write(frame)
    #cv2.imshow('MediaPipe Hands', draw)
    '''
    if cv2.waitKey(1) & 0xFF == 27:
        file_hand.close()
        break
        '''

#file_hand.close()
#data.close()
cap.release()

