#............................LiveCamera with gesture recognize the Indian sign.........................#

import os
import cv2
import time
import numpy as np
from keras.models import load_model
import os

import time

from Reverse import Say
import warnings
warnings.filterwarnings(action = 'ignore')
import warnings
warnings.filterwarnings(action = 'ignore')

path = 'Dataset/'
path2 = 'Preprocessed/Train/'
res1=''

gestures = os.listdir(path)
model = load_model('CNN_model.h5')

dict_labels={}
for i in range(len(gestures)):
    dict_labels[gestures[i]]=i
    

def predict(gesture):
    img = cv2.resize(gesture, (50,50))
    img = img.reshape(1,50,50,1)
    img = img/255.0
    prd = model.predict(img)
    index = prd.argmax()
    return gestures[index]
def check():
    global res1
    vc = cv2.VideoCapture(0)
    rval, frame = vc.read()
    old_text = ''
    pred_text = ''
    count_frames = 0
    total_str = ''
    flag = False

    while True:
        
        if frame is not None: 
            
            frame = cv2.flip(frame, 1)
            frame = cv2.resize( frame, (400,400) )
            
            cv2.rectangle(frame, (300,300), (100,100), (0,255,0), 2)
            
            crop_img = frame[100:300, 100:300]
            grey = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
            
            thresh = cv2.threshold(grey,210,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]
        
            blackboard = np.zeros(frame.shape, dtype=np.uint8)
            cv2.putText(blackboard, "Output : ", (30, 40), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 0, 255))
            if count_frames > 20 and pred_text != "":
                total_str += pred_text
                count_frames = 0
                
            if flag == True:
                old_text = pred_text
                pred_text = predict(thresh)
                #Say(pred_text)
                print(pred_text)
                #f = open('C:/Users/91761/OneDrive/Desktop/FinalProject_SignLang(DL)/output.txt','w')
                #f.write(str(pred_text))
                #f.close()
                '''
                if str(pred_text)=='LEFT':
                    res = 'LEFT'
                    print(res)
                elif str(pred_text)=='THANKYOU':
                    res = 'THANKYOU'
                    print(res)
                elif str(pred_text)=='ONE':
                    res = 'ONE'
                    print(res)
                elif str(pred_text)=='PLAY':
                    res = 'PLAY'
                    print(res)
                elif str(pred_text)=='RIGHT':
                    res = 'RIGHT'
                    print(res)
                elif str(pred_text)=='TWO':
                    res = 'TWO'
                    print(res)
                elif str(pred_text)=='STOP':
                    res = 'STOP'
                    print(res)
                elif str(pred_text)=='NONE':
                    res = 'NONE'
                    print(res)
                elif str(pred_text)=='ZERO':
                    res = 'ZERO'
                    print(res)
                elif str(pred_text)=='NO':
                    res = 'NO'
                    print(res)
                '''
                res=str(pred_text)
                print('res',res)
                #res1=res1+'\n'+res
                f = open('D:/FinalProject_SignLang(DL) - Copy/result.txt','w')
                f.write(str(res))
                f.close()
                if old_text == pred_text:
                    count_frames += 1                
                else:
                    count_frames = 0
            #print('*************',pred_text)
                #f = open('H:/code check/FinalProject_SignLang(DL)/result.txt','w')
                #f.write(str(res))
                #f.close()
                if res!='NONE':
                    cv2.putText(blackboard, res, (30, 80), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 255, 0))
            res = np.hstack((frame, blackboard))
            
            cv2.imshow("image", res)
            keypress = cv2.waitKey(1)
            if keypress == ord('s'):
                Say(pred_text)
            #cv2.imshow("hand", thresh)
            #time.sleep(1)
        rval, frame = vc.read()
        keypress = cv2.waitKey(1)
        if keypress == ord('c'):
            flag = True
        if keypress == ord('q'):
            break

    vc.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    vc.release()

