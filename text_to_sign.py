#....................Display 'THANKYOU' Speech to text to Sign to speech..........................................#

import os
import cv2
import time
import pyttsx3
import numpy as np

def Say(SayThis):
        t2sengine = pyttsx3.init()
        t2sengine.setProperty('rate', 130)
        t2sengine.say(SayThis)
        t2sengine.runAndWait()

def main(letter):
    l1=letter.split(', ')
    print('l1=',l1)
    for i in l1:
        path="Dataset"
        folder_path = path+'/'+i.upper()
        folder_path=folder_path.strip()
        print("=======",folder_path)
        #folder_path = path+'\\'+letter
        

        for(direcpath,direcnames,files) in os.walk(folder_path):
            
            #for files in range(str(folder_path)):
            print(">>>>")
            #files = os.walk(folder_path)
            print("*******",files)
            frame =folder_path + '/' + files[0]
            print("---frame:",frame)

            frame = cv2.imread(frame)
            frame=cv2.resize(frame,(256,256))
            save_img = np.array(frame)
            cv2.imwrite("outputimage.jpg", save_img)
            cv2.imshow(letter,frame)
            Say(letter)
            #cv2.waitKey(10)
            #cv2.destroyAllWindows()



    
