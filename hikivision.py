import cv2
#import cv2
#import urllib.request
import numpy as np
import schedule
import time
from twilio.rest import Client
def person():
    print("person is standing in front of the door!!1pleASE do check!")

def condition():
    if ((classid-1) == 0):
        print(classnames[classid-1])
        client = Client("ACa357037cdd7c244ff248c4c5da8b1710", "73417fb2d06355908900060b35521829 ")
        client.messages.create(to=["+916380325845"],
                               #from_="+12185262623", body="A PERSON IS STANDING IN FRONT OF THE DOOR")
        print("message is send to the phone number:638032****. check your phone")
    else:
        print("SOME OTHER OBJECT IS IDENTIFIED INSTEAD OF A PERSON")
video=cv2.VideoCapture('rtsp://ds_lab:dslab123@10.10.110.254:554/Streaming/Channels/101')
f=open('coco.names','rt')
classnames = []
classnames = f.read().rstrip('\n').split('\n')
configpath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightspath = 'frozen_inference_graph.pb'
net = cv2.dnn_DetectionModel(configpath, weightspath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)
print("****************************HOME SERVILENCE MESSAGING SYSTEM****************************")
print('''This program will detect a person in the cctv camera when an person arive in a camera it will generate an message to the owner of the cctv camera.''')
print("program started")
print("when program started an another tab will open in which the cctv footage will display")
while  True:
    _,frame=video.read()
    #cv2.imshow("RTSP",frame)
    #img_np = np.array(bytearray(frame.read()), dtype=np.uint8)
    #img= cv2.imdecode(img_np, -1)
    classids, confs, bbox = net.detect(frame) #confThreshold=0.5)
    if (len(classids) != 0):
        for classid, confidence, box in zip(classids.flatten(), confs.flatten(), bbox):
            cv2.rectangle(frame,box,color=(0,255,0),thickness=2)
            cv2.putText(frame,classnames[classid-1].upper(),(box[0]+10,box[1]+30),
                    cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
            cv2.putText(frame,str(round( confidence*100,2)), (box[0] + 200, box[1] + 30),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            
            
            #if(classid==0):
                #person()
            #else:
                #print("not a person")
            #cv2.rectangle(img,box,color=(0,255,0),thickness=2)
            #break
        cv2.imshow("output", frame)
        k=cv2.waitKey(10)
        schedule.every(1).seconds.do(condition)
        schedule.run_pending()
        time.sleep(1)
        if(k==ord('q')):
            break
        
    
    
    
    
	
