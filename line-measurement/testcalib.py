import numpy as np 
import cv2 
from multiprocessing import Process
#import csv
import msvcrt as m
from matplotlib import pyplot as plt
from skimage import io
import math  
def process_frame(frame):
    img=frame[140:,:]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

#io.imshow(img[:, :, ::-1])
#plt.show()
    #lower_range = np.array([20,30,190])
    #upper_range = np.array([29,190,240])
    lower_range = np.array([20,10,190])
    upper_range = np.array([37,190,255])
    img = cv2.inRange(img, lower_range, upper_range)
    img = cv2.GaussianBlur(img,(5,5),cv2.BORDER_DEFAULT)
#img[abs(35-img[:,:,0])>=4]=0
#img[img[:,:,0]!=0]=1
#img[abs(img[:,:,0]-36)<5]=1

#io.imshow(img)
#plt.show()
#gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.Canny(img, 60, 60*4)
#io.imshow(edges)
#plt.show()
#lines = cv2.HoughLinesP(edges, 1, np.pi/180, 30, maxLineGap=250)
#lines = cv2.HoughLinesP(edges, 1, np.pi/180, 10, maxLineGap=10)
    #lines = cv2.HoughLinesP(edges, 1, np.pi/180, 8, maxLineGap=50)
    mvariance=0
    xmax=0
    ymax=0
    premean=0
    maxdiffmean=0
    indexmvar=0
    for index in range(20):
    #fragimg.append(img[index*50:index*50+50,:])
        edges=img[index*25:index*25+25,:]
    #io.imshow(edges)
    #plt.show()
    #lines = cv2.HoughLinesP(edges, 1, np.pi/180, 30, maxLineGap=250)
    #lines = cv2.HoughLinesP(edges, 1, np.pi/180, 10, maxLineGap=10)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 15, maxLineGap=20)
        if lines is None:
         #print('no line')
         continue
        

        
        
        anglemat=[]
       
        for indx,line in enumerate(lines):
           x1, y1, x2, y2 = line[0]
           #print('now')
           #print(line)
          
           
          
           #rho = line[0][0]
           #theta = line[0][1]
           #angle= math.degrees(math.atan2((y2-y1),(x2-x1)))
           if (x1!=x2) and (y1!=y2) :
            angle= math.degrees(math.atan2((y2-y1),(x2-x1)))
            if abs(angle)>25:
             
             cv2.line(img, (x1, y1), (x2, y2), (255, 255, 0), 1)
            #print( ' line {} x1 {},y1 {},x2 {},y2 {} angle  {}'.format(indx,x1,y1,x2,y2,angle))
             anglemat.append(angle)
            #if y_min1>y1:
             #y_min1=y1
             #angle1=angle
            #if y_min2>y2:
             #y_min2=y2
             #angle2=angle
            #angle= math.degrees(math.atan((y1-y2)/(x1-x2)))
            
            if (x2>500) and (x2>500) and (25*index+y2>ymax):
               #print('got ymnax')
               ymax=25*index+y2
               xmax=x2
        if (anglemat):     
         if (premean!=0):  
          #if (abs(np.mean(anglemat)-premean)>100):
           #print('diff is over 100 is {} and index is {}'.format(abs(np.mean(anglemat)-premean),index))          
          if abs(np.mean(anglemat)-premean)>maxdiffmean:
           maxdiffmean=np.mean(anglemat)-premean
           
           indexmvar=index
          premean=np.mean(anglemat)
         else:
          premean=np.mean(anglemat) # first time
    
    #print('max diffmean is {} premean  is {} and its index is {}'.format(maxdiffmean,premean,indexmvar))    
    
   #img=cv2.putText(img, "{}".format(dis), (int(x2), int(y2)), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255), 2, cv2.LINE_AA)
    #print('angle mean is {} and var is {}'.format(np.mean(anglemat),np.var(anglemat)))
    #print('gia tri hieu la {}'.format(abs(c1-c2)))
    xm3=math.sqrt((1080-xmax)**2+(ymax)**2)
    a = -0.07280067919587918
    b = 0.0001767467766172589
    c = -0.038716739957248475
    d = 1.1437976876676706e-05
    e = -0.020572336452301287
    f = 11.628985207409308
    xm1=indexmvar
    xm2=maxdiffmean
    #xm3=198.305320150519
    if indexmvar!=0 and maxdiffmean!=0:
     dis=(a*xm1**2+b*xm1)+(b*xm2**2+c*xm2)+(d*xm3**2+e*xm3)+f
    else:
     dis=-1000
    return dis
def calib(cap_receive):
 print('move to position 0 and press key')
 m.getch()
 
 ret,frame = cap_receive.read()
 if not ret:
            print('empty frame 0')
            return 0,0
 #cv2.imshow('frame at zero', frame)
 y_0=float(process_frame(frame))
 
 print('move to position 0,5 and press key')
 m.getch()
 ret,frame = cap_receive.read()
 if not ret:
            print('empty frame 0,5')
            return 0,0
 #cv2.imshow('frame at 0.5', frame)
 y_05=float(process_frame(frame))
 #print('y_05 is {} and real max is {}'.format(y_05,440-y_05))
 slide=float(0.5/(y_05-y_0))
 print('slide is {}'.format(abs(slide)))
 print('move to position 1 and press key')
 m.getch()
 
 ret,frame = cap_receive.read()
 if not ret:
            print('empty frame 1')
            return 0,0
 #cv2.imshow('frame at 1', frame)
 y_1=float(process_frame(frame))
 #rint('y_1 is {} and real max is {}'.format(y_1,440-y_1))
 error=1-(y_1-y_0)*slide
 print('error is {}'.format(error))
 offset=0
 if abs(error)>0.1:
  offset=error
 print('validation (0.5 to 1) is {}'.format(0.5-slide*(y_1-y_05)+offset))
 return slide,offset 
  
def receive():
    
    #f = open('data.csv', 'w',newline='')
    #csv.writer.writerow(["depth", "id"])
    count=1
    
    cap_receive = cv2.VideoCapture('udpsrc port=5000 caps = "application/x-rtp, media=(string)video, clock-rate=(int)90000, encoding-name=(string)H264, payload=(int)96" ! rtph264depay ! decodebin ! videoconvert ! appsink', cv2.CAP_GSTREAMER)

    if not cap_receive.isOpened():
        print('VideoCapture not opened')
        exit(0)       
    #slide,offset=calib(cap_receive)      
    while True:
        ret,frame = cap_receive.read()
       
        if not ret:
            print('empty frame')
            break
        dis=process_frame(frame)
        if (dis!=-1000):
         pre_dis=dis
        else:
         dis=pre_dis
        #dis=
        frame=cv2.putText(frame, "{}".format(int(dis)), (540, 320), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255), 2, cv2.LINE_AA)
        print("{}".format(dis))
        cv2.imshow('receive', frame)
        
       
        if cv2.waitKey(1)&0xFF == ord('q'):
            break            

    #cap_receive.release()

if __name__ == '__main__':
   
    r = Process(target=receive)
   
    r.start()
   
    r.join()

    cv2.destroyAllWindows()