import numpy as np 
import cv2 
from multiprocessing import Process
#import csv
import msvcrt as m
from matplotlib import pyplot as plt
from skimage import io
import math  
def process_frame(frame):
    img=frame[140:,600:]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_range = np.array([20,10,169])    # lower_range = np.array([20,10,190]) upper_range = np.array([37,190,255])
    
    upper_range = np.array([37,190,255])
#lower_range = np.array([71,39,71])  
#upper_range = np.array([189,125,190])
    #lower_range = np.array([39,90,130])  
    #upper_range = np.array([100,189,178])
    img = cv2.inRange(img, lower_range, upper_range)
    
    if cv2.countNonZero(img) < 100:
     print('so far')
    img = cv2.GaussianBlur(img,(5,5),cv2.BORDER_DEFAULT)
    #img[abs(35-img[:,:,0])>=4]=0
    #img[img[:,:,0]!=0]=1
    #img[abs(img[:,:,0]-36)<5]=1

    

    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.Canny(img, 60, 60*4)

    ymax=0
    xmax=0
    #premean=0
    #maxdiffmean=0
    indexmvar=0
    enda=0
    neg_detected=0
    neg_index=0
    neg_value=0
    pos_value=0
    for index in range(20):
    #fragimg.append(img[index*50:index*50+50,:])
        edges=img[index*25:index*25+25,:]
    #io.imshow(edges)
    #plt.show()
    #lines = cv2.HoughLinesP(edges, 1, np.pi/180, 30, maxLineGap=250)
    #lines = cv2.HoughLinesP(edges, 1, np.pi/180, 10, maxLineGap=10)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 15, maxLineGap=100)
        if lines is None:
            
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
            if abs(angle)>20 and enda==0:
             
             
             anglemat.append(angle)
            if x1>350 and x2>350 and 25*index+y2>ymax:
             ymax=25*index+y2
             xmax=x2
    
        if not anglemat or enda==1:
         pass
        else:
         #print('index is {} mean is {} variance is {}'.format(index, np.mean(anglemat),np.var(anglemat)))
         if (anglemat): 
          if (np.mean(anglemat))<-25 and neg_detected==0:
           neg_detected=1
           neg_index=index
           neg_value=np.mean(anglemat)
           continue
          if (neg_detected==1):
            if (np.mean(anglemat))>40:
             if (index>neg_index):
               indexmvar=index
               pos_value=np.mean(anglemat)
               enda=1
           
            
           
         
         '''
             if (premean!=0):  
              #if (abs(np.mean(anglemat)-premean)>100):
               #print('diff is over 100 is {} and index is {}'.format(abs(np.mean(anglemat)-premean),index))          
              if abs(np.mean(anglemat)-premean)>maxdiffmean:
               maxdiffmean=abs(np.mean(anglemat)-premean)
               
               indexmvar=index
              premean=np.mean(anglemat)
             else:
              premean=np.mean(anglemat) # first time
              '''
     
    #print('angle1 {}, y1 is {} and angle2 is  {}, y2 is {}'.format(angle1,y_min1,angle2,y_min2))
   
    xm3=math.sqrt((480-xmax)**2+(ymax)**2)
    #print('max diffmean is {} premean  is {} and its index is {}'.format(maxdiffmean,premean,indexmvar))    
    xm2=pos_value-neg_value
   #img=cv2.putText(img, "{}".format(dis), (int(x2), int(y2)), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255), 2, cv2.LINE_AA)
    #print('angle mean is {} and var is {}'.format(np.mean(anglemat),np.var(anglemat)))
    #print('gia tri hieu la {}'.format(abs(c1-c2)))
    #print(indexmvar)
    '''
    if (indexmvar>10):
      cv2.imwrite('10f.jpeg', frame)
         
    if (indexmvar==3):
      cv2.imwrite('4f.jpeg', frame)
    if (indexmvar==6):
      cv2.imwrite('6f.jpeg', frame)
      '''
    a = -0.486677237039261
    b = 100.0
    c = -0.0025037093896842716
    d = -0.5857928636371362
    xm1=indexmvar
    #xm2=maxdiffmean
    
    if xm3 >0.01 and indexmvar>0.01 :
     dis=b*(xm3**a)+c*xm3+xm1*d+1#b*(xm3**a)+c*xm3+d*xm2+e*(xm1**f)+g*xm1+h*(xm2**i)#b*(xm3**a)+c*xm3+d*xm2+h*(xm2**i)#b*(xm3**a)+c*xm3+d*xm2+e*(xm1**f)+g*xm1+h*(xm2**i) #b*(xm3**a)+c*xm3+d*(xm1**e)+f*xm1# b*(xm3**a)+c*xm3+d*xm2+xm1*e
     if(dis<0):
      dis=0
    else:
     dis=-1000
    return dis

  
def receive():
    
    #f = open('data.csv', 'w',newline='')
    #csv.writer.writerow(["depth", "id"])
    count=1
    
    cap_receive = cv2.VideoCapture('udpsrc port=5000 caps = "application/x-rtp, media=(string)video, clock-rate=(int)90000, encoding-name=(string)H264, payload=(int)96" ! rtph264depay ! decodebin ! videoconvert ! appsink', cv2.CAP_GSTREAMER)

    if not cap_receive.isOpened():
        print('VideoCapture not opened')
        exit(0)       
    #slide,offset=calib(cap_receive)    
    pre_dis=0
    dis_arr=[]
    count_dis=0
    mean_dis=0
    
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