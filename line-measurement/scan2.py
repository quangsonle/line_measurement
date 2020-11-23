import cv2
import os
import pandas as pd
from skimage.transform import rotate
from skimage import feature
from skimage import io
from skimage import filters
from skimage.color import rgb2gray
from matplotlib import pyplot as plt
import numpy as np
import math  
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        print(filename)
        if img is not None:
            images.append(img)
    return images
folder="./curve2//"
images=load_images_from_folder(folder)
#df = pd.DataFrame(columns=['ult', 'ulb', 'dlt','dlb'])

for img in images:
    img=img[140:,:]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

#io.imshow(img[:, :, ::-1])
#plt.show()
    lower_range = np.array([20,30,190])
    upper_range = np.array([29,190,240])
    img = cv2.inRange(img, lower_range, upper_range)
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
    lines = cv2.HoughLinesP(img, 1, np.pi/180, 15, maxLineGap=20)
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
            if abs(angle)>30:
             
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
            
            if (x2>500) and (y2>ymax):
               #print('got ymnax')
               ymax=y2
               xmax=x2
          

          
                  
    print('max variance is {} and its index is {}'.format(np.var(anglemat),np.mean(anglemat)))    
    
   #img=cv2.putText(img, "{}".format(dis), (int(x2), int(y2)), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255), 2, cv2.LINE_AA)
    #print('angle mean is {} and var is {}'.format(np.mean(anglemat),np.var(anglemat)))
    #print('gia tri hieu la {}'.format(abs(c1-c2)))
    dis=math.sqrt((1080-xmax)**2+(ymax)**2)
    
    
    print( ' xmax is {}, ymax is {} ymax after {} and dis is {}  '.format(xmax,ymax,500-ymax,dis))
    
 #print(np.where(np.amax(ii[0])))
# print('where is {} gia tri cot nho nhat {} '.format( np.where(np.amax(ii[0])),ii[1][min(np.where(np.amax(ii[0])))]))
         
#edges=rgb2gray(edges)
 #io.imshow(img)
 #plt.show()
 