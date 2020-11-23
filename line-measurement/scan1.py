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
import pandas as pd 
import csv
def load_images_from_folder(folder):
    images = []
    y_matrix=[]
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        #print(filename)
        parts = filename.split(".")
        y_matrix.append(float(parts[0].replace(',','.')))
        if img is not None:
            images.append(img)
    return images,y_matrix
folder="./curve1//"
images,ym=load_images_from_folder(folder)

#df = pd.DataFrame(columns=['ult', 'ulb', 'dlt','dlb'])
diffangle=[]
indexa=[]
df=pd.read_csv('frame_test.csv')  
#df = pd.DataFrame(columns =['index_of_curve', 'max_angle_difference', 'point_dis','dis'] ) 

xm=[]
for img_index, img in enumerate(images):
    img=img[140:,:]
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
        #else:
         #print('no angle mat consider everything zero')
    diffangle.append(maxdiffmean)
    indexa.append(indexmvar)
    #print('max diffmean is {} premean  is {} and its index is {}'.format(maxdiffmean,premean,indexmvar))    
    
   #img=cv2.putText(img, "{}".format(dis), (int(x2), int(y2)), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255), 2, cv2.LINE_AA)
    #print('angle mean is {} and var is {}'.format(np.mean(anglemat),np.var(anglemat)))
    #print('gia tri hieu la {}'.format(abs(c1-c2)))
    dis=math.sqrt((1080-xmax)**2+(ymax)**2)
    
    xm.append(dis)
    df = df.append(pd.Series([indexmvar,maxdiffmean , dis, ym[img_index]], index=df.columns ), ignore_index=True)
    #print( ' xmax is {}, ymax is {} ymax after {} and dis is {}  '.format(xmax,ymax,500-(ymax),dis))
    
 #print(np.where(np.amax(ii[0])))
# print('where is {} gia tri cot nho nhat {} '.format( np.where(np.amax(ii[0])),ii[1][min(np.where(np.amax(ii[0])))]))
         
#edges=rgb2gray(edges)
 #io.imshow(img)
 #plt.show()
df.to_csv('frame_test.csv', encoding='utf-8', index=False)
print('mean of diffangle is {} and var is {}'.format(np.mean(diffangle),np.var(diffangle))) 
print('mean of index is {} and var is {}'.format(np.mean(indexa),np.var(indexa)))
xm=np.array(xm)
ym=np.array(ym)
print('xm is heerrrrrrrrrrrrr')
print(xm)
print('ym is heerrrrrrrrrrrrr')
print(ym)