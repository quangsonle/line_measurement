import cv2
import numpy as np
import sys
import numpy
np.set_printoptions(threshold=sys.maxsize)
import math  
#img = cv2.imread("./temp_test/curve.jpeg")
#img = cv2.imread("./temp_test/straight.jpeg")
#img = cv2.imread("./curve1/6.jpeg")
#img = cv2.imread("./curve1/6.jpeg")
img = cv2.imread("./train/7.jpeg")
from matplotlib import pyplot as plt
from skimage import io
#img[:, :, ::-1]
img=img[140:,:]
img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

io.imshow(img)
plt.show()
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

ymax=0
xmax=0
for index in range(20):
    #fragimg.append(img[index*50:index*50+50,:])
    edges=img[index*25:index*25+25,:]
#io.imshow(edges)
#plt.show()
#lines = cv2.HoughLinesP(edges, 1, np.pi/180, 30, maxLineGap=250)
#lines = cv2.HoughLinesP(edges, 1, np.pi/180, 10, maxLineGap=10)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 15, maxLineGap=200)
    if lines is None:
         print('no line')
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
         
         #cv2.line(img, (x1, y1), (x2, y2), (255, 255, 0), 1)
        #print( ' line {} x1 {},y1 {},x2 {},y2 {} angle  {}'.format(indx,x1,y1,x2,y2,angle))
         anglemat.append(angle)
        if x1>500 and x2>500 and 25*index+y2>ymax:
         ymax=25*index+y2
         xmax=x2
    
    if not anglemat:
     print('none angle mat at index {}'.format(index))
    else:
     print('index is {} mean is {} variance is {}'.format(index, np.mean(anglemat),np.var(anglemat)))
     
    #print('angle1 {}, y1 is {} and angle2 is  {}, y2 is {}'.format(angle1,y_min1,angle2,y_min2))
dis=math.sqrt((1080-xmax)**2+(ymax)**2)
print( ' xmax is {}, ymax is {} ymax after {} and dis is {}  '.format(xmax,ymax,500-(ymax),dis))
io.imshow(img)
plt.show()
    #cv2.imshow("linesEdges", edges)
    #cv2.imshow("linesDetected", img)
    #cv2.waitKey(0)
cv2.destroyAllWindows()