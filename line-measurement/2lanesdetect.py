import cv2
import numpy as np
import sys
import numpy
np.set_printoptions(threshold=sys.maxsize)
import math  
#img = cv2.imread("./temp_test/curve.jpeg")
#img = cv2.imread("./temp_test/straight.jpeg")
#img = cv2.imread("./curve2/1,7.jpeg")
#img = cv2.imread("./curve3/6.jpeg")
img = cv2.imread("./train/15.jpeg")
from matplotlib import pyplot as plt
from skimage import io
#img[:, :, ::-1]
img=img[140:,:]
img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

#io.imshow(img[:, :, ::-1])
#plt.show()
lower_range = np.array([20,30,190])
upper_range = np.array([29,190,240])
io.imshow(img)
plt.show()
img = cv2.inRange(img, lower_range, upper_range)
#img[abs(35-img[:,:,0])>=4]=0
#img[img[:,:,0]!=0]=1
#img[abs(img[:,:,0]-36)<5]=1

io.imshow(img)
plt.show()
#gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(img, 60, 60*4)
#io.imshow(edges)
#plt.show()
#lines = cv2.HoughLinesP(edges, 1, np.pi/180, 30, maxLineGap=250)
#lines = cv2.HoughLinesP(edges, 1, np.pi/180, 10, maxLineGap=10)
lines = cv2.HoughLinesP(edges, 1, np.pi/180, 15, maxLineGap=20)
xmax=0
ymax=0

curve=0
c1,c2=(0,0)
y_neg_min=500
y_min1=500
y_min2=500
angle_max=0
anglemat=[]

for indx,line in enumerate(lines):
   
   x1, y1, x2, y2 = line[0]
   #print('now')
   #print(line)
  
   
  
   #rho = line[0][0]
   #theta = line[0][1]
   angle= math.degrees(math.atan2((y2-y1),(x2-x1)))
   if (x1!=x2) and (y1!=y2) and abs(angle)>30:
    cv2.line(img, (x1, y1), (x2, y2), (255, 255, 0), 1)
    print( ' line {} x1 {},y1 {},x2 {},y2 {} angle  {}'.format(indx,x1,y1,x2,y2,angle))
    
     
    anglemat.append(angle)
    if y_min1>y1:
     y_min1=y1
     angle1=angle
    if y_min2>y2:
     y_min2=y2
     angle2=angle
    #angle= math.degrees(math.atan((y1-y2)/(x1-x2)))
  
    if (x2>500) and (y2>ymax):
     
     ymax=y2
     xmax=x2
   
    
   #img=cv2.putText(img, "{}".format(dis), (int(x2), int(y2)), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255), 2, cv2.LINE_AA)

#print('angle1 {}, y1 is {} and angle2 is  {}, y2 is {}'.format(angle1,y_min1,angle2,y_min2))
#print('gia tri hieu la {}'.format(abs(c1-c2)))

cor=math.sqrt((1080-xmax)**2+(ymax)**2)
print('m variance is {} and mean is {}'.format(np.var(anglemat),np.mean(anglemat)))    
print(-(2/10**6)*cor**3+0.0022*cor**2-0.83028*cor+109.7)
print( ' xmax is {}, ymax is {} ymax after {} and dis is {}  '.format(xmax,ymax,440-ymax,cor))
io.imshow(img)
plt.show()
cv2.imshow("linesEdges", edges)
cv2.imshow("linesDetected", img)
#cv2.waitKey(-1)
cv2.destroyAllWindows()