
# Vesrion of 1.1.2021
import cv2
import numpy as np
import picamera
from picamera.array import PiRGBArray
import serial  # serial communication
from time import strftime
import time


from multiprocessing import Process
# import csv


import math
# This function takes the linear lines that we found and make an avrrage of them and puts them back. So instead of few
# different direction lines we get just one. This one line will start at the bottom and will go 0.5 of the image size.


def sendInt(servo_angle):
    send_string = str(servo_angle)
    send_string += ">\n"
    send_string = "<an" + send_string
    ser.write(send_string.encode('utf-8'))
    ser.flush()


def showing_the_video(rotated_image, line_image, cropped_image):
    # combines the two images together
    combo_image = cv2.addWeighted(rotated_image, 0.8, line_image, 1, 1)
    result.write(combo_image)
    cv2.imshow('result', combo_image)

# func PID recieves


def PID(error, mean_prev_errors, iteration_time, integral_prior, error_prior, forward, Kp, Kd, Ki):
    #     integral = integral_prior + error * iteration_time
    integral = error * iteration_time
    # print("INT " + str(integral))
    derivative = (error - error_prior) / iteration_time
    output = int(65 + Kp * mean_prev_errors + Ki * integral + Kd * derivative)
    if output == 0:
        output = 1  # for some reason it doesnt get 0
    error_prior = error
    integral_prior = integral
    return [output, integral_prior, error_prior]


def process_frame(frame):
    img = frame[140:, 600:]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # lower_range = np.array([20,10,190]) upper_range = np.array([37,190,255])
    lower_range = np.array([20, 10, 169])

    upper_range = np.array([37, 190, 255])
# lower_range = np.array([71,39,71])
# upper_range = np.array([189,125,190])
    # lower_range = np.array([39,90,130])
    # upper_range = np.array([100,189,178])
    img = cv2.inRange(img, lower_range, upper_range)

    if cv2.countNonZero(img) < 100:
        return -1000
    img = cv2.GaussianBlur(img, (5, 5), cv2.BORDER_DEFAULT)
    # img[abs(35-img[:,:,0])>=4]=0
    # img[img[:,:,0]!=0]=1
    # img[abs(img[:,:,0]-36)<5]=1

    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.Canny(img, 60, 60*4)

    ymax = 0
    xmax = 0
    # premean=0
    # maxdiffmean=0
    indexmvar = 0
    enda = 0
    neg_detected = 0
    neg_index = 0
    neg_value = 0
    pos_value = 0
    for index in range(20):
        # fragimg.append(img[index*50:index*50+50,:])
        edges = img[index*25:index*25+25, :]
    # io.imshow(edges)
    # plt.show()
    # lines = cv2.HoughLinesP(edges, 1, np.pi/180, 30, maxLineGap=250)
    # lines = cv2.HoughLinesP(edges, 1, np.pi/180, 10, maxLineGap=10)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 15, maxLineGap=100)
        if lines is None:

            continue

        anglemat = []
        for indx, line in enumerate(lines):
            x1, y1, x2, y2 = line[0]
            # print('now')
            # print(line)

            # rho = line[0][0]
            # theta = line[0][1]
            # angle= math.degrees(math.atan2((y2-y1),(x2-x1)))
            if (x1 != x2) and (y1 != y2):
                angle = math.degrees(math.atan2((y2-y1), (x2-x1)))
                if abs(angle) > 20 and enda == 0:

                    anglemat.append(angle)
                if x1 > 350 and x2 > 350 and 25*index+y2 > ymax:
                    ymax = 25*index+y2
                    xmax = x2

        if not anglemat or enda == 1:
            pass
        else:
            # print('index is {} mean is {} variance is {}'.format(index, np.mean(anglemat),np.var(anglemat)))
            if (anglemat):
                if (np.mean(anglemat)) < -25 and neg_detected == 0:
                    neg_detected = 1
                    neg_index = index
                    neg_value = np.mean(anglemat)
                    continue
                if (neg_detected == 1):
                    if (np.mean(anglemat)) > 40:
                        if (index > neg_index):
                            indexmvar = index
                            pos_value = np.mean(anglemat)
                            enda = 1

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

    # print('angle1 {}, y1 is {} and angle2 is  {}, y2 is {}'.format(angle1,y_min1,angle2,y_min2))

    xm3 = math.sqrt((480-xmax)**2+(ymax)**2)
    # print('max diffmean is {} premean  is {} and its index is {}'.format(maxdiffmean,premean,indexmvar))
    xm2 = pos_value-neg_value
   # img=cv2.putText(img, "{}".format(dis), (int(x2), int(y2)), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255), 2, cv2.LINE_AA)
    # print('angle mean is {} and var is {}'.format(np.mean(anglemat),np.var(anglemat)))
    # print('gia tri hieu la {}'.format(abs(c1-c2)))
    # print(indexmvar)
    '''
    if (indexmvar>10):
      cv2.imwrite('10f.jpeg', frame)
         
    if (indexmvar==3):
      cv2.imwrite('4f.jpeg', frame)
    if (indexmvar==6):
      cv2.imwrite('6f.jpeg', frame)
      '''

    xm1 = indexmvar
    # xm2=maxdiffmean
    if (xm2 > 40):
        a = -0.486677237039261
        b = 100.0
        c = -0.0025037093896842716
        d = -0.5857928636371362
        if xm3 > 0.01 and indexmvar > 0.01:

            # b*(xm3**a)+c*xm3+d*xm2+e*(xm1**f)+g*xm1+h*(xm2**i)#b*(xm3**a)+c*xm3+d*xm2+h*(xm2**i)#b*(xm3**a)+c*xm3+d*xm2+e*(xm1**f)+g*xm1+h*(xm2**i) #b*(xm3**a)+c*xm3+d*(xm1**e)+f*xm1# b*(xm3**a)+c*xm3+d*xm2+xm1*e
            dis = b*(xm3**a)+c*xm3+xm1*d
            if (dis > 7.5):
                dis += 2
            elif (dis > 4.5):
                dis += 1
            if (dis < 0):
                dis = 0
        else:
            dis = -1000
    else:
        a = -7.652216550045051e-07
        b = 0.00038441947545969705
        c = 0.00019250838739497532
        d = 1.1102884903827761e-05
        e = 1.1102884903827761e-05
        f = -8.148810861200835e-15
        dis = a*xm3**3+b*xm3**2+c*xm3+d + e*xm3**f
    return dis


## CONNECTION WITH ARDUINO SETUP############################
port0 = "/dev/ttyACM0"
port1 = "/dev/ttyACM1"
baudrate = 115200
ser = serial.Serial(port0, baudrate, timeout=1)
ser.flush()

## CAMERA AND VIDEO SETUP #################################
camera = picamera.PiCamera()
size = (640, 480)
camera.resolution = size
camera.framerate = 30
rawCapture = PiRGBArray(camera)  # defines color scales
time.sleep(1)
# fourcc = cv2.VideoWriter_fourcc('M','J', 'P', 'G') #codec
# result = cv2.VideoWriter('from_picamera.avi', fourcc, 30, size)#, isColor = False) - for B&W videos #saves the video

###########################################################
pre_dis = 0
## MAIN ###################################################
# as long as the camera is ope
for frame in camera.capture_continuous(rawCapture, format='bgr', use_video_port=True):
    start_time = time.time()  # to knows how much time it takes for each iteration
    image = frame.array
    dis = process_frame(image)
    if (dis != -1000):
        pre_dis = dis
    else:
        dis = pre_dis
    print('dis is {}'.format(dis))
camera.close()
result.release()
cv2.destroyAllWindows()
# recieves the angle named 'quit' and sends the intager 'send_string' to arduino
sendInt('quit')
ser.close()
