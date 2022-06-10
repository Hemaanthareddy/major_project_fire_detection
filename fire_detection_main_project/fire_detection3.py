import cv2
import threading
import playsound
import smtplib
from twilio.rest import Client
import numpy as np
alarm_initiate_bool=alarm_end_bool=mail_bool=mail_initiated_bool=sms_bool=True


fire_detection = cv2.CascadeClassifier('fire_detection.xml')
input_type=input("Enter your input type:")
if input_type=='pre recorded video':
    vid=cv2.VideoCapture("pre_recorded_non_fire.mp4")
    Valid=True
elif input_type=='live video':
    vid = cv2.VideoCapture(0)
    Valid=True
else:
    print("Please provide valid input!!!")
    Valid=False

runOnce = False

def play_alarm_sound_function():
    playsound.playsound('fire_alarm.mp3',True)

def send_mail_function(): 
    
    recipientmail_address = "hemareddy2916@gmail.com"
    recipientmail_address = recipientmail_address.lower() 
    
    try:
        server_type = smtplib.SMTP('smtp.gmail.com', 587)
        server_type.ehlo()
        server_type.starttls()
        server_type.login("hemareddy2916@gmail.com", 'hema2916')
        SUBJECT="Alert!!!   Warning Fire Accident"+" "+input_type 
        message = 'Subject: {}'.format(SUBJECT)
        server_type.sendmail('hemareddy2916@gmail.com', recipientmail_address, message) 
        print("Alert mail sent sucesfully to {}".format(recipientmail_address)) 
        server_type.close()
        
    except Exception as e:
        print(e) 

def send_sms_function():
    client = Client("AC1cbbd846d5c8e6ad39d6f6d44c1e8720", "a436c0551bc0c6e2649179586c9b2aa7")
    client.messages.create(to="+917997590921", 
                       from_="+14695298453", 
                       body="Alert!!!   Warning Fire Accident "+"-"+input_type)
if Valid:
    while(1):
        return_status, return_frame = vid.read()
        return_frame = cv2.resize(return_frame, (800, 640))
        
        blur_frame = cv2.GaussianBlur(return_frame, (21, 21), 0)
        hsv_frame = cv2.cvtColor(blur_frame, cv2.COLOR_BGR2HSV)

        lower_array = [18, 50, 50]
        upper_array = [35, 255, 255]
        lower_array = np.array(lower_array, dtype="uint8")
        upper_array = np.array(upper_array, dtype="uint8")

        mask = cv2.inRange(hsv_frame, lower_array, upper_array)

        output = cv2.bitwise_and(return_frame, hsv_frame, mask=mask)


        total_count = cv2.countNonZero(mask)

        if int(total_count) > 15000:
            sobel = cv2.Sobel(return_frame,cv2.CV_64F,1,0,ksize=5)

            g = cv2.cvtColor(sobel, cv2.COLOR_BGR2GRAY)
            g_r = g.reshape(g.shape[0]*g.shape[1])
            for i in range(g_r.shape[0]):
                if g_r[i] > g_r.mean():
                    g_r[i] = 1
                else:
                    g_r[i] = 0
            g= g_r.reshape(g.shape[0],g.shape[1])

            fire_frame = fire_detection.detectMultiScale(g, 1.2, 5)

        Alarm_Status = False
        
        for (x,y,w,h) in fire_frame:
            cv2.rectangle(return_frame,(x-20,y-20),(x+w+20,y+h+20),(255,0,0),2)
            roi_g = g[y:y+h, x:x+w]
            roi_color = return_frame[y:y+h, x:x+w]
            if alarm_initiate_bool:
                print("Fire alarm initiated")
                alarm_initiate_bool=False
            threading.Thread(target=play_alarm_sound_function).start()

            if runOnce == False:
                if mail_initiated_bool:
                    print("Mail send initiated")
                    mail_initiated_bool=False
                threading.Thread(target=send_mail_function).start()
                threading.Thread(target=send_sms_function).start()
                runOnce = True
                
            if runOnce == True:
                if mail_bool:
                    print("Mail is already sent once")
                    mail_bool=False

                if sms_bool:
                    print("SMS is already sent once")
                    sms_bool=False
                runOnce = True

        cv2.imshow('frame', return_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
    vid.release()