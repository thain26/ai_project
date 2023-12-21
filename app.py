import cv2
import tkinter as tk
import numpy as np
from tkinter import *
from PIL import ImageTk, Image
from sklearn.tree import export_text

from keras.models import model_from_json
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
from tkinter import filedialog

window = tk.Tk()
window.geometry('650x370+570+170')

#stop resize
window.resizable(width=False, height=False)

label_1 = tk.Label(window, text="PROJECT CUỐI KÌ MÔN TRÍ TUỆ NHÂN TẠO")
label_1.configure(font=("Times New Roman", 20, "bold")) 
label_1.pack()

label_4 = tk.Label(window, text="ĐỀ TÀI: NHẬN DIỆN CẢM XÚC BẢN THÂN")
label_4.configure(font=("Times New Roman", 20, "bold"))
label_4.pack()

label_2 = tk.Label(window, text="NHÓM 15")
label_2.configure(font=("Times New Roman", 20, "bold"))
label_2.pack()



#SPKT logo
img = Image.open("D:/learn/ttnt/nhom15ttnt/logo/daihocthuyloi.png")
logo_rz=img.resize((145,140))
logo = ImageTk.PhotoImage(logo_rz)
label_4 = tk.Label(image=logo) 
label_4.place(relx=0.41, rely=0.6, anchor= NW)

#Load models
model = model_from_json(open("D:/learn/ttnt/nhom15ttnt/model_v2/model_arch_v2.json", "r").read())
model.load_weights('D:/learn/ttnt/nhom15ttnt/model_v2/my_model_v2.h5')
face_haar_cascade = cv2.CascadeClassifier('D:/learn/ttnt/nhom15ttnt/cascade/haarcascade_frontalface_default.xml')

#Import file and recognition
def UploadAction(event=None):
    filename = filedialog.askopenfilename()
    print('Selected:', filename)
    cap = cv2.VideoCapture(filename)
    while cap.isOpened():
        res, frame = cap.read()
        #frame = cv2.resize(frame,(480,480))
        height, width, channel = frame.shape
        gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_haar_cascade.detectMultiScale(gray_image)
        try: 
            for(x, y, w, h) in faces:
                cv2.rectangle(frame, pt1=(x, y), pt2=(x + w, y + h), color=(0, 255, 0), thickness=2)
                roi_gray = gray_image[y:y + h, x:x + w]
                roi_gray = cv2.resize(roi_gray, (48, 48))
                image_pixels = img_to_array(roi_gray)
                image_pixels = np.expand_dims(image_pixels, axis=0)
                image_pixels /= 255
                predictions = model.predict(image_pixels)
                max_index = np.argmax(predictions[0])
                emotion_detection = ('BinhThuong', 'Buon', 'GianDu', 'NgacNhien', 'VuiVe')
                emotion_prediction = emotion_detection[max_index]
                cv2.putText(frame, emotion_prediction, (x+5, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7 , (0, 255, 0), 1, cv2.LINE_AA)
        except:
            pass
        frame[0:int(height/1000), 0:int(width)] = res
        cv2.imshow('', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows() 

    

#Open camera & detect
def detect():
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        res, frame = cap.read()
        height, width, channel = frame.shape
        gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_haar_cascade.detectMultiScale(gray_image)
        try: 
            for(x, y, w, h) in faces:
                cv2.rectangle(frame, pt1=(x, y), pt2=(x + w, y + h), color=(0, 255, 0), thickness=2)
                roi_gray = gray_image[y:y + h, x:x + w]
                roi_gray = cv2.resize(roi_gray, (48, 48))
                image_pixels = img_to_array(roi_gray)
                image_pixels = np.expand_dims(image_pixels, axis=0)
                image_pixels /= 255
                predictions = model.predict(image_pixels)
                max_index = np.argmax(predictions[0])
                emotion_detection = ('BinhThuong', 'Buon', 'GianDu', 'NgacNhien', 'VuiVe')
                emotion_prediction = emotion_detection[max_index]
                cv2.putText(frame, emotion_prediction, (x+5, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        except:
            pass
        frame[0:int(height/20), 0:int(width)] = res
        cv2.imshow('', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows() 

#Button import file and recog
but1=Button(window,padx=5,pady=5,width=30,bg='white',fg='black',relief=GROOVE,text='Import File & Recognition',command=UploadAction,font=('helvetica 15 bold'))
but1.place(relx=0.5,rely=0.44, anchor= CENTER)

#Button only detect
but3=Button(window,padx=5,pady=5,width=30,bg='white',fg='black',relief=GROOVE,command=detect,text='Open Camera & Recognition',font=('helvetica 15 bold'))
but3.place(relx=0.5,rely=0.56, anchor= CENTER)

window.mainloop()
