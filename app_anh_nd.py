import urllib.request as rq
import sqlite3
import os
import numpy as np
import pickle
from tkinter import *
from tkinter.filedialog import askopenfilename
from PIL import ImageTk, Image
from cv2 import *

root = Tk()
root.geometry("1400x700")
cam = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640,480))
img1=ImageTk.PhotoImage(Image.open('nen.jpg'))
#img1 = cv2.imread('nen.jpg')
panel = Label(root, image = img1)
panel.image = img1
panel.place(x = 0, y = 0)

cascPath = "haarcascade_frontalface_default.xml"
detector  = cv2.CascadeClassifier(cascPath)
fontface = cv2.FONT_HERSHEY_SIMPLEX
fontscale = 1
fontcolor = (250,0,0)

recognizer = cv2.face.LBPHFaceRecognizer_create()
path='anh_data_hinh'

def getImagesAndLabels(path):
    imagePaths=[os.path.join(path,f) for f in os.listdir(path)] 
    faces=[]
    IDs=[]
    for imagePath in imagePaths:
        faceImg=Image.open(imagePath).convert('L')
        faceNp=np.array(faceImg,'uint8')
        ID=int(os.path.split(imagePath)[-1].split('.')[1])
        faces.append(faceNp)
            #print(ID)
        IDs.append(ID)
    return IDs, faces

def getProfile(id):
    conn=sqlite3.connect("FaceDB.db")
    cmd="SELECT * FROM Facedata WHERE ID="+str(id)
    cursor=conn.execute(cmd)
    profile=None
    for row in cursor:
        profile=row
    conn.close()
    return profile
#thực hiện chức năng nhận dạng ảnh online
def btn_lay_anh():
    link = lbl_NID.get()
    resource = rq.urlopen(link)
    output = open("file01.jpg","wb")
    output.write(resource.read())
    output.close() 
    img1 = ImageTk.PhotoImage(Image.open('file01.jpg')) 
    panel = Label(root, image = img1)
    panel.image = img1
    panel.place(x = 200, y = 70)
    img = cv2.imread('file01.jpg')
    fontface = cv2.FONT_HERSHEY_SIMPLEX
    fontscale = 1
    fontcolor = (120,0,0)
    Ids,faces=getImagesAndLabels(path)

    recognizer.train(faces,np.array(Ids))
    recognizer.save('reco_anh/trainningData.yml')
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=detector.detectMultiScale(gray,1.3,5)

    for(x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        id,conf=recognizer.predict(gray[y:y+h,x:x+w])
        profile=getProfile(id)
        if(profile!=None):
            cv2.putText(img, "Name: " + str(profile[1]), (x-30,y-20), fontface, fontscale, fontcolor ,2)
            thong_tin=str(profile[1])  
      
    a = "so guong mat nguoi trong anh :" + str(len(faces))
    b = "Ten : " + profile[1]
    lbl3.configure(text = a)
    lbl4.configure(text=b)        
    cv2.imshow('Face',img)
# hàm nhập hình ảnh và nhãn vào csdl và thư mục ảnh  
def btn_nhaphinh():
    tk = Tk()
    tk.geometry("500x200")
    def btn_ghi():
        def insertOrUpdate(Id,Name):
            conn=sqlite3.connect("data_sql/FaceDB.db")
            cmd="SELECT * FROM Facedata WHERE Name="+str(Id)
            cursor=conn.execute(cmd)
            isRecordExist=0
            for row in cursor:
                isRecordExist=1
            if(isRecordExist==1):
                cmd="UPDATE Facedata SET Name="+str(Name)+"WHERE ID="+str(Id)
            else:
                cmd="INSERT INTO Facedata(Id,Name) Values("+str(Id)+","+str(Name)+")"
            conn.execute(cmd)
            conn.commit()
            conn.close()
        id= lbl_NID.get()
        name= lbl_T.get()
        insertOrUpdate(id,name)
        sampleNum=0
        while(True):
            img2 = askopenfilename()
            img = cv2.imread(img2)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, 1.3, 5)
            for (x,y,w,h) in faces:
                cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2) 
                sampleNum=sampleNum+1
                cv2.imwrite("anh_data_hinh/User."+ id +'.'+ str(sampleNum) + ".jpg", gray[y:y+h,x:x+w])
            if sampleNum>50:
                break
    lbl_ID = Label(tk, text="ID", fg="green", font=("Times New Roman", 16), width= 10)
    lbl_ID.place(x=5, y = 15)
    lbl_NID = Entry(tk, width=35, font=("Times New Roman", 14))
    lbl_NID.place(x=100, y = 15)

    lbl_Ten = Label(tk, text="Tên", fg="green", font=("Times New Roman", 16), width= 10)
    lbl_Ten.place(x=5, y = 50)
    lbl_T = Entry(tk, width=35, font=("Times New Roman", 14))
    lbl_T.place(x=100, y=50)

    btn = Button(tk, text="Ghi", font=("Times New Roman", 14), fg="white", bg="green",
              width=12, height=1, command=btn_ghi)
    btn.place(x=170, y=120)
# hàm nhận dạng ảnh từ file ảnh trong thư mục của máy
def btn_nhandang_anh():
    global img
    img = askopenfilename()
    img1 = ImageTk.PhotoImage(Image.open(img))
    panel = Label(root, image = img1)
    panel.image = img1
    panel.place(x = 200, y = 70)
    #đọc bức ảnh với cv2
    img = cv2.imread(img)
    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    fontface = cv2.FONT_HERSHEY_SIMPLEX
    fontscale = 1
    fontcolor = (120,0,0)
    Ids,faces=getImagesAndLabels(path)
    recognizer.train(faces,np.array(Ids))
    recognizer.save('reco_anh/trainningData.yml')

    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=detector.detectMultiScale(gray,1.3,5)
    global thong_tin
    for(x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        id,conf=recognizer.predict(gray[y:y+h,x:x+w])
        profile=getProfile(id)
        if(profile!=None):
            cv2.putText(img, "Name: " + str(profile[1]), (x-30,y-20), fontface, fontscale, fontcolor ,2)
            #thong_tin=profile[1]
        
    thong_tin =str(profile[1])
    a = "so guong mat nguoi trong anh :" + str(len(faces))
    b = "Ten : " + thong_tin
    lbl3.configure(text = a)
    lbl4.configure(text=b)        
    cv2.imshow('Face',img)
#thực hiện chức năng thêm ảnh qua video   
def btn_ghidanh():
    tk = Tk()
    tk.geometry("500x150")
    def btn_ghi():
        def insertOrUpdate(Id,Name):
            conn=sqlite3.connect("data_sql/FaceDB.db")
            cmd="SELECT * FROM Facedata WHERE Name="+str(Id)
            cursor=conn.execute(cmd)
            isRecordExist=0
            for row in cursor:
                isRecordExist=1
            if(isRecordExist==1):
                cmd="UPDATE Facedata SET Name="+str(Name)+"WHERE ID="+str(Id)
            else:
                cmd="INSERT INTO Facedata(ID,Name) Values("+str(Id)+","+str(Name)+")"
            conn.execute(cmd)
            conn.commit()
            conn.close()
        id= lbl_NID.get()
        name= lbl_T.get()#khi nhập để trong cặp dấu nháy 
        insertOrUpdate(id,name)
        sampleNum=0
        while(True):
            ret, img = cam.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, 1.3, 5)
            for (x,y,w,h) in faces:
                cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
                sampleNum=sampleNum+1
                cv2.imwrite("anh_data_hinh/User."+id +'.'+ str(sampleNum) + ".jpg", gray[y:y+h,x:x+w])
                cv2.imshow('frame',img)
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break
            elif sampleNum>50:
                break
        cam.release()
        cv2.destroyAllWindows()

    lbl_ID = Label(tk, text="ID", fg="green", font=("Times New Roman", 16), width= 10)
    lbl_ID.place(x= 15, y= 10)
    lbl_NID = Entry(tk, width=35, font=("Times New Roman", 14))
    lbl_NID.place(x= 150, y= 10)

    lbl_Ten = Label(tk, text="Tên", fg="green", font=("Times New Roman", 16), width= 10)
    lbl_Ten.place(x= 15, y= 45)
    lbl_T = Entry(tk, width=35, font=("Times New Roman", 14))
    lbl_T.place(x= 150, y= 45)
 
    btn = Button(tk, text="Nhập thông tin", font=("Times New Roman", 14), fg="white", bg="green",
              width=15, height=1, command=btn_ghi)
    btn.place(x= 200, y= 90)
#thực hiện chức năng nhận dạng qua camera
def btn_nhandang():
    fontface = cv2.FONT_HERSHEY_SIMPLEX
    fontscale = 1
    fontcolor = (120,0,0)
    
    Ids,faces=getImagesAndLabels(path)
    #trainning
    recognizer.train(faces,np.array(Ids))
    recognizer.save('nhan_dang/trainningData.yml')
   
    while(True):
        ret,img=cam.read()
        gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        faces=detector.detectMultiScale(gray,1.3,5)
        global thong_tin
        for(x,y,w,h) in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            id,conf=recognizer.predict(gray[y:y+h,x:x+w])
            profile=getProfile(id)
            if(profile!=None):
                cv2.putText(img, str(profile[1]), (x,y-20), fontface, fontscale, fontcolor ,2)
                cv2.imshow('Face',img)
                thong_tin= str(len(faces))
        b = "so guong mat nguoi trong anh :" +thong_tin
        lbl3.configure(text = b)  
        #if cv2.waitKey(1)==ord('q'):
        #    break
        #Thực hiện việc lưu video nhận dạng
        if ret==True:
            frame = cv2.flip(img,180)
            out.write(frame)
            #cv2.imshow('frame',frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
   
    cam.release()
    out.release()
    cv2.destroyAllWindows()
#Chức năng đếm số lượng người trong bức ảnh
def btn_demsoluong():
    sampleNum=0
    img2 = askopenfilename()
    img1 = ImageTk.PhotoImage(Image.open(img2))  
    panel = Label(root, image = img1)
    panel.image = img1
    panel.place(x = 200, y = 70)
    img = cv2.imread(img2)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2) 
        
    a = "so guong mat co trong buc anh :" + str(len(faces))
    lbl3.configure(text = a) 

def btn_nhandang_video():
    #fontface, fontscale, fontcolor là kiểu chữ, kích cỡ, màu sắc của chữ lúc hiển thị nhãn nhận dạng
    fontface = cv2.FONT_HERSHEY_SIMPLEX
    fontscale = 1
    fontcolor = (250,0,0)

    Ids,faces=getImagesAndLabels(path)
    #trainning
    recognizer.train(faces,np.array(Ids))
    recognizer.save('nhan_dang/trainningData.yml')
        
    vide =askopenfilename()
    video=cv2.VideoCapture(vide)
    while(True):
        ret,img=video.read()
        gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        faces=detector.detectMultiScale(gray,1.3,5)
        for(x,y,w,h) in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            id,conf=recognizer.predict(gray[y:y+h,x:x+w])
            profile=getProfile(id)
            if(profile!=None):
                cv2.putText(img, str(profile[1]), (x,y-20), fontface, fontscale, fontcolor ,2)
                cv2.imshow('Face',img)
    video.release()
    cv2.destroyAllWindows()
def btn_dong():
    root.destroy()


def btn_huongdan():
    tk = Tk()
    tk.title("Hướng dẫn sử dụng")
    tk.geometry("800x600")
    a = open("huongdan.txt",'r')
    a= a.read()
    lblt = Label(tk, text=a, font=("Times New Roman", 15), fg='blue')
    lblt.place(x=15, y=15)

lbl_NID = Entry(root, width=80, font=("Times New Roman", 14))
lbl_NID.place(x=180, y=15)
btn2 = Button(root, text="Nhận dạng online", font=("Times New Roman", 14), fg="white", bg="green",
              width=15, command=btn_lay_anh)              
btn2.place(x=15, y=12)
btn_hinh = Button(root, text="Thêm người mới", font=("Times New Roman", 14), fg="white", bg="green",
              width=15, height=1, command=btn_nhaphinh)
btn_hinh.place(x=15, y=65)
btn = Button(root, text="Nhận dạng qua ảnh", font=("Times New Roman", 14), fg="white", bg="green",
              width=15, height=1, command=btn_nhandang_anh)
btn.place(x=15, y=115)
btn_ghiten = Button(root, text="Nhập ảnh camera", font=("Times New Roman", 14), fg="white", bg="green",
              width=15, height=1, command=btn_ghidanh)
btn_ghiten.place(x=15, y=165)
btn_recog= Button(root, text="Nhận dạng camera", font=("Times New Roman", 14), fg="white", bg="green",
              width=15, height=1, command=btn_nhandang)
btn_recog.place(x=15, y=215)
btn_demsoluong= Button(root, text="Đếm số lượng", font=("Times New Roman", 14), fg="white", bg="green",
              width=15, height=1, command=btn_demsoluong)
btn_demsoluong.place(x=15, y=265)
btn_nhandang_video= Button(root, text="Nhận dạng video", font=("Times New Roman", 14), fg="white", bg="green",
              width=15, height=1, command=btn_nhandang_video)
btn_nhandang_video.place(x=15, y=315)
btn1 = Button(root, text="Đóng", font=("Times New Roman", 14), fg="white", bg="red",
              width=15, height=1, command=btn_dong)
btn1.place(x=1200, y=650)
btn2 = Button(root, text="Hướng dẫn", font=("Times New Roman", 14), fg="white", bg="red",
              width=15, height=1, command=btn_huongdan)
btn2.place(x=15, y=650)
lbl3 = Label(root, text= " ", font=("Times New Roman", 14), fg="red")
lbl3.place(x=1000, y =10)
lbl4 = Label(root, text= " ", font=("Times New Roman", 14), fg="red")
lbl4.place(x=1000, y =35)
root.mainloop()
