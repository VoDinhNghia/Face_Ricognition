# Thư viện urllib.request dùng để mở (get) url từ trình duyệt
import urllib.request as rq
# csdl sqlite3
import sqlite3
# dùng cho dòng code 48 thư viện làm việc với hệ thống
import os
import numpy as np
import pickle
# thư viện tkinter dùng để viết giao diện ứng dụng (Lập trình gui)
from tkinter import *
# askopenfilename trong tkinter dùng để mở một file
from tkinter.filedialog import askopenfilename
# PIL thư viện dùng cho xử lý ảnh (mở một bức ảnh, hiển thị...)
from PIL import ImageTk, Image
from cv2 import *
# Tk() tạo một cửa sổ làm việc
root = Tk()
# set kích thước của cửa sổ làm việc
root.geometry("1400x700")
# VideoCapture(0) dùng để chụp một video số 0 là chỉ mục thiết bị chỉ định của máy ảnh có thể thay nó =-1
cam = cv2.VideoCapture(0)
# dòng code 23 và 24 dùng để lưu đoạn video với đuôi .avi
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640,480))
# mở bức ảnh từ file ảnh
img1=ImageTk.PhotoImage(Image.open('nen.jpg'))
# nếu dùng cv2 đọc bức ảnh thì nó chỉ hiển thị mảng các điểm ảnh
#img1 = cv2.imread('nen.jpg')
# dòng code 30 và 31 Tạo hình nền cho cửa sổ làm việc 
panel = Label(root, image = img1)
panel.image = img1
# vị trí tọa độ của bức ảnh
panel.place(x = 0, y = 0)
# dòng code 35, 36 phát hiện khuôn mặt
cascPath = "haarcascade_frontalface_default.xml"
detector  = cv2.CascadeClassifier(cascPath)
#fontface, fontscale, fontcolor là kiểu chữ, kích cỡ, màu sắc của chữ lúc hiển thị nhãn nhận dạng
fontface = cv2.FONT_HERSHEY_SIMPLEX
fontscale = 1
fontcolor = (250,0,0)
#pip install opencv-contrib-python neu da cai cv2 rồi thì gở bỏ pip uninstall
# hàm nhận dạng khuôn mặt (pridect) trong opencv
recognizer = cv2.face.LBPHFaceRecognizer_create()
#đường dẫn thư mục chứa ảnh khuôn mặt
path='anh_data_hinh'
# hàm lấy chỉ mục id và khuôn mặt trong thư mục ảnh hàm này trả về 2 mảng faces và IDs
def getImagesAndLabels(path):
    #lấy đường dẫn thư mục 
    imagePaths=[os.path.join(path,f) for f in os.listdir(path)] 
    # khởi tạo 2 mảng rỗng dùng để chứa khuôn mặt và id
    faces=[]
    IDs=[]
    #tạo vòng lặp để lấy hết tất cả đường dẫn bức ảnh và hình ảnh trong thư mục
    for imagePath in imagePaths:
        # mở ảnh và chuyển về chế độ L(Thang độ xám) ngoài ra còn có chế độ P
        faceImg=Image.open(imagePath).convert('L')
        # ép kiểu dữ liệu uint8
        faceNp=np.array(faceImg,'uint8')
        #cắt lấy chỉ mục id của đường dẫn bức ảnh
        ID=int(os.path.split(imagePath)[-1].split('.')[1])
        # thêm vào mảng faces và IDs tạo trước đó
        faces.append(faceNp)
            #print(ID)
        IDs.append(ID)
    return IDs, faces
# hàm tạo kết nối và truy vấn vào bảng FaceDB trong csdl để lấy id và tên
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
    # lấy đường link từ ô nhập liệu trên app
    link = lbl_NID.get()
    # mở đường link
    resource = rq.urlopen(link)
    #mở file tạm 
    output = open("file01.jpg","wb")
    # ghi ảnh từ url vào file tạm và đóng lại
    output.write(resource.read())
    output.close() 
    #mở file tạm có chứa ảnh từ đường dẫn url
    img1 = ImageTk.PhotoImage(Image.open('file01.jpg')) 
    # dòng code 90 và 91 hiển thị bức ảnh lấy từ url lên giao diện
    panel = Label(root, image = img1)
    panel.image = img1
    #set vị trí bức ảnh
    panel.place(x = 200, y = 70)
    #đọc bức ảnh từ file tạm với cv2
    img = cv2.imread('file01.jpg')
    #fontface, fontscale, fontcolor là kiểu chữ, kích cỡ, màu sắc của chữ lúc hiển thị nhãn nhận dạng
    fontface = cv2.FONT_HERSHEY_SIMPLEX
    fontscale = 1
    fontcolor = (120,0,0)
    # gọi lại hàm getImagesAndLabels() đã xây dựng ở trên để lấy id và khuôn mặt 
    Ids,faces=getImagesAndLabels(path)
    # trainning dữ liệu
    recognizer.train(faces,np.array(Ids))
    #Lưu file tranning vào thư mục
    recognizer.save('reco_anh/trainningData.yml')
    #chuyển về ảnh xám
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # detectMultiScale phát hiện khuôn mặt trong bức ảnh xám
    faces=detector.detectMultiScale(gray,1.3,5)
    #global thong_tin
    #vòng lặp lấy hết khuôn mặt đc phát hiện trong faces
    for(x,y,w,h) in faces:
        #vẽ hình chữ nhật lên khuôn mặt được phát hiện trong bức ảnh đầu vào
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        #dự đoán(nhận dạng khuôn mặt)
        id,conf=recognizer.predict(gray[y:y+h,x:x+w])
        #gọi hàm getProfile(id) để lấy id và tên từ csdl id là tham số đc lấy từ hàm predict
        profile=getProfile(id)
        if(profile!=None):
            #ghi nhãn dự đoán đc lên bức ảnh
            cv2.putText(img, "Name: " + str(profile[1]), (x-30,y-20), fontface, fontscale, fontcolor ,2)
            #lấy tên để hiển thị lên giao diện app (profile[1] là tên profile[0] là id)
            thong_tin=str(profile[1])  
      
    a = "so guong mat nguoi trong anh :" + str(len(faces))
    b = "Ten : " + profile[1]
    #hiển thị số gương mặt và tên có trong bức ảnh lên giao diện app
    lbl3.configure(text = a)
    lbl4.configure(text=b)        
    cv2.imshow('Face',img)
# hàm nhập hình ảnh và nhãn vào csdl và thư mục ảnh  
def btn_nhaphinh():
    #Tạo cửa sổ làm việc mới
    tk = Tk()
    #set size
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
        # lấp id và name từ ô nhập liệu tên app
        id= lbl_NID.get()
        name= lbl_T.get()
        #khi nhập để trong cặp dấu nháy 
        insertOrUpdate(id,name)
        sampleNum=0
        while(True):
            #mở file chọn ảnh
            img2 = askopenfilename()
            #đọc bức ảnh
            img = cv2.imread(img2)
            #chuyển về ảnh xám
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # detectMultiScale phát hiện khuôn mặt trong bức ảnh xám
            faces = detector.detectMultiScale(gray, 1.3, 5)
            for (x,y,w,h) in faces:
                #tạo khung hình chữ nhật trc khuôn mặt của bức ảnh dòng code này không cần thiết trong vòng lặp này
                cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2) 
                sampleNum=sampleNum+1
                #thêm vào thư mục 
                cv2.imwrite("anh_data_hinh/User."+ id +'.'+ str(sampleNum) + ".jpg", gray[y:y+h,x:x+w])
            # nếu lớn hơn 50 thì dừng vòng lặp cắt ảnh
            if sampleNum>50:
                break
    #tạo tiêu đề
    lbl_ID = Label(tk, text="ID", fg="green", font=("Times New Roman", 16), width= 10)
    #set vị trí
    lbl_ID.place(x=5, y = 15)
    #tạo ô nhập liệu trên app
    lbl_NID = Entry(tk, width=35, font=("Times New Roman", 14))
    lbl_NID.place(x=100, y = 15)

    lbl_Ten = Label(tk, text="Tên", fg="green", font=("Times New Roman", 16), width= 10)
    lbl_Ten.place(x=5, y = 50)
    lbl_T = Entry(tk, width=35, font=("Times New Roman", 14))
    lbl_T.place(x=100, y=50)
    # tạo button sự kiện để ghi ảnh vào thư mục, id và tên vào csdl
    btn = Button(tk, text="Ghi", font=("Times New Roman", 14), fg="white", bg="green",
              width=12, height=1, command=btn_ghi)
    btn.place(x=170, y=120)
# hàm nhận dạng ảnh từ file ảnh trong thư mục của máy
def btn_nhandang_anh():
    #tạo biến toàn cục
    global img
    #mở file để chọn ảnh
    img = askopenfilename()
    #mở file ảnh vừa chọn
    img1 = ImageTk.PhotoImage(Image.open(img))
    #dòng 199 và 200 hiển thị bức ảnh lên giao diện app 
    panel = Label(root, image = img1)
    panel.image = img1
    #set vị trí bức ảnh
    panel.place(x = 200, y = 70)
    #đọc bức ảnh với cv2
    img = cv2.imread(img)
    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #fontface, fontscale, fontcolor là kiểu chữ, kích cỡ, màu sắc của chữ lúc hiển thị nhãn nhận dạng
    fontface = cv2.FONT_HERSHEY_SIMPLEX
    fontscale = 1
    fontcolor = (120,0,0)
    #gọi hàm getImagesAndLabels() để lấy id và dữ liệu khuôn mặt
    Ids,faces=getImagesAndLabels(path)
    recognizer.train(faces,np.array(Ids))
    recognizer.save('reco_anh/trainningData.yml')
    #chuyển về ảnh xám
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # detectMultiScale phát hiện khuôn mặt trong bức ảnh xám
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
    #fontface, fontscale, fontcolor là kiểu chữ, kích cỡ, màu sắc của chữ lúc hiển thị nhãn nhận dạng
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
