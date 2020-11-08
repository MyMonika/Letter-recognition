
import numpy as np
import pandas as pd
import cv2 as cv
import sys
import os
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import pickle as pkl


def preprocess(image):#preprocess
    
    edges=cv.Canny(image,70,70) #canny edge detection
    #Sobel
    x = cv.Sobel(edges, cv.CV_16S, 1, 0) #Gray weighted difference between left and right adjacent points
    y = cv.Sobel(edges, cv.CV_16S, 0, 1) #Gray weighted difference of upper and lower adjacent points
    absX = cv.convertScaleAbs(x) #Converts to an unsigned 8-bit type.
    absY = cv.convertScaleAbs(y)
    dst = cv.addWeighted(absX, 0.5, absY, 0.5, 0)#overlap
    
    #Adaptive threshold binarization
    image = cv.adaptiveThreshold(dst, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 25, 3)
    
    #reshape the image
    image=cv.resize(image,(30,30))
    image=image.ravel() #flatting the two-dimensional matrix into one-dimensional matrix
    image[image>10]=1
    
    return image


def getimg(image):#get the cut image 
    width = 0
    
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (5,5), 0)
    thresh = cv.adaptiveThreshold(blur, 255, 1, 1, 11, 2)
    contours, _ = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    max_area = 3500
    c = 0
    global img_cut
    for i in contours:
            area = cv.contourArea(i)
            if area > max_area:
                max_area = area
                best_cnt = i
                x, y, w, h = cv.boundingRect(best_cnt)
                # For critical points coordinates
                topleft = [x,y]
                topright = [x+w,y]
                downleft = [x,y+h]
                downright = [x+w,y+h]
                cv.rectangle(image,(x,y),(x+w,y+h),(0,255,0),6)
                try: #if import error, skip it
                    img_cut = image[y+width:y+h-width,x+width:x+w-width]
                except UnboundLocalError:
                    pass

                cv.waitKey(0)
    
    return preprocess(img_cut)#call the preprocess to get the data


img_mat=[]# img container

#Process all images under this path
file_pathname=["./TrainData/A","./TrainData/B","./TrainData/C","./TrainData/D"]
for i in range(4):
    for filename in os.listdir(file_pathname[i]):
        img = cv.imread(file_pathname[i]+'/'+filename)
        pre_img = getimg(img)
        pre_ravel = pre_img.ravel()
        list1 = list(pre_ravel)
        list1.append(i+1)
        pre_ravel = np.array(list1)
        img_mat.append(pre_ravel)



data = pd.DataFrame(img_mat)
data[data > 10] = 1

#split the labels and features
X = data.iloc[:, 0:900]
y = data.iloc[:,[900]]
y = y.values.ravel()#change the format 


def classifier():#import the model
   
    fr = open("./pkl/svm.pkl", "rb")
    CF = pkl.load(fr)
    fr.close()
    # %%
    return CF, True

#start the application 
def start(classif, UI_2):
    cap = cv.VideoCapture(0)#call the camera
    while cap.isOpened():
        ok, frame = cap.read() #get the image
        if not ok:
            break

        cv.imshow('window_name', frame)
        c = cv.waitKey(10)
        
        con_flag = np.arange(5)# a array for matching the same data type
        if c & 0xFF == ord('r'):  # "R" catch the image 
        
            X_test = getimg(frame)
            X_test = X_test.ravel()
            X_test[X_test > 10] = 1
            X_test = np.array(X_test).reshape(1, -1)
            predict_cf=classif.predict(X_test)[0]
            # A:[1]  B:[2]  C:[3]  D:[4]
            if predict_cf == con_flag[1]:
                str = "the predict letter is:A"
                UI_2.write(str)
            elif predict_cf == con_flag[2]:
                str = "the predict letter is:B"
                UI_2.write(str)
            elif predict_cf == con_flag[3]:
                str = "the predict letter is:C"
                UI_2.write(str)
            elif predict_cf == con_flag[4]:
                str = "the predict letter is:D"
                UI_2.write(str)
            else:
                UI_2.write("can not disgusting it!")

        if c & 0xFF == ord('q'):  # "q" EXIT
            break
            # release the camera and destoy all the windows
    cap.release()
    cv.destroyAllWindows()

class UI_1(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setFixedSize(600, 800) #size
        self.setWindowIcon(QIcon('./图标.jpg'))#icon
        self.setWindowTitle('Letter recognition')#title
        self.paintBackground('./页面1.jpg')#background
        self.setAutoFillBackground(True)
        # define the first button
        self.btn1 = QPushButton(QIcon('./按钮29.ico'), 'Import training model', self)
        self.btn1.setFixedSize(300, 50)
        self.btn1.move(150, 325)
        self.btn1.setStyleSheet('''QPushButton{border:none;}
        QPushButton:hover{color:blue;
                    border:2px solid #F3F3F5;
                    border-radius:35px;
                    background:darkGray;}''')
        self.btn1.show()
        self.btn1.clicked.connect(self.train)
        # define the second button
        btn2 = QPushButton(QIcon('./按钮29.ico'), 'Start to recognize', self)
        btn2.setFixedSize(300, 50)
        btn2.move(150, 425)

        btn2.setStyleSheet('''QPushButton{border:none;}
            
                QPushButton:hover{color:blue;
                            border:2px solid #F3F3F5;
                            border-radius:35px;
                            background:darkGray;}''')
        btn2.clicked.connect(self.recognize)
        btn2.show()
        print("Init is gone!")
        self.show()

    # Pop up prompt box when closing
    def closeEvent(self, event):
        reply = QMessageBox.question(self, '警告', '你想要退出吗？', QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()

    def train(self):
        print("is training")
        self.box = QMessageBox(QMessageBox.Information, "Result", "模型训练完毕")
        self.box.setIcon(0)
        self.box.setGeometry(500, 500, 0, 0)
        
        #add a button
        yes = self.box.addButton('Yes', QMessageBox.YesRole)
        try:
            Classifier_, flag = classifier()
            self.box.show()
            if self.box.clickedButton() == yes:
                self.show()
        except RuntimeError:
            string1 = "模型训练错误，请重新训练"
            self.box.setText(string1)
            self.box.show()
            if self.box.clickedButton() == yes:
                self.box.hide()
                self.show()

    def recognize(self):
        # A new page will open here
        try:
            Classifier_, flag = classifier(  ) #
            b = UI_2(self)
            b.show()
            self.hide()
            str = start(Classifier_, b)
            b.write(str)
        except RuntimeError:
            string = "槽信号发送失败"
            reply2 = QMessageBox.critical(self, string, QMessageBox.Yes)
            if reply2 == QMessageBox.Yes:
                self.show()

    #     Import picture as background
    def paintBackground(self, str):
        palette = QPalette()
        pix = QPixmap(str)
        pix = pix.scaled(self.width(), self.height())
        palette.setBrush(QPalette.Background, QBrush(pix))
        self.setPalette(palette)

    def signalCall1(self):
        self.show()


class UI_2(QWidget):
    signal1 = pyqtSignal()

    def __init__(self, UI_1):
        super().__init__()
        self.initUI(UI_1)

    def initUI(self, UI_1):
        self.setFixedSize(600, 800)
        self.setWindowIcon(QIcon('./图标.jpg'))
        self.setWindowTitle('letter recognition')
        self.paintBackground('./页面2.jpg')
        self.update()
        self.signal1.connect(UI_1.signalCall1)
        # 定义一个按钮
        btn3 = QPushButton(QIcon('./按钮29.ico'), 'Previous page', self)
        btn3.setFixedSize(150, 40)
        btn3.move(0, self.height() - 40)
        btn3.setStyleSheet('''
                QPushButton:hover{color:blue;
                            border:2px solid #F3F3F5;
                            border-radius:35px;
                            background:darkGray;}''')
        btn3.clicked.connect(self.Goback)
        btn3.show()
        #   Define a label to store the result given by the classifier
        self.label = QLabel(self)
        self.label.setFixedSize(400, 100)
        self.label.move(80, 620)
        self.label.setAlignment(Qt.AlignCenter)

        self.label.setStyleSheet("QLabel{background:white;}"
                                 "QLabel{border:1px solid black;font-size:10px}"
                                 "QLabel{color:rgb(300,300,300,120);font-size:20px;font-weight:bold;font-family:宋体;}"
                                 )
        self.label.setText("let us play Letter Recognition")
        self.show()

    def write(self, str):
        self.label.setText("")
        self.label.setText(str)

    #     Back to previous unit
    def Goback(self):
        self.hide()
        self.signal1.emit()

    def paintBackground(self, str):
        palette = QPalette()
        pix = QPixmap(str)
        pix = pix.scaled(self.width(), self.height())
        palette.setBrush(QPalette.Background, QBrush(pix))
        self.setPalette(palette)


app = QApplication(sys.argv)
a = UI_1()
sys.exit(app.exec_())