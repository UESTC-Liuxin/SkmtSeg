#!/usr/bin/python3
# @Time    : 2020/4/11 下午8:46
# @Author  : zwenc
# @Email   : zwence@163.com
# @File    : mainUI.py
import sys
sys.path.append(".")
import os
import numpy as np
import cv2
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QFileDialog, QComboBox, QLineEdit, QListWidget, QListWidgetItem
from PyQt5 import QtGui
from PyQt5.QtCore import pyqtSignal, Qt
from PyQt5 import QtCore
from UI.layoutFile.Ui_MainWindows import Ui_MainWindow

from predict import Predict

class MainWindow(QMainWindow, Ui_MainWindow):
    disImageSignal = pyqtSignal()

    def __init__(self, config):
        super(MainWindow, self).__init__()
        self.setupUi(self)

        self.config = config

        # init UI
        # self.imageRadio.setChecked(True)

        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("UI/Logo/logo.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.setWindowIcon(icon)
        self.background.setPixmap(QtGui.QPixmap("UI/logo/background.png"))
        self.background.setScaledContents(True)
        self.userlogo.setPixmap(QtGui.QPixmap("UI/logo/userlogo.png"))
        self.userlogo.setScaledContents(True)

        self.setWindowFlags(Qt.FramelessWindowHint)  # 去边框
        self.setAttribute(Qt.WA_TranslucentBackground)  # 设置窗口背景透明
        
        # self.setWindowTitle("骨骼肌超声")

        with open("UI/css/style.css","r", encoding='utf-8') as file_css:
            self.setStyleSheet(file_css.read())

        # init parameter
        self.fileName = None
        self.m_drag = False

        self.process = Predict(self.config, self.callback)

        # init Signal
        self.disImageSignal.connect(self.imageShow)
        self.statusBar().showMessage("正在加载权重，请耐性等待", 1000)

    def loadFileButton(self):
        try:
            fileName, fileType = QFileDialog.getOpenFileName(self, "选取文件", os.getcwd(),
                                                                "All Files(*);;Text Files(*.txt)")
            if fileName == "":
                return
            self.fileName = fileName

            self.config.show_image = cv2.imread(fileName)
            if self.config.show_image is None:
                return 

            self.disImageSignal.emit()
                # self.beginRecognizeButton()
                # self.displayImage = cv2.imread(fileName)

            self.statusBar().showMessage("数据导入成功", 1000)

        except Exception as e:
            print(e)

    def callback(self):
        # print("callback return")
        self.disImageSignal.emit()

    def boxCheck(self, a):
        if self.checkBox.isChecked():
            self.beginRecognizeButton()            

    def beginRecognizeButton(self):
        if self.fileName is not None:
            self.process.run(self.fileName, int(self.comboBox.currentIndex() + 1),callBack=self.callback)
            self.statusBar().showMessage("请耐性等待计算", 1000)
        else:
            self.statusBar().showMessage("请先加载图片", 1000)

    def imageShow(self):
        try:
            while len(self.config.message) != 0:
                msg = self.config.message.pop()
            
                if msg == "load weight success":
                    self.checkBox.setChecked(True)
                self.statusBar().showMessage(msg, 2000)

            # self.process.show()
            if self.config.show_image is not None:
                temp = cv2.cvtColor(self.get_exhibit_image(self.config.show_image, self.image.width(), self.image.height()), 
                                    cv2.COLOR_BGR2RGB)
                qtImage = QtGui.QImage(temp.data, temp.shape[1],
                                    temp.shape[0], temp.shape[1] * 3, QtGui.QImage.Format_RGB888)
                self.image.setPixmap(QtGui.QPixmap(qtImage))

        except Exception as e:
            print(e)

    def get_exhibit_image(self, image, width, height):

        if isinstance(image, type(None)):
            logger.warning("haven't set show image")
            return cv2.imread("UI/Logo/logo.png")
            # return None
        # logger.info("image size = {}, label width = {}, label height = {}".format(self.image.shape, width, height))
        rate = min (height / image.shape[0], width / image.shape[1], float(self.config.config["windows"]["max_enlarge_rate"]))
        
        return cv2.resize(image, (0, 0), fx=rate, fy=rate)

    def minWindow(self):
        self.setWindowState(Qt.WindowMinimized)

    def mcloseWindow(self):
        sender = self.sender()
        app = QApplication.instance()
        app.quit()
            
    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self.m_drag = True
            self.m_DragPosition = e.globalPos() - self.pos()
            e.accept()

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.LeftButton:
            self.m_drag = False

    def mouseMoveEvent(self, e):
        try:
            if Qt.LeftButton and self.m_drag:
                self.move(e.globalPos() - self.m_DragPosition)
                e.accept()
        except Exception as e:
            print("错误代码:000x0", e)


# test code
if __name__ == "__main__":
    app = QApplication(sys.argv)
    main = MainWindow()
    main.show()
    sys.exit(app.exec_())
