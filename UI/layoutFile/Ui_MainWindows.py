
# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'd:\Users\Zwen\Desktop\SkmtSeg\UI\layoutFile\MainWindows.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(914, 632)
        MainWindow.setMinimumSize(QtCore.QSize(914, 632))
        MainWindow.setMaximumSize(QtCore.QSize(914, 632))
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.background = QtWidgets.QLabel(self.centralwidget)
        self.background.setEnabled(True)
        self.background.setGeometry(QtCore.QRect(-40, -40, 991, 701))
        self.background.setStyleSheet("border-color: rgb(5, 147, 255);")
        self.background.setText("")
        self.background.setPixmap(QtGui.QPixmap("../Logo/background.png"))
        self.background.setScaledContents(True)
        self.background.setWordWrap(False)
        self.background.setObjectName("background")
        self.userlogo = QtWidgets.QLabel(self.centralwidget)
        self.userlogo.setGeometry(QtCore.QRect(-30, -30, 321, 191))
        self.userlogo.setText("")
        self.userlogo.setPixmap(QtGui.QPixmap("../Logo/userlogo.png"))
        self.userlogo.setScaledContents(True)
        self.userlogo.setObjectName("userlogo")
        self.minwindow = QtWidgets.QPushButton(self.centralwidget)
        self.minwindow.setGeometry(QtCore.QRect(830, 30, 21, 21))
        self.minwindow.setText("")
        self.minwindow.setObjectName("minwindow")
        self.closeWindow = QtWidgets.QPushButton(self.centralwidget)
        self.closeWindow.setGeometry(QtCore.QRect(860, 30, 21, 21))
        self.closeWindow.setText("")
        self.closeWindow.setObjectName("closeWindow")
        self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox.setGeometry(QtCore.QRect(60, 110, 271, 481))
        self.groupBox.setObjectName("groupBox")
        self.pushButton = QtWidgets.QPushButton(self.groupBox)
        self.pushButton.setGeometry(QtCore.QRect(40, 40, 191, 51))
        self.pushButton.setObjectName("pushButton")
        self.checkBox = QtWidgets.QCheckBox(self.groupBox)
        self.checkBox.setEnabled(False)
        self.checkBox.setGeometry(QtCore.QRect(40, 450, 71, 16))
        self.checkBox.setCheckable(True)
        self.checkBox.setObjectName("checkBox")
        self.comboBox = QtWidgets.QComboBox(self.groupBox)
        self.comboBox.setGeometry(QtCore.QRect(40, 130, 191, 51))
        self.comboBox.setObjectName("comboBox")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.groupBox_2 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_2.setGeometry(QtCore.QRect(350, 110, 531, 481))
        self.groupBox_2.setStyleSheet("gridline-color: rgb(0, 85, 255);")
        self.groupBox_2.setObjectName("groupBox_2")
        self.image = QtWidgets.QLabel(self.groupBox_2)
        self.image.setGeometry(QtCore.QRect(20, 20, 491, 441))
        self.image.setAcceptDrops(False)
        self.image.setStyleSheet("")
        self.image.setText("")
        self.image.setAlignment(QtCore.Qt.AlignCenter)
        self.image.setObjectName("image")
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        self.minwindow.clicked.connect(MainWindow.minWindow)
        self.closeWindow.clicked.connect(MainWindow.mcloseWindow)
        self.pushButton.clicked.connect(MainWindow.loadFileButton)
        self.comboBox.activated['int'].connect(MainWindow.boxCheck)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.groupBox.setTitle(_translate("MainWindow", "控制"))
        self.pushButton.setText(_translate("MainWindow", "加载图片"))
        self.checkBox.setText(_translate("MainWindow", "Ready"))
        self.comboBox.setItemText(0, _translate("MainWindow", "切片1"))
        self.comboBox.setItemText(1, _translate("MainWindow", "切片2"))
        self.comboBox.setItemText(2, _translate("MainWindow", "切片3"))
        self.comboBox.setItemText(3, _translate("MainWindow", "切片4"))
        self.comboBox.setItemText(4, _translate("MainWindow", "切片5"))
        self.groupBox_2.setTitle(_translate("MainWindow", "显示"))

