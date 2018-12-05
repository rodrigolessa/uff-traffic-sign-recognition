# Timer -	Displays a seconds timer with start and stop buttons.
#           Control-c and control-q cause it to exit.
# Selecionar imagem
# Se for RGB
#   Selecionar cada canal da imagem para processamento
#   Selecionar tipo de conversão em outros espaços de cores
# Selecionar tipo de filtro para remover ruido (gaussiano)
# Selecionar tipo de treshold
# Selecionar grau do treshold
# Selecionar tipo de binarização
# Selecionar extração de caracteristicas

import sys
from PyQt5.QtWidgets import (QApplication, QWidget, QHBoxLayout, QVBoxLayout, QLabel, QLineEdit, QSlider, QPushButton, QCheckBox, QRadioButton, QTextEdit, QFileDialog)
#from PyQt5 import QtWidgets, QtGui
from PyQt5.QtCore import Qt
#from PyQt5.QtGui import *

class Window(QWidget):

    def __init__(self):

        super(Window, self).__init__()
        self.texte = QTextEdit(self)

        self.init_ui()

    def init_ui(self):
        self.lblVideo = QLabel('Video')
        self.lblOutput = QLabel('Output file')
        self.lblThresh = QLabel('Threshold')
        self.lblBlob = QLabel('Minimum blob size')

        self.txtOutput = QLineEdit()

        self.sld = QSlider(Qt.Horizontal)
        self.sld.setMinimum(1)
        self.sld.setMaximum(70)
        self.sld.setValue(25)
        self.sld.setTickInterval(10)
        self.sld.setTickPosition(QSlider.TicksBelow)

        self.chk = QCheckBox('Gaussian filter')

        self.aFilter = QRadioButton("A Filter")
        self.bFilter = QRadioButton("B Filter")

        self.btnAnalyse = QPushButton('Analyse')
        self.btnClear = QPushButton('Clear')

        self.btnAnalyse.clicked.connect(lambda: self.btnAnalyseClick(self.btnAnalyse, 'Analyse'))
        self.btnClear.clicked.connect(lambda: self.btnAnalyseClick(self.btnClear, 'Clear'))
        self.sld.valueChanged.connect(self.v_change)

        hbox = QHBoxLayout()
        hbox.addWidget(self.lblVideo)
        hbox.addStretch()
        hbox.addWidget(self.lblOutput)
        hbox.addWidget(self.txtOutput)
        hbox.addStretch()

        hboxFilter = QHBoxLayout()
        hboxFilter.addWidget(self.aFilter)
        hboxFilter.addWidget(self.bFilter)
        hboxFilter.addStretch()

        vbox = QVBoxLayout()
        vbox.addLayout(hbox)
        vbox.addLayout(hboxFilter)
        vbox.addWidget(self.chk)
        vbox.addWidget(self.lblThresh)
        vbox.addWidget(self.sld)
        vbox.addWidget(self.lblBlob)
        vbox.addWidget(self.texte)
        vbox.addWidget(self.btnAnalyse)
        vbox.addWidget(self.btnClear)

        # Left, Top, Width, Height
        #self.setGeometry(500, 400, 500, 250)
        self.setLayout(vbox)
        self.setWindowTitle('Computer Vision - Analysing Image')
        self.show()

    def btnAnalyseClick(self, b, texto):
        if b.text() == 'Analyse':
            self.lblBlob.setText(self.txtOutput.text())
        else:
            self.texte.clear()
        if self.chk.isChecked():
            print('Gaussian filter')
        print(texto)
        if self.aFilter.isChecked():
            print('Filter A')
        if self.bFilter.isChecked():
            print('Filter B')

    def v_change(self):
        sldval = str(self.sld.value())
        self.lblBlob.setText(sldval)

app = QApplication(sys.argv)
clsWindow = Window()

sys.exit(app.exec_())