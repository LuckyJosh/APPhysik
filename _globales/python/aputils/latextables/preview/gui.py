# -*- coding: utf-8 -*-
"""
Created on Sun Mar 16 05:45:16 2014

@author: JoshLaptop
"""

import sys
import conversion

from PyQt4 import QtGui


class Previewer(QtGui.QWidget):

    def __init__(self, imgpath, obj):
        super(Previewer, self).__init__()
        self.initUI(imgpath)
        self.obj = obj

    def initUI(self, imgpath):
        hbox = QtGui.QHBoxLayout(self)
        pixmap = QtGui.QPixmap(imgpath)

        lbl = QtGui.QLabel(self)
        lbl.setPixmap(pixmap)

        hbox.addWidget(lbl)
        self.setLayout(hbox)

        self.move(300, 200)
        self.setWindowTitle('Table Preview')
        self.show()

    def closeEvent(self, event):
        conversion.Converter.deleteTemp(self.obj)


def _show(imgpath, obj):
    app = QtGui.QApplication(sys.argv)
    preview = Previewer(imgpath, obj)
    app.exec_()


#def preview(imgpath):
#    try:
#        Gui(imgpath)
#    except (ImportError, NameError):
#        print("PyQt4 (not installed), is needed to preview the table\n"
#              + "get the Installer for 32bit or 64bit here:\n"
#              + " http://www.riverbankcomputing.co.uk/software/pyqt/download\n"
#              + "- PyQt4-4.10.3-gpl-Py2.7-Qt4.8.5-x32.exe\n"
#              + "- PyQt4-4.10.3-gpl-Py2.7-Qt4.8.5-x64.exe")
