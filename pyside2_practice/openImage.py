import sys
from PySide2 import QtGui, QtCore
from PySide2.QtWidgets import QFileDialog,QLabel,QAction,QMainWindow,QApplication

class Window(QMainWindow):
    def __init__(self):
        super(Window, self).__init__()
        self.setGeometry(100, 100, 500, 300)
        self.setWindowTitle("PyQT Show Image")

        openFile = QAction("&File", self)
        openFile.setShortcut("Ctrl+O")
        openFile.setStatusTip("Open File")
        openFile.triggered.connect(self.file_open)

        self.statusBar()

        mainMenu = self.menuBar()

        fileMenu = mainMenu.addMenu('&File')
        fileMenu.addAction(openFile)

        self.lbl = QLabel(self)
        self.setCentralWidget(self.lbl)

        self.home()

    def home(self):
        self.show()

    def file_open(self):
        name = QFileDialog.getOpenFileName(self, 'Open File')
        #print(name)
        self.image = QtGui.QImage(name[0])





def main():
    # a new app instance
    app = QApplication(sys.argv)
    form = Window()
    form.show()
    # without this, the script exits immediately.
    sys.exit(app.exec_())

# python bit to figure how who started This
if __name__ == "__main__":
    main()
        