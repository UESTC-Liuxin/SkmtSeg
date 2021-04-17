# -*- coding: utf-8 -*-
'''
@copyright: zwenc
@email: zwence@163.com
@Date: 2020-05-02 21:02:42
@FilePath: \SkmtSeg\mainui.py
'''

from UI.mainUI import *
from UI.datainfo import *

if __name__ == "__main__":
    config = DataInfo()
    config.init("config/config.ini")
    
    app = QApplication(sys.argv)
    main = MainWindow(config)
    main.show()
    sys.exit(app.exec_())
