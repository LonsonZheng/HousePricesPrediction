# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'connect_me.ui'
#
# Created by: PyQt5 UI code generator 5.11.3
#
# WARNING! All changes made in this file will be lost!
#导入程序运行必须模块
import sys
import torch
import torch.nn as nn
import numpy as np
import pickle
import sklearn.model_selection as ms
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor as XGBR
from sklearn.ensemble import GradientBoostingRegressor
import pandas as pd
#PyQt5中使用的基本控件都在PyQt5.QtWidgets模块中
from PyQt5.QtWidgets import QApplication, QMainWindow
#导入designer工具生成的login模块
from UI import Ui_Form

class Linear_Net(nn.Module):
    def __init__(self,in_dim=14,out_dim=1,hidden=100) :
        super().__init__()
        self.body=nn.Sequential(
            nn.Linear(in_dim,hidden),
            nn.ReLU(),
            nn.Linear(hidden,hidden),
            nn.ReLU(),
            nn.Linear(hidden,hidden),
            nn.ReLU(),
            nn.Linear(hidden,out_dim),
        )

    def forward(self,inp):
        return  self.body(inp)

class MyMainForm(QMainWindow, Ui_Form):
    def __init__(self, parent=None):
        super(MyMainForm, self).__init__(parent)
        self.setupUi(self)
        #添加登录按钮信号和槽。注意display函数不加小括号()
        self.pushButton1.clicked.connect(self.display)
        #添加退出按钮信号和槽。调用close函数
        self.pushButton2.clicked.connect(self.close)
    def area(self, Str):
        if Str == "城区":
            return 10
        if Str == "海淀区":
            return 1
        if Str == "朝阳区":
            return 8
        if Str == "丰台区":
            return 9
        if Str == "石景山区":
            return 7
        if Str == "大兴区":
            return 2
        if Str == "昌平区":
            return 11
        if Str == "通州区":
            return 12
        if Str == "顺义区":
            return 3
        if Str == "门头沟区":
            return 14
        if Str == "房山区":
            return 13
        if Str == "怀柔区":
            return 15
        if Str == "密云区":
            return 4
        if Str == "延庆区":
            return 6
        if Str == "平谷区":
            return 5
        return 12
    def get_input(self):
        ans = []
        DOM = 800
        ans.append(DOM)
        followers = 100
        ans.append(followers) #欢迎人数
        square = self.lineEdit2.text()
        ans.append(float(square))  #面积
        livingRoom = self.lineEdit.text()
        ans.append(float(livingRoom)) #卧室
        drawingRoom = self.lineEdit_2.text()
        ans.append(float(drawingRoom)) #课厅
        kitchen = self.lineEdit_3.text()
        ans.append(float(kitchen)) #厨房
        bathRoom = self.lineEdit_4.text()
        ans.append(float(bathRoom)) #浴室
        floor = self.lineEdit3.text()
        ans.append(float(floor)) #楼层
        renovationCondition = 2
        ans.append(renovationCondition) #翻新条件
        buildingStructure = 6
        ans.append(buildingStructure) #建造结构
        elevator = 0
        if self.comboBox.currentText() == "有":
            elevator = 1
        else:
            elevator = 2
        ans.append(elevator) #电梯有无
        subway = 0
        if self.comboBox_2.currentText() == "有":
            subway = 1
        else:
            subway = 0
        ans.append(subway) #地铁有无
        district = self.area(self.comboBox2.currentText())
        ans.append(float(district)) #区域
        tradeYear = self.comboBox1.currentText()
        ans.append(float(tradeYear)) #交易年份
        return ans
    def derive_positions(self, x_test, models):
        answer = pd.DataFrame()
        for model in models.keys():
            answer[model] = models[model].predict(x_test)
        return answer
    def display(self):
        data = self.get_input()
        scale_list = np.array([1677.0, 1143, 1745.5, 9.0, 5.0, 4, 5.0, 63.0, 4, 6, 1.0, 1.0, 13, 2018])

        data /= scale_list
        data = data.reshape(1, -1)
        inp = torch.FloatTensor(np.array(data))
        dnn = Linear_Net()
        pretrained = torch.load('pretrained_deep_learning.pth', map_location='cpu')
        dnn.load_state_dict(pretrained['state_dict'], strict=True)
        with torch.no_grad():
            dnn_ans = dnn(inp)
        models = {}

        f = open('DT.pickle', 'rb')
        models['DT'] = pickle.load(f)
        f.close()

        f = open('GBR.pickle', 'rb')
        models['GBR'] = pickle.load(f)
        f.close()

        f = open('XGBR.pickle', 'rb')
        models['XGBR'] = pickle.load(f)
        f.close()

        ans = self.derive_positions(data, models)
        ans['DNN'] = dnn_ans
        scale = 18130
        ans *= scale
        print(ans)
        temp = round(float((ans['DT']+ans['GBR']+ans['XGBR'])/3),2)
        self.textBrowser1.setText("该地区房价"+str(temp)+"万元")
if __name__ == "__main__":
    #固定的，PyQt5程序都需要QApplication对象。sys.argv是命令行参数列表，确保程序可以双击运行
    app = QApplication(sys.argv)
    #初始化
    myWin = MyMainForm()
    #将窗口控件显示在屏幕上
    myWin.show()
    #程序运行，sys.exit方法确保程序完整退出。
    sys.exit(app.exec_())