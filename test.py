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
def derive_positions(x_test,models):
    answer = pd.DataFrame()
    for model in models.keys():
        answer[model] = models[model].predict(x_test)
    return answer
# def pickle_process(models):
#     for model in models.keys():
#         models[model] = models[model].best_estimator_
#         f = open('{}.pickle'.format(model),'wb')
#         pickle.dump(models[model],f)
#         f.close()

# 输入为numpy array格式的数据，共14维，形状为[14, ]，1D
# process_col=["DOM", "followers", "square", "livingRoom", "drawingRoom", "kitchen", "bathRoom", "floor", "renovationCondition", "buildingStructure", "elevator", "subway", "district", "tradeyear"]
# 输出房价预测, dataframe类型，包括'DT'，'RF'，'GBR'，'XGBR'，'DNN'，单位：万元

def test(data):
    data=data.copy()
    scale_list=np.array([1677.0, 1143, 1745.5, 9.0, 5.0, 4, 5.0, 63.0, 4, 6, 1.0, 1.0, 13, 2018])
    
    data/=scale_list
    data=data.reshape(1,-1)
    inp=torch.FloatTensor(np.array(data))
    dnn=Linear_Net()
    pretrained=torch.load('pretrained_deep_learning.pth')
    dnn.load_state_dict(pretrained['state_dict'],strict=True)
    with torch.no_grad():
        dnn_ans=dnn(inp)
    models={}

    f = open('DT.pickle','rb')
    models['DT'] = pickle.load(f)
    f.close()

    f = open('GBR.pickle','rb')
    models['GBR'] = pickle.load(f)
    f.close()

    f = open('XGBR.pickle','rb')
    models['XGBR'] = pickle.load(f)
    f.close()

    ans=derive_positions(data,models)
    ans['DNN']=dnn_ans
    scale=18130
    ans*=scale
    return ans

if __name__=='__main__':
    data = pd.read_csv("data_pre3.csv", encoding="gbk")#
    process_col=["DOM", "followers", "square", "livingRoom", "drawingRoom", "kitchen", "bathRoom", "floor", "renovationCondition", "buildingStructure", "elevator", "subway", "district", "tradeyear"]
    x =data[process_col]
    # for nkey in process_col:
    #     x[nkey]=x[nkey].div(x[nkey].max())
    print('GT:{} '.format(data['totalPrice'].loc[5]))
    x1=np.array([1.464e+03,1.060e+02, 1.310e+02, 2.000e+00 ,1.000e+00, 1.000e+00, 1.000e+00, 2.600e+01, 3.000e+00, 6.000e+00, 1.000e+00, 1.000e+00 ,7.000e+00, 2.016e+03])
    x2=np.array([9.0300e+02,1.2600e+02, 1.3238e+02, 2.0000e+00 ,2.0000e+00, 1.0000e+00
        ,2.0000e+00 ,2.2000e+01 ,4.0000e+00 ,6.0000e+00 ,1.0000e+00 ,0.0000e+00,
        7.0000e+00 ,2.0160e+03])
    x3=np.array([8.610e+02, 5.700e+01, 5.300e+01 ,1.000e+00 ,0.000e+00 ,1.000e+00 ,0.000e+00,
            8.000e+00 ,3.000e+00 ,6.000e+00 ,1.000e+00, 0.000e+00, 7.000e+00 ,2.016e+03])
    print(test(x1))
    print(test(x1))
    print(test(x1))
    print(test(x2))
    print(test(x2))
    print(test(x3))
    print(test(x3))
    
