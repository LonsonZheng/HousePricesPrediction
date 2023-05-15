# HousePricesPrediction

`data_sys.ipynb`预处理数据并进行相关可视化分析

`machine_learning.ipynb`, `deep_learning.py`分别使用几种传统机器学习方法，以及深度神经网络来进行房价预测。预训练权重分别在`DT.picle`, `GBR.pickle`, `XGBR.pickle`, `pretrained_deep_learning.pth`提供。推理方法在`test.py`中的`test`接口给出。

`UI.py`为前端组件,`house-prediction.py`为程序入口，加载了组件和模型，数据预测部分也在这一部分给出。
