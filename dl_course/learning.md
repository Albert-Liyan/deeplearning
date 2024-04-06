
课程大纲

- 1 神经网络和深度学习
- 2 深度神经网络实践（超参调优、正则化、高级优化算法）
- 3 结构化机器学习project，最佳实践
- 4 CNN
- 5 序列模型，NLP

Tools of deep learning！ You be able to apply them, to advance your career.


第一门课 神经网络和深度学习(Neural Networks and Deep Learning)
==============================================================

#### 1 神经网络和深度学习
- [1.1 什么是神经网络](#11-什么是神经网络)
- [1.2 神经网络进行监督学习](#12-神经网络进行监督学习)
- [1.3 为什么深度学习会兴起](#13-为什么深度学习会兴起)

##### 1.1 什么是神经网络
Housing Price Prediction
 - A simple neuro network
 ![alt text](image.png)
 - A larger neuro network
 ![alt text](image-1.png) 

神经网络非常擅长计算x->y的映射函数，只需提供足够训练样本（x，y）
监督学习例子

##### 1.2 神经网络进行监督学习

Supervised Learning
![alt text](image-2.png)

术语：
- 结构化数据 Structured
每个特征 如房间大小、用户年龄等有明确、清晰定义
公司的海量数据等

- 非结构化数据 Unstructured
音频、原始音频、图像、文本等
特征：图像的像素值、文本中的单词...

计算机更难理解非结构化数据，但人类更擅长理解非结构化数据（音视频信号、文本）。
正是由于深度学习、神经网络，计算机可以更好的解释非结构化数据。

##### 1.3 为什么深度学习会兴起

x轴表示数据量
y轴表示机器学习算法性能（svm、逻辑回归等）
![alt text](image-3.png)

传统机器学习算法在 喂入数据量越来越多时，算法性能出现“瓶颈”。近20年来，收集海量数据，远远超过传统机器学习算法能发挥作用的数据规模。


![alt text](image-4.png)

“scale”一直在推动深度学习进步
“规模”不仅仅是神经网络的规模，还有数据规模！

术语：
labled data 标签数据，（x，y）
m表示 标签数据量

总结下来，深度学习技术发展的因素
- Data
- Computation
- Algo
算法创新 是代码运行的更快，从而能够训练更大的神经网络
举例：Sigmod函数 -> Relu函数，梯度下降法会运行更快

#### 2 神经网络编程
- [2.1 二分类](#21-二分类binary-classification)
- [2.2 逻辑回归](#22-逻辑回归logistic-regression)
- [2.3 逻辑回归的代价函数](#23-逻辑回归的代价函数logistic-regression-cost-function)
- [2.4 梯度下降法](#24-梯度下降法gradient-descent)
- [2.5 导数](#25-导数derivatives)
- [2.6 更多导数例子](#26-更多的导数例子more-derivative-examples)
- [2.7 计算图](#27-计算图computation-graph)
- [2.8 使用计算图求导数](#28-使用计算图求导数derivatives-with-a-computation-graph)
- [2.9 逻辑回归中的梯度下降](#29-逻辑回归中的梯度下降logistic-regression-gradient-descent)
- [2.10 m个样本的梯度下降](#210-m个样本的梯度下降gradient-descent-on-m-examples)
- [2.11 向量化](#211-向量化vectorization)
- [2.12 向量化的更多例子](#212-向量化的更多例子more-examples-of-vectorization)
- [2.13 向量化逻辑回归](#213-向量化逻辑回归vectorizing-logistic-regression)
- [2.14 向量化logistic回归的梯度输出](#214-向量化-logistic-回归的梯度输出vectorizing-logistic-regressions-gradient)
- [2.15 Python 中的广播](#215-python-中的广播broadcasting-in-python)
- [2.16 关于 python numpy向量的说明](#216-关于-python-_-numpy-向量的说明a-note-on-python-or-numpy-vectors参考视频)
- [2.17 Jupyter/iPython Notebooks快速入门](#217-jupyteripython-notebooks快速入门quick-tour-of-jupyteripython-notebooks)
- [2.18 logistic 损失函数的解释](#218-选修logistic-损失函数的解释explanation-of-logistic-regression-cost-function)

关键词：编程技巧，神经网络计算过程，正向传播，反向传播，logistic回归算法

##### 2.1 二分类(Binary Classification)
符号：
维度 nx,n
样本 (x,y)
m个样本

![alt text](image-5.png)

##### 2.2 逻辑回归(Logistic Regression)
二元分类问题，输出y标签是0或1

![alt text](image-6.png)

##### 2.3 逻辑回归的代价函数（Logistic Regression Cost Function）
符号说明，训练样本i对应的预测值y^(i)，上标i

损失函数
误差平方 不是一个好的选择，因为会得到一个非凸的优化问题，梯度下降法得不到全局最优解

损失函数 vs 成本函数
损失函数Loss Function，衡量单个训练样本
成本函数Cost Function，衡量m个训练样本总的损失

![alt text](image-7.png)

逻辑回归可以看做非常小的神经网络

##### 2.4 梯度下降法（Gradient Descent）

![alt text](image-9.png)
梯度下降法原理
dw、db表示偏导

##### 2.5 导数（Derivatives）
不需要深入学习微积分，直观认识即可

##### 2.6 更多的导数例子（More Derivative Examples）
##### 2.7 计算图（Computation Graph）
一个神经网络计算过程：计算神经网络输出，反向传输操作，计算对应梯度或导数
why？用这样方式

![alt text](image-10.png)
蓝色箭头表示 前向计算，红色表示反向计算
这是计算导数最自然的方式

##### 2.8 使用计算图求导数（Derivatives with a Computation Graph）

微积分链式法则

![alt text](image-11.png)
代码中 dvar表示对变量var求导数

##### 2.9 逻辑回归中的梯度下降（Logistic Regression Gradient Descent）
单个训练样本梯度下降算法
![alt text](image-12.png)

##### 2.10 m个样本的梯度下降(Gradient Descent on m Examples)
for循环是低效的，需要用向量化代替
![alt text](image-13.png)

##### 2.11 向量化(Vectorization)
SIMD 单指令多数据并行技术，使向量化 np.dot(w,x)计算加速
![alt text](image-14.png)

##### 2.12 向量化的更多例子（More Examples of Vectorization）
尽量使用numpy内置函数，避免使用for循环

##### 2.13 向量化逻辑回归(Vectorizing Logistic Regression)
正向计算 向量化

![alt text](image-15.png)

##### 2.14 向量化 logistic 回归的梯度输出（Vectorizing Logistic Regression's Gradient）
反向计算 向量化

![alt text](image-16.png)

##### 2.15 Python 中的广播（Broadcasting in Python）
矩阵A (3,4)
cal = A.sum(axis=0) // axis=0表示沿着竖直方向求和，水平轴是axis=1
percentage = 100 * A/(cal.reshape(1, 4))  //广播

神经网络中主要用到的广播形式：
![alt text](image-17.png)

##### 2.16 关于 python _ numpy 向量的说明（A note on python or numpy vectors）参考视频：
python numpy提供编程灵活性优势，但也容易引入编程bug。如果不是非常熟悉python numpy，很容易引入错误。
技巧：
不要使用rank=1的数组，例如：
a = np.random.randn(5) //rank为1的数组，既不是行向量、也不是列向量
print(a.shape)  // (5,)
应该定义成行向量 或列向量
a = np.random.randn(1，5)
a = np.random.randn(5，1)
##### 2.17 Jupyter/iPython Notebooks快速入门（Quick tour of Jupyter/iPython Notebooks）
##### 2.18 （选修）logistic 损失函数的解释（Explanation of logistic regression cost function）


#### 3 浅层神经网络(Shallow neural networks)

- [3.1 神经网络概述](#31-神经网络概述neural-network-overview)
- [3.2 神经网络的表示](#32-神经网络的表示neural-network-representation)
- [3.3 计算一个神经网络的输出](#33-计算一个神经网络的输出computing-a-neural-networks-output)
- [3.4 多样本向量化](#34-多样本向量化vectorizing-across-multiple-examples)
- [3.5 向量化实现的解释](#35-向量化实现的解释justification-for-vectorized-implementation)
- [3.6 激活函数](#36-激活函数activation-functions)
- [3.7 为什么需要非线性激活函数](#37-为什么需要非线性激活函数why-need-a-nonlinear-activation--function)
- [3.8 激活函数的导数](#38-激活函数的导数derivatives-of-activation-functions)
- [3.9 神经网络的梯度下降](#39-神经网络的梯度下降gradient-descent-for-neural-networks)
- [3.10（选修）直观理解反向传播](#310选修直观理解反向传播backpropagation-intuition)
- [3.11 随机初始化](#311-随机初始化randominitialization)

##### 3.1 神经网络概述（Neural Network Overview）
什么是神经网络？

逻辑回归 vs 神经网络
![alt text](image-18.png)

##### 3.2 神经网络的表示（Neural Network Representation）
“激活”的含义：网络中准备传递给下一层的值
a[0] = x
a[1] 
2 layers网络（输入层x不算）

##### 3.3 计算一个神经网络的输出（Computing a Neural Network's output）
![alt text](image-19.png)
##### 3.4 多样本向量化（Vectorizing across multiple examples）

##### 3.5 向量化实现的解释（Justification for vectorized implementation）
##### 3.6 激活函数（Activation functions）
激活函数：sigmod、tanh、relu
选择激活函数的经验法则：  
1 如果输出值是0或1，或二分类，sigmod适合作为输出层激活函数，其他单元使用ReLU  
2 tanh几乎在任何场合比较优越  
3 ReLU是激活函数默认选择（吴恩达力推ReLU，原因是斜率和0相差很远，网络学习会很快）  
4 Leaky ReLU与ReLU类似

如何选择激活函数，确定参数？ 除了参考热门应用，应该在开发集上跑跑实验，看哪组参数比较好。

![alt text](image-20.png)


##### 3.7 为什么需要非线性激活函数？（why need a nonlinear activation ##### function?）
如果不使用非线性激活函数，或者说是使用线性激活，隐层是线性的，线性隐层一点用都没有，无论多少层，一直在做的是线性计算，这个模型复杂度和标准的逻辑回归没区别。
![alt text](image-21.png)

只有回归问题才需要线性激活函数，例如房价预测问题，输出层使用线性激活
![alt text](image-22.png)

##### 3.8 激活函数的导数（Derivatives of activation functions）
##### 3.9 神经网络的梯度下降（Gradient descent for neural networks）
##### 3.10（选修）直观理解反向传播（Backpropagation intuition）
##### 3.11 随机初始化（Random+Initialization）