# Numpy学习笔记

[TOC]




## 一、Numpy的介绍

### 1、NumPy简介

NumPy的全名为Numeric Python，是一个开源的Python科学计算库，它包括：
一个强大的N维数组对象ndrray；
比较成熟的（广播）函数库；
用于整合C/C++和Fortran代码的工具包；
实用的线性代数、傅里叶变换和随机数生成函数

### 2、NumPy的优点

- 对于同样的数值计算任务，使用NumPy要比直接编写Python代码便捷得多;
- NumPy中的数组的存储效率和输入输出性能均远远优于Python中等价的基本数据结构，且其能够提升的性能是与数组中的元素成比例的；
- NumPy的大部分代码都是用C语言写的，其底层算法在设计时就有着优异的性能，这使得NumPy比纯Python代码高效得多

### 3、NumPy的缺陷

- 由于NumPy使用内存映射文件以达到最优的数据读写性能，而内存的大小限制了其对TB级大文件的处理；

- NumPy数组的通用性不及Python提供的list容器。

- 因此，在科学计算之外的领域，NumPy的优势也就不那么明显。

  不过使用Numpy处理TB级以下的数据是够用的，但要求计算机内存够处理你的数据。如果你觉得此库不能满足你的数据分析要求，还可以采用如下功能库进行数据分析：

  **1.pandas库**

  >pandas 是一个开源的软件，它具有 BSD 的开源许可，为 Python 编程语言提供高性能，易用数据结构和数据分析工具。在数据改动和数据预处理方面，Python 早已名声显赫，但是在数据分析与建模方面，Python 是个短板。Pandas 软件就填补了这个空白，能让你用 Python 方便地进行你所有数据的处理，而不用转而选择更主流的专业语言，例如 R 语言。

  **2.IPython**

  > IPython 是一个在多种编程语言之间进行交互计算的命令行 shell，最开始是用 python 开发的，提供增强的内省，富媒体，扩展的 shell 语法，tab 补全，丰富的历史等功能。IPython 提供了如下特性:
  >
  > - 更强的交互 shell（基于 Qt 的终端）
  > - 一个基于浏览器的记事本，支持代码，纯文本，数学公式，内置图表和其他富媒体
  > - 支持交互数据可视化和图形界面工具
  > - 灵活，可嵌入解释器加载到任意一个自有工程里
  > - 简单易用，用于并行计算的高性能工具
  

**3.GraphLab Greate**

>GraphLab Greate 是一个 Python 库，由 C++ 引擎支持，可以快速构建大型高性能数据产品。 GraphLab Greate 的特点：
  >可以在您的计算机上以交互的速度分析以 T 为计量单位的数据量。在单一平台上可以分析表格数据、曲线、文字、图像。最新的机器学习算法包括深度学习，进化树和 factorization machines 理论。可以用 Hadoop Yarn 或者 EC2 聚类在你的笔记本或者分布系统上运行同样的代码。借助于灵活的 API 函数专注于任务或者机器学习。在云上用预测服务便捷地配置数据产品。为探索和产品监测创建可视化的数据。

  **4. Scikit-Learn**

  >Scikit-Learn 是一个简单有效地数据挖掘和数据分析工具（库）。关于最值得一提的是，它人人可用，重复用于多种语境。它基于 NumPy，SciPy 和 mathplotlib 等构建。Scikit 采用开源的 BSD 授权协议，同时也可用于商业。Scikit-Learn 具备如下特性:

  > - 分类（Classification） – 识别鉴定一个对象属于哪一类别
> - 回归（Regression） – 预测对象关联的连续值属性
  > - 聚类（Clustering） – 类似对象自动分组集合
  >  - 降维（Dimensionality Reduction） – 减少需要考虑的随机变量数量
  >  - 模型选择（Model Selection） –比较、验证和选择参数和模型
  >  - 预处理（Preprocessing） – 特征提取和规范化





## 二、Numpy 安装

### 1、安装

```
pip install -i https://pypi.douban.com/simple numpy scipy matplotlib
```

## 三、Ndarry对象

> python 默认浅拷贝，即更改元素后对应元素也会发生变化，需要设置Copy = True，才会是深拷贝
>
> **order 中的 C应该是从低维度开始读写，而 F 顺序则是从高维度开始读写。**
>
> Numpy数组可以自定义数据类型，每一个数的类型可以不一样

### 1、构造函数

`numpy.array(object, dtype = None, copy = True, order = None, subok = False, ndmin = 0)`

参数说明：

| 名称   | 描述                                                      |
| :----- | :-------------------------------------------------------- |
| object | 数组或嵌套的数列                                          |
| dtype  | 数组元素的数据类型，可选                                  |
| copy   | 对象是否需要复制，可选                                    |
| order  | 创建数组的样式，C为行方向，F为列方向，A为任意方向（默认） |
| subok  | 默认返回一个与基类类型一致的数组                          |
| ndmin  | 指定生成数组的最小维度                                    |

话虽如此，我们经常用的比较简单：

#### np.array  （默认深拷贝）

``` python
a = np.array([1,2,3,4])
a = np.array((1,2,3,4))
a = np.array((1,2,(1,2,3)))
a = np.array([1,0,1,0,1],dtype = 'bool')
# a = [ True False  True False  True]  a.dtype = bool
a = np.array((1,2),dtype = 'object')
a = np.array([1,2,(1,2,3)],dtype = 'object')
# 列表, 列表的元组, 元组, 元组的元组, 元组的列表，多维数组
# 但最外层的括号会被直接忽略，如上述
# In:  a = np.array([1,2,(1,2,3)],dtype = 'object')
# Out: a = [1 2 (1, 2, 3)]
# 	   a.dtype = object
```

#### np.asarray  （默认浅拷贝，若order与默认不同，则深拷贝）
``` python
a	任意形式的输入参数，可以是，列表, 列表的元组, 元组, 元组的元组, 元组的列表，多维数组
dtype	数据类型，可选
order	可选，有"C"和"F"两个选项,分别代表，行优先和列优先，在计算机内存中的存储元素的顺序。
```
``` python
import numpy as np# 默认底数是 10

print(np.arange(0,10).flags)
a = np.arange(0,10).reshape(2,5)
b = np.asarray(a,order = 'F')

a[1][0] = 101
print(a)
print(a.flags)

print(b)
print(b.flags)
```

>```
>  C_CONTIGUOUS : True
>  F_CONTIGUOUS : True
>  OWNDATA : True
>  WRITEABLE : True
>  ALIGNED : True
>  WRITEBACKIFCOPY : False
>  UPDATEIFCOPY : False
>
>[[  0   1   2   3   4]
> [101   6   7   8   9]]
>  C_CONTIGUOUS : True
>  F_CONTIGUOUS : False
>  OWNDATA : False
>  WRITEABLE : True
>  ALIGNED : True
>  WRITEBACKIFCOPY : False
>  UPDATEIFCOPY : False
>
>[[0 1 2 3 4]
> [5 6 7 8 9]]
>  C_CONTIGUOUS : False
>  F_CONTIGUOUS : True
>  OWNDATA : True
>  WRITEABLE : True
>  ALIGNED : True
>  WRITEBACKIFCOPY : False
>  UPDATEIFCOPY : False
>```



### 2、属性

| 属性             | 说明                                                         |
| :--------------- | :----------------------------------------------------------- |
| ndarray.ndim     | 秩，即轴的数量或维度的数量                                   |
| ndarray.shape    | 数组的维度，对于矩阵，n 行 m 列                              |
| ndarray.size     | 数组元素的总个数，相当于 .shape 中 n*m 的值                  |
| ndarray.dtype    | ndarray 对象的元素类型                                       |
| ndarray.itemsize | ndarray 对象中每个元素的大小，以字节为单位                   |
| ndarray.flags    | ndarray 对象的内存信息                                       |
| ndarray.real     | ndarray元素的实部                                            |
| ndarray.imag     | ndarray 元素的虚部                                           |
| ndarray.data     | 包含实际数组元素的缓冲区，由于一般通过数组的索引获取元素，所以通常不需要使用这个属性。 |



``` python
ndim #数组的维数
shape #数组的形状，（24，）表示一个一维，（24，1）表示二维，24行1列，（1，24）表示二维，1行，24列
flags #数组的属性，好东西
size #总个数
dtype #类型
itemsize #元素大小
```

#### a.shape

``` python
a = np.arange(0, 25)
a.shape = (5, 5)

# 相当于 a = np.arange(0,25).reshape(5,5)
```



#### a.reshape

ndarray.reshape 通常返回的是非拷贝副本，即改变返回后数组的元素，原数组对应元素的值也会改变。

```
In [1]: import numpy as np
In [2]: a=np.array([[1,2,3],[4,5,6]])

In [3]: a
Out[3]:
array([[1, 2, 3],
    [4, 5, 6]])

In [4]: b=a.reshape((6,))
In [5]: b
Out[5]: array([1, 2, 3, 4, 5, 6])

In [6]: b[0]=100
In [7]: b
Out[7]: array([100,   2,   3,   4,   5,   6])

In [8]: a
Out[8]:
array([[100,   2,   3],
    [  4,   5,   6]])
```



#### a.flatten

> a = a.flatten()

降维打击，坍塌成一维，默认**深拷贝**



### 3、生成数列的构造函数

#### np.arange

```
numpy.arange(start, stop, step, dtype)

生成的是一维的
参数	描述
start	起始值，默认为0
stop	终止值（不包含）
step	步长，默认为1
dtype	返回ndarray的数据类型，如果没有提供，则会使用输入数据的类型。
```



``` python
a = np.arange(5) #[0  1  2  3  4]
x = np.arange(5, dtype =  float) #[0.  1.  2.  3.  4.]
x = np.arange(10,20,2)  #[10  12  14  16  18]
```

#### np.linspace（等差）

numpy.linspace 函数用于创建一个一维数组，数组是一个等差数列构成的，格式如下：

```
np.linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None)

参数	描述
start		序列的起始值
stop		序列的终止值，如果endpoint为true，该值包含于数列中
num			要生成的等步长的样本数量，默认为50
endpoint	该值为 true 时，数列中包含stop值，反之不包含，默认是True。 
retstep		如果为 True 时，生成的数组中会显示间距，反之不显示。  True时会生成一个元组，第一维度是数组，第二维是间距
dtype		ndarray 的数据类型
以下实例用到三个参数，设置起始点为 1 ，终止点为 10，数列个数为 10。
```
以下实例用到三个参数，设置起始点为 1 ，终止点为 10，数列个数为 10。 

``` python
#设置起始点为 1 ，终止点为 10，数列个数为 10。
a = np.linspace(1,10,10)
a = np.linspace(start = 1,stop = 10,num=10)

#上述等价
#[ 1.  2.  3.  4.  5.  6.  7.  8.  9. 10.]
#float64

a =np.linspace(1,10,10,retstep= True)
#(array([ 1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10.]), 1.0)

```

#### np.logspace（等比）

```
np.logspace(start, stop, num=50, endpoint=True, base=10.0, dtype=None) 

参数	描述
start		序列的起始值为：base ** start
stop		序列的终止值为：base ** stop。如果endpoint为true，该值包含于数列中
num			要生成的等步长的样本数量，默认为50
endpoint	该值为 true 时，数列中中包含stop值，反之不包含，默认是True。
base		对数 log 的底数。
dtype		ndarray 的数据类型
```

> np.logspace(a,b,n)创建行向量，第一个是10^a^ ，最后一个10^b^，形成总数为n个元素的等比数列。

``` python
a = np.logspace(1.0,  2.0, num =  10)  
#[ 10.           12.91549665     16.68100537      21.5443469  27.82559402      
# 35.93813664   46.41588834     59.94842503      77.42636827    100.    ]

a = np.logspace(0,9,10,base=2)
#将对数的底数设置为 2 :
#[  1.   2.   4.   8.  16.  32.  64. 128. 256. 512.]
```



### 4、生成形状的构造函数

#### np.empty

``` python
numpy.empty(shape, dtype = float, order = 'C')
默认浮点数

参数	描述
shape	数组形状
dtype	数据类型，可选
order	有"C"和"F"两个选项,分别代表，行优先和列优先，在计算机内存中的存储元素的顺序。

x = np.empty([3,2], dtype = int) 

[[ 6917529027641081856  5764616291768666155]
 [ 6917529027641081859 -5764598754299804209]
 [          4497473538      844429428932120]]
```



#### np.zeros

``` python
numpy.zeros(shape, dtype = float, order = 'C')
默认浮点数

shape	数组形状
dtype	数据类型，可选
order	'C' 用于 C 的行数组，或者 'F' 用于 FORTRAN 的列数组

x = np.zeros([3,2], dtype = int) 
x = np.zeros((3,2), dtype = int) #均可
[[0 0]
 [0 0]
 [0 0]]
```

#### np.ones

``` python
numpy.ones(shape, dtype = None, order = 'C')
默认1，整形

shape	数组形状
dtype	数据类型，可选
order	'C' 用于 C 的行数组，或者 'F' 用于 FORTRAN 的列数组

x = np.ones([3,2], dtype = int) 
x = np.ones((3,2), dtype = int) #均可
[[1 1]
 [1 1]
 [1 1]]
```

#### np.random.rand  （均匀）

0~1均匀分布小数

``` python
np.random.rand(3)
np.random.rand(2,3)#生成shape是2，3
np.random.rand(2,3,4)#生成shape是2，3，4
……
```

#### np.random.randn （正态）

标准正态分布小数 ，是以0为均值、以1为标准差的正态分布，记为N（0，1）。

``` python
np.random.randn(3)
np.random.randn(2,3)#生成shape是2，3
np.random.randn(2,3,4)#生成shape是2，3，4
……
```

#### numpy.random.randint

- 返回随机整数，范围区间为[low,high），包含low，不包含high

``` python
np.random.randint(1,5) # 返回1个[1,5)的随机整数
np.random.randint(1,5,size=(2,3)) # 返回shape(2,3)的[1,5)的随机整数
```



### 5、广播

广播机制用来进行两个Ndarry数组的计算。

简单来说，若`a = np.random.rand(3,4,5,6,7)`

那么，它可以和如下的进行计算：

``` python
a = np.random.rand(3,4,5,6,7)

b = np.random.rand(3,4,5,6,7) 	#维数一模一样
b = np.random.rand(7)
b = np.random.rand(6,7)
b = np.random.rand(5,6,7)	 	#后面往前一模一样，前面省略
b = np.random.rand(5,6,1)	 	
b = np.random.rand(5,1,1) 		#前面省略，后面往前面对应位存在一个1（无论是a还是b），或维度相同

以下的不行：
b = np.random.rand(4,1,1) 		#从低到高数，第三维维度不一致
```



### 6、切片和索引

#### 一维数组

这里切的片是原数组的值，而不是索引值



这里有一个`::`运算符 其实就是冒号运算符省略了中间那个数

``` python
print(x[::-1])#反向打印数据，从最后一个index开始，间隔为1
print(x[::-3])#反向打印数据，从最后一个index开始，间隔为3
print(x[7:2:-1])#反向打印index=2(不包含)到index=7之间的数据
```



``` python
a = np.arange(10)

s = slice(2,7,2)   # 从索引 2 开始到索引 7 停止，间隔为2
print (a[s])
#[2  4  6]

a[:]	#从头到尾所有值
#[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

a[2:] 	#从索引2开始到最后
#[2, 3, 4, 5, 6, 7, 8, 9]

a[:2]	#从索引0开始到索引2（不包含）
#[0, 1]

a[2:7] 	#从索引0开始到索引7,步长默认1（不包含）
#[2, 3, 4, 5, 6]

a[2:7:2] 	# 从索引 2 开始到索引 7 停止，间隔为2
#[2  4  6]

a[[1,3,5,7]]		#
a.take([1,3,5,7])	#二者均是取1357对应的值作为列表
#[1, 3, 5, 7]
```

#### 二维数组

取某一列，某一行，返回值是**一维列表**：

```python
a = np.array([[1,2,3],
              [3,4,5],
              [4,5,6]])  
print (a[...,1])   # 第2列元素
print (a[:  ,1])   # 第2列元素

print (a[1,...])   # 第2行元素
print (a[1,:  ])   # 第2行元素

print (a[...,1:])  # 第2列及剩下的所有元素(返回二位列表)

#[2 4 5]
#[2 4 5]

#[3 4 5]
#[3 4 5]

#[[2 3]
# [4 5]
# [5 6]]
```

以行为轴，取若干行，返回值是**二维列表**：

``` python
a = np.array([[1,2,3],
              [3,4,5],
              [4,5,6]])  
a[1:]	#取第一行以及以后全部
a[1:2]	#取第一行到第二行（不包含）
a[1:3]	#取第一行到第三行（不包含）
```

以列为轴，取若干列，返回值是**二维列表**：

``` python
a = np.array([[1,2,3],
              [3,4,5],
              [4,5,6]])  
a[:,1:]		#取第一列以及以后全部
a[:,1:2]	#取第一列到第二列（不包含）
a[:,1:3]	#取第一列到第三列（不包含）
```



切片，行列都切（连续），返回值**二维矩阵**：

``` python
a=np.arange(0,12).reshape(3,4)

print(a)		
print(a[0:2,1:3])	#从0~2行（不含），1~3列（不含），取矩阵

#[[ 0  1  2  3]
# [ 4  5  6  7]
# [ 8  9 10 11]]
#[[1 2]
# [5 6]]
```



切片，行列都切（不连续），返回值**二维矩阵**：

注：原理与上面连续矩阵切片一样，只不过分开写容易归纳

这里可以**传列表作为每一维的参数**，若列表取值为离散就可以实现（不连续）。

``` python
a=np.arange(0,80).reshape(8,10)

#若行列有一维是冒号表达式，则另一维用列表即可
a[range(5),1:3]			#取0~5行(不包括)，1~3列（不包括）
a[[1,3,5,7],1:3]		#取1357行，1~3列
```

``` python
a=np.arange(0,80).reshape(8,10)
#若行列都离散，需要这样来算：
r = [1,3,5,7] # r = np.array([1,3,5,7]) 均可
c = [2,4,6,8,9] # c = np.array([2,4,6,8]) 均可
a[r,:][:,c]	#其中冒号可以换成...
a[np.ix_(r,c)]# 利用笛卡尔积
#out:
#array([[12, 14, 16, 18, 19],
#       [32, 34, 36, 38, 39],
#       [52, 54, 56, 58, 59],
#       [72, 74, 76, 78, 79]])
```





#### 高维数组

仿二维

``` python
a=np.arange(0,160).reshape(8,10,2)
r = [1,3,5,7]  		# r = np.array([1,3,5,7]) 均可
c = [2,4,6,8,9] 	# c = np.array([2,4,6,8,9]) 均可
h = [1] 			# c = np.array([1]) 均可

a[r,...][:,c,:][...,h]
a[np.ix_(r,c,h)]	#笛卡尔积
```



### 7、高级索引

#### 数组索引

``` python
x = np.array([[0,1,2],[3,4,5],[6,7,8],[9,10,11]])  
rows = np.array([0,0,3,3]) 
cols = np.array([0,2,0,2]) 
y = x[rows,cols]  #这样会坍塌成一维
#[ 0 2 9 11]

rows = np.array([[0,0],[3,3]]) 
cols = np.array([[0,2],[0,2]]) 
y = x[rows,cols]  #这样就是二维了
#[[ 0  2]
# [ 9 11]]

#笛卡尔积

```

#### 笛卡尔积

``` python
a=np.arange(0,80).reshape(8,10)
#若行列都离散，需要这样来算：
r = [1,3,5,7] # r = np.array([1,3,5,7]) 均可
c = [2,4,6,8,9] # c = np.array([2,4,6,8]) 均可
a[r,:][:,c]	#其中冒号可以换成...
a[np.ix_(r,c)]# 利用笛卡尔积
#out:
#array([[12, 14, 16, 18, 19],
#       [32, 34, 36, 38, 39],
#       [52, 54, 56, 58, 59],
#       [72, 74, 76, 78, 79]])
```

#### 布尔索引

通过和原数组等大的布尔数组 索引目标数组，坍塌成一维

``` python
x = np.array([[0,1,2],[3,4,5],[6,7,8],[9,10,11]])  
print ('我们的数组是：')
print (x)
print ('\n')
# 现在我们会打印出大于 5 的元素  
print  ('大于5且小于7的元素是：')
print (x[(x>5) & (x<7)])
#[ 0  1  2  3  4  5  6  7  8  9 10 11]
x[x<10] = 3  #将x<10的数赋值成3 
x[~np.isnan(x)] = 3 #将非NaN的数字赋值成3
```



### 8、迭代器

``` python
a = np.arange(6).reshape(2,3)
for x in np.nditer(a):	#最朴素
    print (x, end=", " )
for x in np.nditer(a, order =  'C'):  #行优先
    print (x, end=", " )
for x in np.nditer(a, order =  'F'):  #列优先
    print (x, end=", " )


for x in np.nditer(a, op_flags=['readwrite']): 
	x[...]=2*x 
#注意，迭代器默认readonly 即不可以修改只可以读取
#x[...] 是修改原 numpy 元素，x 只是个拷贝。
```



## 四、数组操作

### 1、修改数组形状

| 函数      | 描述                                               |
| :-------- | :------------------------------------------------- |
| `reshape` | 不改变数据的条件下修改形状                         |
| `flat`    | 数组元素迭代器                                     |
| `flatten` | 返回一份数组拷贝，对拷贝所做的修改不会影响原始数组 |
| `ravel`   | 返回展开数组                                       |

### 2、翻转数组（转置）

| 函数        | 描述                       |
| :---------- | :------------------------- |
| `transpose` | 对换数组的维度             |
| `ndarray.T` | 和 `self.transpose()` 相同 |
| `rollaxis`  | 向后滚动指定的轴           |
| `swapaxes`  | 对换数组的两个轴           |

``` python
a = a.transpose()
a = a.T
```





### 3、修改数组维度

| 维度           | 描述                       |
| :------------- | :------------------------- |
| `broadcast`    | 产生模仿广播的对象         |
| `broadcast_to` | 将数组广播到新形状         |
| `expand_dims`  | 扩展数组的形状             |
| `squeeze`      | 从数组的形状中删除一维条目 |

### 4、连接数组(np.xx)

| 函数          | 描述                           |
| :------------ | :----------------------------- |
| `concatenate` | 连接沿现有轴的数组序列         |
| `stack`       | 沿着**新的轴**加入一系列数组。 |
| `hstack`      | 水平堆叠序列中的数组（列方向） |
| `vstack`      | 竖直堆叠序列中的数组（行方向） |

``` python
a = np.array([[1,2],[3,4]])
b = np.array([[5,6],[7,8]])
print (np.concatenate((a,b)))
print (np.concatenate((a,b),axis = 1))
#沿轴 0 连接两个数组：
#[[1 2]
# [3 4]
# [5 6]
# [7 8]]
#沿轴 1 连接两个数组：
#[[1 2 5 6]
# [3 4 7 8]]
```



``` python
a = np.array([[1,2],[3,4]])
b = np.array([[5,6],[7,8]])
c = np.hstack((a,b))

水平堆叠：往水平方向堆叠，也就是从每一行的末尾添加一点东西
[[1 2 5 6]
 [3 4 7 8]]
```

``` python
a = np.array([[1,2],[3,4]])
b = np.array([[5,6],[7,8]])
c = np.vstack((a,b))
竖直堆叠：
[[1 2]
 [3 4]
 [5 6]
 [7 8]]
```



### 5、分割数组(np.xx)

| 函数     | 数组及操作                             |
| :------- | :------------------------------------- |
| `split`  | 将一个数组分割为多个子数组             |
| `hsplit` | 将一个数组水平分割为多个子数组（按列） |
| `vsplit` | 将一个数组垂直分割为多个子数组（按行） |

### 6、数组元素的添加与删除(np.xx)

| 函数     | 元素及描述                               |
| :------- | :--------------------------------------- |
| `resize` | 返回指定形状的新数组                     |
| `append` | 将值添加到数组末尾                       |
| `insert` | 沿指定轴将值插入到指定下标之前           |
| `delete` | 删掉某个轴的子数组，并返回删除后的新数组 |
| `unique` | 查找数组内的唯一元素                     |

numpy.unique 函数用于去除数组中的重复元素。

```
numpy.unique(arr, return_index, return_inverse, return_counts)
```

- `arr`：输入数组，如果不是一维数组则会展开
- `return_index`：如果为`true`，返回新列表元素在旧列表中的位置（下标），并以列表形式储
- `return_inverse`：如果为`true`，返回旧列表元素在新列表中的位置（下标），并以列表形式储
- `return_counts`：如果为`true`，返回去重数组中的元素在原数组中的出现次数

``` python
u = np.unique(a)
print (u)
```



## 五、常用函数

### 字符串

``` python
np.char.add(['hello'],[' xyz'])	#连接
np.char.multiply('Runoob',3)	#复制若干份
np.char.lower(['RUNOOB','GOOGLE'])#小写
np.char.upper(['runoob','google'])#大写
np.char.split ('www.runoob.com', sep = '.') #分隔符默认空格，这里是.
numpy.char.splitlines() #函数以换行符作为分隔符来分割字符串，并返回数组。

# 移除字符串头尾的 a 字符
print (np.char.strip('ashok arunooba','a'))
# 移除数组元素头尾的 a 字符
print (np.char.strip(['arunooba','admin','java'],'a'))

a = np.char.encode('runoob', 'cp500') 
np.char.decode(a,'cp500')

print (np.char.join([':','-'],['runoob','google']))
#['r:u:n:o:o:b' 'g-o-o-g-l-e']

np.char.replace ('i like runoob', 'oo', 'cc') #替换，还挺有意思的
```

### 数学

``` python
# a = np.array([0,30,45,60,90])
np.sin(a*np.pi/180)
np.cos(a*np.pi/180)
np.tan(a*np.pi/180)

np.around(a, decimals =  1)
np.around(a, decimals =  -1)#将舍入到整数部分

np.floor(a)	#下取整
np.ceil(a)	#上取整
```



### 算术

``` python
#维度相同 或者 符合广播原则
np.add(a,b)
np.subtract(a,b)
np.multiply(a,b)
np.divide(a,b)
np.reciprocal(a) #倒数
np.power(a,b) #相当于快速幂
np.mod(a,b)		 #模
np.remainder(a,b)#模
```



### 统计

``` python
np.amin(a,1)	#一行行取最小值，返回list
np.amin(a,0)	#一列列取最小值，返回list
np.ptp(a, axis =  1) #行极差
np.ptp(a, axis =  0) #列极差
np.percentile(a, 50，axis=1) #从小到大行排名第50%那个数 %0是最小的
np.percentile(a, 50，axis=0) #从小到大列排名第50%那个数
np.median(a, axis =  1) #每行中位数
np.median(a, axis =  0)	#每列中位数
np.mean(a, axis =  ？)#算数平均数
np.average(a,weights = [], axis = ？) #加权平均数，3
np.std(a, axis =  ？) #标准差
np.var(a, axis =  ？) #方差
```



### 排序、条件筛选

```
np.sort() 函数返回输入数组的排序副本。

np.argsort() 函数返回的是数组值从小到大的索引值。

np.lexsort() 多关键字排序

np.argmax() 和 numpy.argmin()函数分别沿给定轴返回最大和最小元素的索引。

np.nonzero() 函数返回输入数组中非零元素的索引。

np.where(con) 函数返回输入数组中满足给定条件的元素的索引。
满足con 的值赋值成x，否则赋值成y

con = x<5	#返回一个x<5的bool数组 赋值给 con
np.where(con,x,y) 返回和原数组一样大小的数组，若值满足con，则赋值x，否则赋值y



np.extract(condition, x)通过布尔数组condition，选择x对应的值，布尔数组和x shape一致
```



### 视图、浅拷贝、深拷贝

**视图一般发生在：**

- 1、numpy 的切片操作返回原数据的视图。
- 2、调用 ndarray 的 view() 函数产生一个视图。

**副本一般发生在：**

- Python 序列的切片操作，调用deepCopy()函数。
- 调用 ndarray 的 copy() 函数产生一个副本。

**无复制**

简单的赋值不会创建数组对象的副本。 相反，它使用原始数组的相同id()来访问它。 id()返回 Python 对象的通用标识符，类似于 C 中的指针。

此外，一个数组的任何变化都反映在另一个数组上。 例如，一个数组的形状改变也会改变另一个数组的形状。



## 六、矩阵

NumPy库里面有个矩阵库：numpy.matlib

``` python
.T #转置
numpy.matlib.empty(shape) #随机填充
numpy.matlib.zeros(shape) #0矩阵
numpy.matlib.ones(shape) #1矩阵
np.matlib.eye(n =  3, M =  4, k =  0, dtype =  float) #像单位矩阵的矩阵，对角1，其他零，M默认与n相等
np.matlib.identity(5, dtype =  float) #返回大小为5的单位矩阵
np.matlib.rand(3,3)	#返回3*3的矩阵，随机填充

k = np.asmatrix (j)  #将j转变为矩阵k。

```



| 函数          | 描述                             |
| :------------ | :------------------------------- |
| `dot`         | 两个数组的点积，即元素对应相乘。 |
| `vdot`        | 两个向量的点积                   |
| `inner`       | 两个数组的内积                   |
| `matmul`      | 两个数组的矩阵积                 |
| `determinant` | 数组的行列式                     |
| `solve`       | 求解线性矩阵方程                 |
| `inv`         | 计算矩阵的乘法逆矩阵             |



## 保存读取txt，csv，tsv

```
np.loadtxt(FILENAME, dtype=int, delimiter=' ')
np.savetxt("out.txt",a,fmt="%d",delimiter=",")
b = np.loadtxt("out.txt",delimiter=",") # load 时也要指定为逗号分隔
```

