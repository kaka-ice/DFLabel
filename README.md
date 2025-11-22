置顶软件下载链接：[https://github.com/kaka-ice/DFLabel](https://github.com/kaka-ice/DFLabel
)

# 1.DFLabel简介

DFLabel是一款专门为YOLO(Ultralytics)算法而生的标注软件，包括水平矩形标注、多边形标注、关键点标注以及旋转矩形标注，如下所示。

![](https://secure2.wostatic.cn/static/7NjjKbvuQG52fNFpeL5Bax/image.png?auth_key=1763788275-i6g2J5nfAuZtHQWWdyjWG8-0-d682ffabfb40a92047bb10e1b813bbbc)

DFLabel的主要亮点：

（1）众所周知，分割标注的工作量是最大的，所以DFLabel除了提供了手动标注模式，还特地增加了基于Sam的智能标注模式。

（2）除了标注的功能，当然还包括直接读取YOLO专属的txt标签并可视化的功能，这样减少了我们使用脚本代码可视化txt标签的工作量；其中实例分割txt标签的可视化是我根据ultralytics处理分割标签的方式来实现的；总之，可视化功能让我们可以更方便地去检查标签，这个功能在平时工作中是非常重要的。

（3）当然，考虑到让DFLabel标注软件能够更灵活，DFLabel标注后生成的json标签是完全兼容labelme的json格式的，

当然，DFLabel这个软件不可能那么尽善尽美，只能在以后使用过程中慢慢去优化迭代了。

在正式介绍软件如何使用之前，请允许我夹带私货介绍下标注软件的图标和名字，DF源于我柴的首字母缩写，软件的图标也是我柴的剪影，希望DFLabel能够受到大家的欢迎。

![](https://secure2.wostatic.cn/static/vcN6wa1zGpEcccygczbHwx/image.png?auth_key=1763788275-wagnhdKY5G8ctLBpqoYFUA-0-2e898b63f1135a1a717314abbc249ed9)

# 2.标注功能

## 2.1界面及基本操作

整个界面很简单，和LabelImg/Labelme等标注软件的界面长得都差不多，如下所示。

![](https://secure2.wostatic.cn/static/hL2XZSdJwL44yoRdPvTJxM/image.png?auth_key=1763788275-ok5rrNnQVuDEpccyMcWDqE-0-31b757f02ac767aeec6b3e5524c57cc1)

下面是操作介绍：

**1.导入图片: **

**“文件—>加载图片”**

**2.添加类别:**

**“添加—>输入类别—>OK”**

**3.编辑类别：**

**“双击类别—>修改类别—>选择类别—>OK”**

![](https://secure2.wostatic.cn/static/eMrJjQVToJgnwcpu6y7GdT/image.png?auth_key=1763788275-sBdEmaJGfvHkiwyZWeyyCS-0-9ddb99d882d30a053b499d9c59ac45c4)

**4.删除类别：**

**“选中类别—>删除”**

注意：对于已标注过的类别会有提示说明无法进行删除。

**5.切换图片：**

**A键：上一张；**

**D键：下一张；**

**6.图片显示**

**鼠标中键：平移视图**

**鼠标滚轮：缩放视图**

## 2.2标注操作

矩形标注、多边形标注、点标注以及旋转框标注可以同时在同一张图片中进行标注，这个没有进行一个互斥限制，如下图中标注了四种类型的标签：

![](https://secure2.wostatic.cn/static/qLpskdoMnjKySR5KSuejuJ/image.png?auth_key=1763788275-kvCa8V7BiGdPKcPDE4y1LM-0-a7154ac9b47548923fc279ce358cb8a0)

生成的json如下所示，水平框（rectangle)保存的点坐标是左上角点和右下角点，旋转框(rotatedRect)保存的是四个顶点的坐标以及旋转角度，多边形（polygon)保存的是多边形的XY坐标，点（points)保存的是点的XY坐标。

这里需要注意一个description，我把每个实例的ID和标签的颜色保存在description中，其中实例ID是每个标签都对应于唯一确定的一个ID，方便我用来删除和管理标签用的；还有一个groupID是组ID，用于实例分割标签和关键点任务的标签中，这个groupID并不是唯一的，是可以取相同的。

![](https://secure2.wostatic.cn/static/2Z7s4QANHaDAY8fMxeuYLC/image.png?auth_key=1763788275-s5iBX24oper2BfaLfeZ3hr-0-9fe13ca39f1575b90f44f17f42de560d)

**1.矩形标注：**

**（1）新建标签**

  **选中"矩形标注"—>选中类别—>长按鼠标左键—>释放鼠标左键—>标注完成**

**（2）缩放标签**

  **点击矩形框—>选中白色控制柄—>拖动缩放**

**（3）移动标签**

  **点击矩形框—>移动标签**

**（4）删除标签**

  **点击矩形框—>Backspace键**

  或者直接在**标签控制界面中，选中标签—>鼠标右键—>删除标注**

**（5）编辑标签**

  **标签控制界面中，选中标签—>鼠标右键—>修改类别与GroupID**

  ![](https://secure2.wostatic.cn/static/o5mZeDCbVuEabmwhKXGPCA/image.png?auth_key=1763788275-hoJcDkz3hShrGCWqtzP5sk-0-53ff6effc348909fa3e43b7fa9d50d86)

  然后根据实际情况修改就行，最后点击确认。



**2.多边形标注：**

基本操作和矩形标注是一样的，这里就提一下多边形标注中和矩形标注不一样的，后面的点和旋转框标注也是。

**（1）新建标签**

  **选中"多边形标注"—>选中类别—>鼠标左键添加多边形顶点（循环）—>鼠标右键结束标注—>标注完成**

**（2）标注中途的撤回操作**

  **在鼠标左键添加多边形顶点（循环）**中，想要撤回点，可以通过**快捷键CTRL+Z** 进行撤回。

**（3）取消标注**

  **在鼠标左键添加多边形顶点（循环）**中，想要取消本次标注，可以通过**快捷键ESC**进行取消。



**3.点标注：**

**（1）新建标签**

  **选中"点标注"—>选中类别—>鼠标左键添加点—>标注完成**

  

  关键点标注这块比较特殊，包含点的类别和框的类别，所以这块我并没有直接生成可用于训练的YOLO标签，可以通过脚本代码把生成的json标签做一个转换后再训练。



**4.旋转框标注：**

**（1）新建标签**

  **选中"旋转框标注"—>选中类别—>****长按鼠标左键—>释放鼠标左键—>标注完成**

  调整角度通过以下快捷键：

  **Z键：逆时针旋转5度；**

  **X键：逆时针旋转1度；**

  **C键：顺时针旋转1度；**

  **V键：顺时针旋转5度；**

  

  # 3.Sam智能标注

  首先得确保\deploy\sam_dependencies路径下有sam_vit_b_01ec64.pth权重，然后选择**“SAM模式—>设置SAM环境—>选择Python解释器路径”**

  **注意：**其实就是torch+torchvision环境，如果你有YOLO的虚拟环境，那么你就可以不用重新配置环境，直接选中YOLO虚拟环境路径下的python.exe就行，比如我的路径在H:\anaconda3\envs\v8_env\python.exe.

  ![](https://secure2.wostatic.cn/static/oHtE5jTZ5tPExExAq6LkDG/image.png?auth_key=1763788999-cAuY8tKuJZyR9hYr8VrzUF-0-d46f399ccbfd39bf5058140c38f01688)

  如果你之前没有配置有YOLO环境，可以参考我下面这篇博客来配置一下环境。

  [软件安装及YOLOv8环境配置及验证_yolov8 环境验证-CSDN博客](https://blog.csdn.net/weixin_45144684/article/details/138453506)

**1.SAM标注：**

**（1）新建标签**

  **选中"SAM标注"/按键Q—>选中类别—>鼠标左键/右键—>显示结果—>按键E结束标注**

  其中鼠标左键表示感兴趣提示点(绿色），鼠标右键表示非感兴趣提示点（红色）。

  **注意：**我这里是直接显示和保存的都是最大的轮廓点，如下图所示，所视即所得。

  ![](https://secure2.wostatic.cn/static/6B1PYatpNRR63GJAcft986/image.png?auth_key=1763789628-49AFPNTdAvkxNCQtxbbgpT-0-a3c12aa7226370b3226553e88ae7d46d)

**（2）撤回操作**

  **在鼠标左键/右键添加prompt点进行提示时**，想要撤回当前点，可以通过**快捷键CTRL+Z** 进行撤回。

  

  特别说明，在第一次调用SAM模型的时候会启动一个线程会感觉到一点卡顿，后面的话就还好，我用的4060显卡，会占用大概3G显存。

# 4.可视化功能

对YOLO的txt标签进行可视化，只需要添加一个classes.txt就行，这个和LabelImg标注软件是保持一样的，软件也会进行提示，只需要在图片路径下添加classes.txt，并键入类别就行了。

![](https://secure2.wostatic.cn/static/s4goGjHZQTDWTSnAoBEHgm/image.png?auth_key=1763790223-tM4EEsgY9QRMimL4p97om4-0-fecd8792b9069aabfe5963f6510791b4)

![](https://secure2.wostatic.cn/static/cdmrgedRo6MYzUQTADXn86/image.png?auth_key=1763790266-92GZvZoXimacb6TXZWdqJg-0-625198771fe23913a1a57ee053d41065)



对于实例分割的标签，可以按“Space"空格键进行可视化，有时候对于类似于同心圆的项目，这种可视化可以查看标注是否正确。

![](https://secure2.wostatic.cn/static/fR9fc4tuivJ3QBNxjP3EhB/image.png?auth_key=1763790636-czurRpZZwrcoxSW5t4yVmZ-0-e4aadd7cc996cf3ec594ee80bbf5563c)

![](https://secure2.wostatic.cn/static/po61vkbdPRa9GpbSVLomqw/image.png?auth_key=1763790656-s4Lgi6RoarrAKmZPVCSjv5-0-133141af2d29cd06e6760ebc56021051)

# 5.参考资料

在DFLabel的开发过程中，我参考了以下几款优秀的开源标注工具，在此向这些项目的所有贡献者表示诚挚的感谢，它们的成果为我提供了宝贵的思路和参考。

|**标注工具**|**GitHub地址**|标注任务|
|-|-|-|
|**ISAT with SAM**|[https://github.com/yatengLG/ISAT_with_segment_anything](https://github.com/yatengLG/ISAT_with_segment_anything)|分割|
|**LabelMe**|[https://github.com/wkentaro/labelme](https://github.com/wkentaro/labelme)|分割、目标检测等|
|**LabelImg**|[https://github.com/tzutalin/labelImg](https://github.com/tzutalin/labelImg)|目标检测|
|**RoLabelImg**|[https://github.com/cgvict/roLabelImg](https://github.com/cgvict/roLabelImg)|旋转目标检测|
