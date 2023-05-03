# easy_diffusion
*easy diffusion的意思是easy to train and modify arch的diffusion模型，是我在通读diffusion系列的论文和理解相应的源码之后，自己实现的一个文生动图和文生”视频“的模型。我做这个项目的原因有两个：第一是公司的业务有一个文生图的需求，业务特点是中文垂直领域的文生图，所以stable diffusion和latent diffusion预训练好的大模型在我们公司需求领域生成效果不大理想，所以需要在垂直中文领域进行定制化的文生图训练；第二是目前开源的diffusion模型代码实现的非常好了，但是对我来说有一个很大的不便（可能也是很多人的不便），就是想自己做一些定制化的修改比较费劲，因为那些代码基本上把model架构和流程都通过config文件配置好了，所以想做一些自己的修改会比较麻烦，所以我才总结和设计了这个，完全由基础的dataset， dataloader， model arch，train epoch组成的代码架构。*


## Requirements
pytorch, cn_clip, sd-v1-5-inpainting
三个模块：
    cn_clip：将中文菜品名编码成embedding，在我的项目里我用的是微调过后的cn_clip
    easy_diffusion: diffusion模块，做条件和非条件生成
    stable_diffusion: inpainting模块，做图像补充

## How To Train and Sample
实现的流程代码都在easy_diffusion/ipynb的jupyter文件里面，包含有数据预处理，模型训练，模型推理等。


## 可能遇到的问题和解决方案
1.训练几个step之后loss为nan和学习率与优化器的选择
从头开始训练diffusion模型对学习率和优化器非常敏感，当优化器为SGD的时候，由于从头训练学习率设置非常难（我没用成功过）；当优化器为AdamW的时候，学习率太大会几轮过后直接变成nan，经过我多次的实验发现设置为0.0001可以达到不错的收敛速度和效率，但是会在一定小的loss之间震荡；所以我采用的方案是先用AdamW+0.0001训练到loss震荡，再用SGD+0.000001做最终的收敛和微调。

2.训练diffusion是一件及其耗费显存的模型，因为他的UNet模块中存在很多大channel1x1卷积操作，和cross attention和self attention，这三个操作非常的吃显存，我将模型参数调整到了370MB左右，在V100 32GB显存的机器上训练分辨率3x128x128的图片也基本上只能24 batch size（没有加入vlb loss,加入vlb loss的情况下batch size只能16）

3.在潜变量空间训练模型，多线程的问题。
这里分成两种方案，第一种：AutoEncoder单独训练，保存它对图像的转换结果，这就无形中增大了存储空间的要求，特别是当原来的数据集就很大的时候，这样就更加加重了存储空间的需求，如果你的机器各方面配置很充足，我推荐这种；
第二种：训练完成AutoEncoder之后，直接利用AutoEncoder作为预处理器和diffusion模型一起训练，使用dataloader多线程在GPU上训练的时候会出现Cuda初始化的错误，这里需要在dataloader上加上下面这句代码。


**文生动图的示例图**

![土鸡蛋炒番茄_镜头拉远](easy_diffusion/image/土鸡蛋炒番茄_镜头拉远2.gif)
![土鸡蛋炒番茄_镜头拉远](easy_diffusion/image/土鸡蛋炒番茄_镜头拉远.gif)
![米饭_镜头右移](easy_diffusion/image/米饭_镜头右移.gif)
![葱油生菜_镜头右上45度移](easy_diffusion/image/葱油生菜_镜头右上45度移.gif)
