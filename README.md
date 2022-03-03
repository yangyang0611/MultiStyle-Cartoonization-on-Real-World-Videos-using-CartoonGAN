# MultiStyle-Cartoonization-on-Real-World-Videos-using-CartoonGAN
**Target: Transform the real world photos into cartoon-style with the inplementation of CartoonGAN**

## Introduction
Cartoon is a drawing, a simple example of creative thinking, and perception of reality and dream. In a few lines, the cartoonist can capture the entire idea he wants to convey, to observe and exaggerate the characters’ features. Many famous cartoon images are created based on real-world scenes. However, drawing high-quality cartoon images will require a tremendous amount of time. Recently, leaning-based style transfer methods on artificial intelligence have drawn considerable attention. Although there are a variety of methods have been developed to create images with flat shading, mimicking cartoon styles. They use either image filtering or formulations in optimization problems. Such methods are too complicated to capture rich artistic styles. Thus in this paper, we propose Fast Style Transfer and GartoonGAN for photo cartoonization. From the perspective of computer vision algorithms, the goal of cartoon stylization is to map images in the photo manifold into the cartoon manifold while keeping the content unchanged. To achieve this goal, we propose to use a dedicated architecture and this will be introduced in Part 3 System Works later. 

## Motivation

After watching the video that turned a real person’s face into cartoon style,we got interested in this technique. The url is https://www.youtube.com/watch?v=5ZKzN8HtOkE&ab_channel=%E5%85%AD%E6%8C%87%E6%B7%B5Huber.  We want to implement a multi style cartoonization on real-world videos and bring value to people who need it.

### **a. Creating a virtual background for MetaVerse**

As we all know, the idea of MetaVerse is getting popular. Our model can help build a virtual world’s background where people can choose which style of universe they want to stay in according to their preference. 

### **b. To help painter reducing their workload**

Traditional animated movies run at  24 frames (think 1 drawing = 1 frame) per second of screen time. That’s 1400 frames per minute and 86400 frames per hour of screen time. This resulted in a huge investment in time and effort.  Thus, creators can rely on this model by transforming the real world picture into their own style, relieving their burden on works.

### **c. To help people creating their own video**

Nowadays, photo shooting has become a common behavior. We can notice there are many applications and photo editing software that have created an amusing filter or AR which can overlay the digital content and information into the physical world. With the help of this model, people may produce their own video and obtain great fascination. 

## System Framework

To achieve our goal of transforming real video into cartoon style, we have implemented 2 methods including Fast Style Transfer and CartoonGAN. We will explain what we have done and make an introduction on models’ framework.

## 3.1 Fast Style Transfer
First, we choose some scene in the Pixar movie as our transformation style target.The training dataset for content is 4000 photos of persons in 2017 coco dataset.

![](https://imgur.com/S7bd2RQ.jpg) <br>
_**Figure 1 images in training set**_

On the basis of the paper of Johnson et al [1], Model’s framework is divided into 2 parts (Fig 2). The first part is image transform network which consists of Deep Residual Network (Fig 3). Input is image to convert X.Output is converted image Y which combines style and content.
![](https://imgur.com/zoKLtZA.jpg) <br>
_**Figure 2 system overview**_

![](https://imgur.com/KfIPtxy.jpg) <br>
_**Figure 3 image transform network**_

The second part is loss network which consists of pretrained VGG-19 model.We replace VGG-16 from the paper.The model has well-trained image filter.We get high level features to calculate loss(fig 4). We calculate content loss on only one layer while calculating style loss on all the layers (Fig 1). 

![](https://imgur.com/uZzE7aJ.jpg) <br>
_**Figure 4 loss function**_

![](https://imgur.com/sQx1Jvo.jpg) <br>
_**Figure 5 content output from the loss network layers**_

![](https://imgur.com/UxBaV8a.jpg) <br>
_**Figure 6 style output from the loss network layers**_

Using a pretrained model saves a lot of time. It takes less than 4 seconds to run the Fast Style Transfer model.

## 3.2  CartoonGAN
CartoonGAN is based on the structure of GAN. GAN consists of two models, namely the generator and the discriminator. In the case of CartoonGAN [3] , the discriminator classifies input images as cartoon or non-cartoon images. On the other hand, the generator takes real-world images as inputs and aims to generate cartoonized outputs that are able to fool the discriminator.

![](https://imgur.com/WWB50iZ.jpg) <br>
_**Figure 7 CartoonGAN network architecture**_

### 3.2.1 Loss Function
Our objective is to solve the min-max problem argminGmaxDL(G,D). The loss function consists of two parts: (1) the adversarial loss Ladv(G,D) and (2) the content loss Lcon(G,D). We calculate the total loss by addition of the two terms. 

![](https://imgur.com/JGu5WN6.jpg)

$L_{adv}(G,D)$ represents how similar the discriminator inputs look to the desired cartoon style. One important feature of cartoon images is that they have clear edges. However, the edges only take a little portion of the whole image. The clear edge feature is easily overlooked. A generated image with correct shading but unclear edges is likely to fool the discriminator. To avoid this problem, a dataset called “cartoon-smoothed” is created. This dataset contains cartoon images that go through edge smoothing filters. This dataset is labeled as non-cartoon for discriminator classification. Thus, $L_{adv}(G,D)$ is defined as

![](https://imgur.com/B5sAati.jpg)

$S_{data}(c)$ is the cartoon dataset, which contains cartoon images from artists of the desired style. $S_{data}(e)$ is the cartoon-smoothed dataset. $S_{data}(p)$ is the photo dataset, which contains real-world photos that will be converted to cartoonized images.
In addition to synthesizing real-world photos with cartoon style, the semantic content of real-world photos needs to be preserved as well. $L_{con}(G,D)$ represents the semantic content difference between real-world photos and the outputs of Generator. We obtain the content loss using the high-level feature map on a pretrained VGG model. 

![](https://imgur.com/Y2HVSzQ.jpg) <br>
l refers to the “conv4_4” layer of the VGG network.

### 3.2.2 Dataset
We collected 7,382 stills of Studio Ghibli movies as cartoon dataset, and 4,000 photos of people in the COCO dataset as real-world photo dataset.

## Results
### 4.1 Fast Style Transfer
We pick three movie scenes as our Pixar's style target. Fig 8, 9, and 10 show our result.The style of color is similar to our target and people in the photo are smooth like Pixar's characters.

![](https://imgur.com/Iiod8uq.jpg) <br>
_**Figure 8 result from Luca**_

![](https://imgur.com/gBtSeyz.jpg) <br>
_**Figure 9 result from Turning Red**_

![](https://imgur.com/sDVC4q8.jpg) <br>
_**Figure 10 result from Monster Inc.**_

## 4.2  CartoonGAN
We pick three real-world photos from COCO dataset and transform them into anime photos in the style of Studio Ghibli using cartoonGAN model. Fig 11, 12, and 13 show our result. It can be found that some obvious animation features, such as black borders around important objects, and soft colors are produced.

![](https://imgur.com/5XBhMdk.jpg) <br>
![](https://imgur.com/ssr9kiJ.jpg) <br>
![](https://imgur.com/0e8sjbD.jpg)

## 4.2.1 FLOPs and Parameters
- FLOPs : 61.73G
- Parameters：11.13M

## 4.3 Time Performance
As we had tried to train on two different models, we came out the efficiency of the two models to convert images was different.
![](https://imgur.com/LId4lkm.jpg) <br>
            Time consuming to transform one photo (Test image size:256x256)
            
## Demonstration
We implemented our training model by using the QT interface on the Windows system.

![](https://imgur.com/Kasij2G.jpg) <br>
_**Figure 11 UI interface**_

### video link
- fox.mp4
original
https://drive.google.com/file/d/12kZDxAn8cAgbr9VKDSCop-MagL9vjvEe/view?usp=sharing

transfered
https://drive.google.com/file/d/1wzid9dVkHpzclSzgUTmV1YC0d-sRRhrH/view?usp=sharing 
 
- skateboard.mp4
https://drive.google.com/file/d/1RAFLHmOniz3Bsu5L-tDKWsA_soFAlcZw/view?usp=sharing 

## Conclusions and Future Works
Aiming at transforming real-world photos into cartoon style images have reached a successful accomplishment. The experiments showed that Fast Style Transfer and CartoonGAN are able to finish the job but with different FLOPs and time consuming. 
	In the future work, we decide to implement some facial extraction on this model and make it more functional when it comes to cartoon stylization on human faces. Although we had added a gaussian noise layer before the discriminator to make the d_loss change from zero, we believe that there will be more improvements we can make. We will try similar ideas for the entire network layer, which will investigate further.    

## Requirements
You will need the following to run the above:
- TensorFlow
- Pytorch
- Colab

If your computer didn’t have GPU, it will consume plenty of time for training

## References
[1]Johnson, Justin, Alexandre Alahi, and Li Fei-Fei. "Perceptual losses for real-time style transfer and super-resolution." European conference on computer vision. Springer, Cham, 2016. 

[2]Yezhi Shu, Ran Yi, Mengfei Xia, Zipeng Ye, Wang Zhao, Yang Chen, Yu-Kun Lai, Yong-Jin Liu, GAN-based Multi-Stylr Photo Cartoonization, 2021

[3] Y. Chen, Y. -K. Lai and Y. -J. Liu, "CartoonGAN: Generative Adversarial Networks for Photo Cartoonization," 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2018, pp. 9465-9474, doi: 10.1109/CVPR.2018.00986.

[4]  Logan Engstrom. Fast Style Transfer, 2016 
