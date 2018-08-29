# ***Segmentation of flowers***
## ***INTRODUCTION***  
* This project is to do flower segmentation by applying U-Net. The data set that used in this project can be found in this [link](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html).  
  
* This data set contains 102 different kinds of flowers which are very common in UK. There are about 40-258 images for each kind of flowers. In total, There are 8189 images in the data set. I divide it into training set and validation set which contains 7500 images and 689 images respectively.

* Since all these images in the training set and validation set are in jpeg format, it is very easy to use keras to implement this project.
The only thing need to do is to put images in a subdirectory of these directories just as the following.   

        input/
            train_flower/
                class_0/
                    0.jpg
                    1.jpg
                    2.jpg
                    3.jpg
                    ...
            train_mask/
                class_0/
                    0.jpg
                    1.jpg
                    2.jpg
                    3.jpg
                    ...
            test_flower/
                class_0/
                    0.jpg
                    1.jpg
                    2.jpg
                    3.jpg
                    ...
            test_mask/
                class_0/
                    0.jpg
                    1.jpg
                    2.jpg
                    3.jpg
                    ...





## ***Data Illustrates***  
<img src="https://github.com/NusLuoKe/102flowers/blob/master/readme_img/image_00001.jpg" style="zoom:50%" />           

<img src="https://github.com/NusLuoKe/102flowers/blob/master/readme_img/mask_00001.jpg" style="zoom:50%" />

Above two images are the original image and it's corresponding mask in the data set.


* It is worth mention that I resized all the training image and there masks into 64*64 when training, which is 
  relatively small compare to the original images. This is because the aspect ratio of many images is not 1 : 1 and it is better to resize all the 
  images into a particular size although we are using a FCN which can accept all size of images. It is also due the limited 
  performance of my devices to do this project.    

<img src="https://github.com/NusLuoKe/102flowers/blob/master/readme_img/resized_flower_00001.jpg" style="zoom:50%" />

<img src="https://github.com/NusLuoKe/102flowers/blob/master/readme_img/resized_mask_00001.jpg" style="zoom:50%" />

The images and corresponding masks used to train the model are showing above. 
shape = (400, 400, 3)

---
## ***Results***
I have tried three models to do the segmentation. And I applied early stopping as the callback function for all these three cases.  

* This first model I tied is Unet_128, which does not contain batch normalization. And the dice on the validation set is about 0.78.  
* Then in my second try, I used Unet_256, which is a deeper model compared to the first one. But the dice did not improve much. The final
dice on the validation set is about 0.79.
* The third model is Unet_128_bn, which a add batch normalization layer and a "relu" activation function after each convolutional 
layer. It has a little improvement compare to the Unet_128, but the final dice is also not very high. It is about 0.81 on the validation set.
  
I test the Unet_128_bn model on some fresh input images. The input image should be gray image which has a shape 
as (256, 256, 1). Results are as follows. For each group, there are three images in order, the original image, 
the cropped grayscale image and grayscale image corresponding segmentation.
     
### test image 1
<img src="https://github.com/NusLuoKe/102flowers/blob/master/readme_img/test_flower_1.jpg" style="zoom:50%" />

<img src="https://github.com/NusLuoKe/102flowers/blob/master/readme_img/test_flower_1_gray.png" style="zoom:50%" />

<img src="https://github.com/NusLuoKe/102flowers/blob/master/readme_img/test_flower_1_pred.png" style="zoom:50%" />

### test image 2

<img src="https://github.com/NusLuoKe/102flowers/blob/master/readme_img/test_flower_2.jpg" style="zoom:50%" />

<img src="https://github.com/NusLuoKe/102flowers/blob/master/readme_img/test_flower_2_gray.png" style="zoom:50%" />

<img src="https://github.com/NusLuoKe/102flowers/blob/master/readme_img/test_flower_2_pred.png" style="zoom:50%" />