# ***Segmentation of flowers***
This project is to do flower segmentation by applying U-Net. The data set that used in this project can be found in this [link](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html).  
This data set contains 102 different kinds of flowers which are very common in UK. There are about 40-258 images for each kind of flowers.  
In total, There are 8189 images in the data set. I divide it into training set and validation set which contains 7500 images and 689 images respectively.

Since all these images in the training set and validation set are in jpeg format, it is very easy to use keras to implement this project.
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


It is worth mention that because of the limitation of my devices to do this project, I resized all the training 
image and there masks into 64*64, which is relatively small compare to the original images.  


## INTRODUCTION  
![ori_image](https://github.com/NusLuoKe/102flowers/blob/master/readme_img/image_00001.jpg)
![ori_image](https://github.com/NusLuoKe/102flowers/blob/master/readme_img/mask_00001.jpg)  
Above two images are the original image and it's corresponding mask in the data set.
