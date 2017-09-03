ori=imread('/Users/xiamin/Desktop/test.jpeg');
 grayimg=rgb2gray(ori);
 gim=im2double(grayimg);
 
 [Eim,Oim,Aim]=spatialgabor(gim,3,90,0.5,0.5,1);%90-vertical===0-horizontal
 imshow(Aim);