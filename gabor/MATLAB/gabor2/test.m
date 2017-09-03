function [G,gabout] = gaborfilter1(I,Sx,Sy,f,theta)
%The Gabor filter is basically a Gaussian (with variances sx and sy along x and y-axes respectively)
%modulated by a complex sinusoid (with centre frequencies U and V along x and y-axes respectively)
%described by the following equation
%%%%%%%%%%%%%%%%%%%%%%
%                                  -1     x' ^      y' ^
%%% G(x,y,theta,f) =  exp ([----{(----) 2+(----) 2}])*cos(2*pi*f*x'); cos代表实部
%                                   2     sx'       sy'
%%% x' = x*cos(theta)+y*sin(theta);
%%% y' = y*cos(theta)-x*sin(theta);
%% Describtion :
%% I : Input image
%% Sx & Sy : Variances along x and y-axes respectively 方差
%% f : The frequency of the sinusoidal function
%% theta : The orientation of Gabor filter
%% G : The output filter as described above
%% gabout : The output filtered image
% %%isa判断输入参量是否为指定类型的对象
if isa(I,'double')~=1
    I = double(I);
end
%%%%Sx,Sy在公式里分别表示Guass函数沿着x,y轴的标准差，相当于其他的gabor函数中的sigma.
%%同时也用Sx,Sy指定了gabor滤波器的大小。（滤波器矩阵的大小）
%%这里没有考虑到相位偏移.fix(n)是取小于n的整数（往零的方向靠）
for x = -fix(Sx):fix(Sx)
    for y = -fix(Sy):fix(Sy)
        xPrime = x * cos(theta) + y * sin(theta);
        yPrime = y * cos(theta) - x * sin(theta);
        G(fix(Sx)+x+1,fix(Sy)+y+1) = exp(-.5*((xPrime/Sx)^2+(yPrime/Sy)^2))*cos(2*pi*f*xPrime);
    end
end
Imgabout = conv2(I,double(imag(G)),'same');
Regabout = conv2(I,double(real(G)),'same');
gabout = sqrt(Imgabout.*Imgabout + Regabout.*Regabout);
