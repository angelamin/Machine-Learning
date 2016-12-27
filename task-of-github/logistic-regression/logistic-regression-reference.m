clear all; close all; clc

x = load('q1x.dat'); %每一行是一个样本

y = load('q1y.dat');

[m, n] = size(x);

sample_num = m;

x = [ones(m, 1), x]; %x增加一维。

% Plot the training data

% Use different markers for positives and negatives

figure;

pos = find(y == 1); neg = find(y == 0);%pos 和neg 分别是 y元素=1和0的所在的位置序号组成的向量

plot(x(pos, 2), x(pos,3), '+')%用+表示那些yi=1所对应的样本

hold on

plot(x(neg, 2), x(neg, 3), 'o')

hold on

xlabel('Exam 1 score')

ylabel('Exam 2 score')

itera_num=500;%迭代次数

g = inline('1.0 ./ (1.0 + exp(-z))'); %这个就相当于制造了一个function g（z）=1.0 ./ (1.0 + exp(-z))

plotstyle = {'b', 'r', 'g', 'k', 'b--', 'r--'};

figure;%建立新的窗口

alpha = [ 0.0009, 0.001,0.0011,0.0012,0.0013 ,0.0014 ];%下面就分别用这几个学习速率看看哪个更好

for alpha_i = 1:length(alpha) %alpha_i是1,2，...6，表示的是学习速率向量和曲线格式向量的坐标：alpha(alpha_i)，plotstyle(alpha_i)

    theta = zeros(n+1, 1);%thera表示样本Xi各个元素叠加的权重系数，这里以向量形式表示，且初始都为0，三维向量

    J = zeros(itera_num, 1);%J是个100*1的向量，第n个元素代表第n次迭代cost function的值（下面用negtive 的对数似然函数，

    %因为是negtive 的，所以是求得极小值）

   for i = 1:itera_num %计算出某个学习速率alpha下迭代itera_num次数后的参数  

        z = x * theta;%这个z是一个列向量，每一个元素是每一个样本Xi的线性叠加和，因为X是所有的样本，因此这里不是一个一个样本算的，

        %而是所有样本一块算的，因此z是一个包含所有样本Xi的线性叠加和的向量。在公式中，是单个样本表示法，而在matlab中都是所有的样本一块来。

        h = g(z);%这个h就是样本Xi所对应的yi=1时，映射的概率。如果一个样本Xi所对应的yi=0时，对应的映射概率写成1-h。

        J(i) =(1/sample_num).*sum(-y.*log(h) - (1-y).*log(1-h));%损失函数的矢量表示法 这里Jtheta是个100*1的列向量。

        grad = (1/sample_num).*x'*(h-y);%这个是向量形式的，我们看到grad在公式中是gradj=1/m*Σ（Y（Xi）-yi）Xij ，写得比较粗略，

        %这里（Y（Xi）-yi）、Xij %都是标量，而在程序中是以向量的形式运算的，所以不能直接把公式照搬，所以要认真看看，代码中相应改变一下。

        theta = theta - alpha(alpha_i).*grad;

     end


    plot(0:itera_num-1, J(1:itera_num),char(plotstyle(alpha_i)),'LineWidth', 2)
%此处一定要通过char函数来转换因为包用（）索引后得到的还是包cell，

     %所以才要用char函数转换，也可以用{}索引，这样就不用转换了。

    %一个学习速率对应的图像画出来以后再画出下一个学习速率对应的图像。   

    hold on

    if(1 == alpha(alpha_i)) %通过实验发现alpha为0.0013 时效果最好，则此时的迭代后的theta值为所求的值

         theta_best = theta;

   end

 end

legend('0.0009', '0.001','0.0011','0.0012','0.0013' ,'0.0014');%给每一个线段格式标注上

xlabel('Number of iterations')

ylabel('Cost function')


prob = g([1, 20, 80]*theta);
%把[1, 20, 80]*theta这个带入g（z）得到的就是在exam1=20、exam2=80的条件下，通过的概率（也就是Y=1）的概率是多少。

 %画出分界面

 % Only need 2 points to define a line, so choose two endpoints

 plot_x = [min(x(:,2))-2,  max(x(:,2))+2];%两个点就可以画出来直线，这里这么取x1坐标是让直接的视野更容易包含那些样本点。


plot_y = (-1./theta(3)).*(theta(2).*plot_x +theta(1));%分界面怎么画呢？问题也就是在x1，x2坐标图中找到
%那些将x1，x2带入1/(1+exp(-wx))后，使其值>0.5的（x1,x2）

 %坐标形成的区域，因为我们知道1/(1+exp(-wx))>0.5


 %意味着该区域的（x1，x2）表示的成绩允许上大学的概率>0.5,那么其他的区域就是不被允许上大学，那么1/(1+exp(-wx))=0.5解出来的一个关
%于x1，x2的方程就是那个分界面。

%我们解出来以后发现，这个方程是一个直线方程：w（2）x1+w（3）x2+w（1）=0


 %注意我们不能因为这个分界面是直线，就认为logistic回归是一个线性分类器，注意logistic回归不是一个分类器，他没有分类的功能，
%这个logistic回归是用来


 %预测概率的，这里具有分类功能是因为我们硬性规定了一个分类标准：把>0.5的归为一类，<0.5的归于另一类。这是一个很强的假设，
%因为本来我们可能预测了一个样本


%所属某个类别的概率是0.6，这是一个不怎么高的概率，但是我们还是把它预测为这个类别，只因为它>0.5.所以最后可能logistic回归加上这
%个假设以后形成的分类器

%的分界面对样本分类效果不是很好，这不能怪logistic回归，因为logistic回归本质不是用来分类的，而是求的概率。

figure;

plot(x(pos, 2), x(pos,3), '+')%把分界面呈现在样本点上面，所以还要把样本点画出来。

 hold on

plot(x(neg, 2), x(neg, 3), 'o')

 hold on

 plot(plot_x, plot_y)

legend('Admitted', 'Not admitted', 'Decision Boundary')

 hold off 