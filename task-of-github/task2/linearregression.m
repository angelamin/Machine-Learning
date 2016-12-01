clc;
clear;
close all;
%载入数据
X = load('q2x.dat');
Y = load('q2y.dat');
X = [ones(size(X,1),1),X];
xlabel('X');
ylabel('Y');
axis([-6 12 -2 2.5]);
m = size(X,1);%样本数
n = size(X,2);%特征维数
exa = [0.01 0.04 0.1 0.4 1 4 10 100];
for k = 1:length(exa)
    w = zeros(m,m);
    theta = zeros(n,m);
    for i = 1:m
        for j = 1:m
            w(j,j)=exp(-(X(i,2)-X(j,2))^2/(2*exa(k)^2));
        end;
        theta(:,i)=pinv(X'*w*X)*X'*w*Y;
    end;
    plot(X(:,2),Y,'r.');%原始数据
    hold on;
    y_fit = X*theta;
    y = diag(y_fit);
    data = [X(:,2),y];
    data = sortrows(data,1);
    hold on;
    plot(data(:,1),data(:,2));
end;

legend('training data','exa=0.01','exa=0.04','exa=0.1','exa=0.4','exa=1.0','exa=4.0','exa=10.0','exa=100.0')

