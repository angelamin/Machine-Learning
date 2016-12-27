clc;
clear;
close all;

X=load('q1x.dat');
Y=load('q1y.dat');

[m,n]=size(X);
X=[ones(m,1),X]; %��X����һά

%pos��neg�ֱ��ʾYԪ�ص���1��0�ĵ�λ�������ɵ�����
pos=find(Y==1);
neg=find(Y==0);

plot(X(pos,2),X(pos,3),'+');
hold on;

plot(X(neg,2),X(neg,3),'o');
hold on;

theta = [1;1;1];
alpha = 1;

%�ݶ��½�
while 1
    h = 1./(1 + exp(-X*theta));
    ntheta = theta - (1/m)*alpha*X'*(h-Y);
    if abs(sum(ntheta - theta)) <= 0.1
        theta = ntheta;
        break;
    end;
    theta = ntheta;
end;

plot_x = [min(X(:,2))- 3 max(X(:,2)+3)];%������Ϳ��Ի���ֱ��
plot_y = [-1./theta(3) * (theta(2) * plot_x + theta(1))];

plot(plot_x,plot_y);
hold off;
legend('Positive','negative','decision bundary');


