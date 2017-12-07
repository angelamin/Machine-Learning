#使用方法
由于gpu机器在内网才可访问，所以通过在想要传输目录的另一台机器下运行
> python pyftpserver.py

然后打开网址
> ftp://xxxxx:2020

复制传输的文件链接，文件夹压缩

在gpu下
> wget 链接

从本机上传到机器
进入到本机目录，
ftp
open ip 2020
put
