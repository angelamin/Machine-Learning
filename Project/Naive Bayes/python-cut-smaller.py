# coding: utf-8

#设置分句的标志符号；可以根据实际需要进行修改
cutlist ="。！？；：，|、/".decode('utf-8')

#检查某字符是否分句标志符号的函数；如果是，返回True，否则返回False
def FindToken(cutlist, char):
    if char in cutlist:
        return True
    else:
        return False

#进行分句的核心函数
def Cut(cutlist,lines):          #参数1：引用分句标志符；参数2：被分句的文本，为一行中文字符
    l = []         #句子列表，用于存储单个分句成功后的整句内容，为函数的返回值
    line = []    #临时列表，用于存储捕获到分句标志符之前的每个字符，一旦发现分句符号后，就会将其内容全部赋给l，然后就会被清空

    for i in lines:         #对函数参数2中的每一字符逐个进行检查 （本函数中，如果将if和else对换一下位置，会更好懂）
        print("11111")
        print(i)
        if FindToken(cutlist,i):       #如果当前字符是分句符号
            line.append(i)          #将此字符放入临时列表中
            l.append(''.join(line))   #并把当前临时列表的内容加入到句子列表中
            print(l)
            line = []  #将符号列表清空，以便下次分句使用
        else:         #如果当前字符不是分句符号，则将该字符直接放入临时列表中
            line.append(i)
    l.append(''.join(line))   #并把当前临时列表的内容加入到句子列表中

    return l

#以下为调用上述函数实现从文本文件中读取内容并进行分句。
for lines in file("t.txt"):
    print("lines")
    print(lines)
    l = Cut(list(cutlist),list(lines.decode('utf-8')))
    print("........")
    print(l)
    for line in l:
        print line
