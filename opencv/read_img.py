from skimage import io
image = io.imread("http://m.wangchao.net.cn/resizeimg.jsp?imageurl=http://image.wangchao.net.cn/it/1317056657070.jpg")
print image
io.imshow(image)
io.show()
