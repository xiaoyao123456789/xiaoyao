#导包
import cv2.cv2 as cv2
import numpy as np
import matplotlib.pyplot as plt

#图片读取
#原图
image1=cv2.imread("E:\\LearnData\\Data1\\sx\\use_IMG\\opencv\\123.jpg",cv2.IMREAD_UNCHANGED)
#灰度
image2=cv2.imread("E:\\LearnData\\Data1\\sx\\use_IMG\\opencv\\123.jpg",cv2.IMREAD_GRAYSCALE)
#彩色
image3=cv2.imread("E:\\LearnData\\Data1\\sx\\use_IMG\\opencv\\123.jpg",cv2.IMREAD_COLOR)


cv2.imshow("1",image1)
cv2.imshow("2",image2)
cv2.imshow("3",image3)

cv2.waitKey(0)
cv2.destroyAllWindows()
