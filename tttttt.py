# 将tensor数据转为numpy数据.py
# import cv2
# # import dlib
# import numpy as np
#
# from torchvision import transforms
#
# path = "E:/GD/HRNet-Facial-Landmark/HRNet-Facial-Landmark-Detection-master/images/face.png"
#
# image = cv2.imread(path)
# cv2.imshow('Example', image)
# cv2.waitKey()
#
# transform2 = transforms.Compose([transforms.ToTensor()])
# tensor2 = transform2(image)
#
# array1 = tensor2.numpy()  # 将tensor数据转为numpy数据
# maxValue = array1.max()
# array1 = array1 * 255 / maxValue  # normalize，将图像数据扩展到[0,255]
# mat = np.uint8(array1)  # float32-->uint8
# print('mat_shape:', mat.shape)  # mat_shape: (3, 982, 814)
# mat = mat.transpose(1, 2, 0)  # mat_shape: (982, 814，3)
# cv2.imshow("img", mat)
# cv2.waitKey()
#
# mat=cv2.cvtColor(mat,cv2.COLOR_BGR2RGB)
# cv2.imshow("img",mat)
# cv2.waitKey()


##################################################

# 通过dlib在图像上识别关键点并标记在原图上
# import cv2
# import dlib
# import numpy as np
#
# path = "/data/opencv12/mv.jpg"
# img = cv2.imread(path)
# cv2.imshow("original", img)
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
# detector = dlib.get_frontal_face_detector()
# predictor = dlib.shape_predictor("/data/opencv12/shape_predictor_68_face_landmarks.dat")
# rects = detector(gray, 0)
# for i in range(len(rects)):
#     landmarks = np.matrix([[p.x, p.y] for p in predictor(img, rects[i]).parts()])
#     for idx, point in enumerate(landmarks):
#         pos = (point[0, 0], point[0, 1])
#         cv2.circle(img, pos, 3, color=(0, 255, 0))
#         font = cv2.FONT_HERSHEY_SIMPLEX
#         cv2.putText(img, str(idx + 1), pos, font, 0.3, (0, 0, 255), 1, cv2.LINE_AA)
# cv2.imshow("imgdlib", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

########################

# torch.max函数的dim参数的测试

# import torch

# output = torch.tensor(
#     [[1, 2, 3],
#      [2, 4, 5]])
# predict = torch.max(output, dim=0)
# print(predict)
# # print("predict.shape", predict.shape)
# predict = torch.max(output, dim=1)
# print(predict)
#
#
# torch.argmax(torch.max(output, 1).values, 0)
# print("predict.shape", predict.shape)


# import torch
# import multiprocessing as mp
#
# print(torch.__version__)  # 1.9.1+cu111
# print(torch.version.cuda)  # 11.1
# print(torch.backends.cudnn.version())  # 8005
# print(torch.cuda.current_device())  # 0
# print(torch.cuda.is_available())  # TRUE
#
# print(f"num of CPU: {mp.cpu_count()}")


# import os
# import time
#
# while 1:
#     os.system('nvidia-smi')
#     time.sleep(1)  # 1秒刷新一次
#     os.system('cls')  # 这个是清屏操作


# num = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
#        10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
#        20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
#        30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
#        40, 41, 42, 43, 44, 45, 46, 47, 48, 49,
#        50, 51, 52, 53, 54, 55, 56, 57, 58, 59,
#        60, 61, 62, 63, 64, 65, 66, 67, 68, 69,
#        70, 71, 72, 73, 74, 75, 76, 77, 78, 79,
#        80, 81, 82, 83, 84, 85, 86, 87, 88, 89,
#        90, 91, 92, 93, 94, 95, 96, 97, ]
# eye = 33
# new_num = len(num) - eye
# for i in range(new_num):
#     print(i)
#     offset = i + eye
#     print(offset)
#     print(num[offset])


# import os
#
# output_dir = ".\\output\\WFLW\\face_alignment_wflw_hrnet_w18_65"
# # filename = "checkpoint_0.pth"
# filename = "1.txt"
# # latest_path = os.path.join(output_dir, 'latest.pth')
# latest_path = os.path.join(output_dir, 'latest.txt')
# if os.path.islink(latest_path):
#     os.remove(latest_path)
# print("output_dir:", output_dir)
# print("latest_path:", latest_path)
# print("os.path.join(output_dir, filename)", os.path.join(output_dir, filename))
# if os.access(os.path.join(output_dir, filename), os.R_OK):
#     print("yes")
#     os.symlink(os.path.abspath(os.path.join(output_dir, filename)), os.path.abspath(latest_path))


# 是的，可以实现自适应的方法。为了自动调整`scale`参数，可以根据图像中的特征（例如人脸大小）来计算一个合适的值。这里以OpenCV的人脸检测作为示例：

# # ```python
# import cv2
#
# def detect_face_scale(img, min_face_size_ratio=0.1, base_scale=1.0):
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
#     faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
#
#     if len(faces) > 0:
#         face_sizes = [w * h for x, y, w, h in faces]
#         max_face_size = max(face_sizes)
#         img_size = img.shape[0] * img.shape[1]
#         face_size_ratio = max_face_size / img_size
#
#         scale_factor = base_scale * min_face_size_ratio / face_size_ratio
#         return scale_factor
#     else:
#         return base_scale
#
# # Example usage:
# img = cv2.imread("path/to/image.jpg")
# adaptive_scale = detect_face_scale(img)
# cropped_image = crop(img, center, adaptive_scale, output_size, rot=0)
# # ```

# 在这个示例中，`detect_face_scale`函数使用OpenCV的人脸检测器检测图像中的人脸，并根据人脸大小计算一个自适应的`scale`值。您可以根据需要调整`min_face_size_ratio`参数来控制放大程度。


# import matplotlib.pyplot as plt
# import numpy as np
# import math
#
#
# def sigmoid(x):
#     return 1 / (1 + np.exp(-x))
# def tanh(x):
#     # result = np.exp(x)-np.exp(-x)/np.exp(x)+np.exp(-x)
#     result = (math.e ** (x) - math.e ** (-x)) / (math.e ** (x) + math.e ** (-x))
#     return result
#
#
# def relu(x):
#     result = np.maximum(0, x)
#     return result
#
# x = np.arange(-10, 10, 0.1)  # 起点，终点，间距
# y = relu(x)
# plt.plot(x, y)
# plt.show()


x = self.conv1(x)
x = self.bn1(x)
x = self.relu(x)
x = self.conv2(x)
x = self.bn2(x)
x = self.relu(x)
x = self.layer1(x)