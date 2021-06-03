import cv2
import os
import torch
import torchvision
import numpy as np

filename0 = "data\\"+os.listdir("data")[0]+"\\"+"16.jpg"
filename1 = "data\\"+os.listdir("data")[0]+"\\"+"17.jpg"
filename2 = "data\\"+os.listdir("data")[0]+"\\"+"18.jpg"
# img0 = cv2.imread(filename0)
# img1 = cv2.imread(filename1)
# img2 = cv2.imread(filename2)
# img0_GRAY = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
# img1_GRAY = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
# img2_GRAY = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
# cv2.imshow("img0", img0)
# cv2.imshow("img1", img1)
# print(cv2.absdiff(img0_GRAY, img1_GRAY).reshape((img2_GRAY.shape[0], img2_GRAY.shape[1], 1)))
# ret, threshold0 = cv2.threshold(cv2.absdiff(img0_GRAY, img1_GRAY), 20, 255, cv2.THRESH_BINARY)
# ret, threshold1 = cv2.threshold(cv2.absdiff(img1_GRAY, img2_GRAY), 20, 255, cv2.THRESH_BINARY)
# cv2.imshow("temp0&1", threshold0)
# cv2.waitKey()
# temp = list()
# temp.append(img)
# temp.append(img1)
# temp_np = np.append(img, img1, axis=2)
# print(filename)
# print(img.shape)
# print(img1.shape)
# print(temp_np.shape)
# print(torch.tensor(temp).shape)

# input_tensor = torch.randn((1, 3, 4, 480, 640))  # batch, inchannel, join_num, W, H
# Conv3D = torch.nn.Conv3d(in_channels=3, out_channels=3, kernel_size=(3, 5, 5), padding=(0, 2, 2))
# input_tensor = Conv3D(input_tensor)
# temp_tensor = input_tensor
# print(input_tensor.shape)
# flatten = torch.nn.Flatten(start_dim=1, end_dim=2)
# input_tensor = flatten(input_tensor)
# print(input_tensor.shape)
# unflatten = torch.nn.Unflatten(1, (3, 2))
# input_tensor = unflatten(input_tensor)
# print(input_tensor.shape)
# print(temp_tensor - input_tensor)

# image = 0
# image_dirlist = [filename0, filename1, filename2]
# temp = list()
# for dirname in image_dirlist:
#     if type(image) == int:
#         image = cv2.imread(dirname)
#         # preframe = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     else:
#         tempframe = cv2.imread(dirname)
#         image = np.append(image, tempframe, axis=2)
# image = torchvision.transforms.F.to_tensor(image)
# unflatten = torch.nn.Unflatten(0, (3, 3))
# image = unflatten(image)
# print(image.shape)
# cv2.imshow()
# img = (np.random.random((100, 100))*256).astype(np.uint8)
# cv2.imshow("gray", img)
# img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
# cv2.imshow("BGR", img)
# cv2.waitKey()
# cv2.destroyAllWindows()
# CLASS = {"冰害": 0, "覆冰过载": 1, "脱冰跳跃": 2, "舞动": 3, "风偏": 4, "雷击": 5, "反击": 6, "绕击": 7, "鸟害": 8, "其他": 9,
#          "外破": 10, "山火": 11, "施工碰线": 12, "异物短路": 13, "污闪": 14}
# print(CLASS["山火"])
# torch.nn.Linear(2048, num_classes)
a = torch.tensor([[5, 7, 9],
                  [6, 1, 3],
                  [4, 8, 2]])
pred, idx = a.max(1)
label = torch.tensor([2, 2, 2])
print(idx)
print((idx == label).float().mean())
print(len(label))
