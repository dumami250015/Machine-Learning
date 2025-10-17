import cv2
import numpy as np
import matplotlib as plt

# def calH(img):
#     l = np.max(img)
#     w, h = img.shape
#     p = [0] * (l + 1)
#     for i in range(w):
#         for j in range(h):
#             p[img[i][j]] += 1
#     return p

# img = np.array([[3, 2, 8],
#                 [5, 1, 0],
#                 [4, 6, 7],
#                 [5, 1, 2]])

# def out(p):
#     for i in p:
#         print(i, end = ' ')
#     print()
#     print("=================")

img = cv2.imread("tu.jpg")
imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
imgHSV = cv2.resize(imgHSV, (500, 500))

val = 0
for i in range(len(imgHSV)):
    for j in range(len(imgHSV[0])):
        if imgHSV[i][j][1] + val <= 255:
            imgHSV[i][j][1] += val

# imgAfter = cv2.rectangle(imgHSV, (50, 50), (10000, 10000), (100, 0, 0), 5)
imgAfter = cv2.cvtColor(imgHSV, cv2.COLOR_HSV2BGR)
cv2.imshow("HSV", imgAfter)

imgGRAY = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
imgGRAY = cv2.resize(imgGRAY, (500, 500))
cv2.imshow("GRAY", imgGRAY)

l = np.max(imgGRAY)
w, h = imgGRAY.shape
p = [0] * (l + 1)

for i in range(w):
    for j in range(h):
        p[imgGRAY[i][j]] += 1
for i in range(l + 1):
    p[i] = (p[i] * l) / (w * h)
for i in range(1, l + 1):
    p[i] = p[i - 1] + p[i]
for i in range(l + 1):
    p[i] = round(p[i])

imgBalance = np.copy(imgGRAY)
for i in range(w):
    for j in range(h):
        imgBalance[i][j] = p[imgGRAY[i][j]]

print("After", imgBalance)
# plt.title("Afer balance")
# plt.plot(calH(imgBalance))
# plt.show()
imgBalance = cv2.resize(imgBalance, (500, 500))
cv2.imshow("BALANCE", imgBalance)


# val = 0
# for i in range(len(imgHSV)):
#     for j in range(len(imgHSV[0])):
#         if imgHSV[i][j][1] + val <= 255:
#             imgHSV[i][j][1] += val

# imgAfter = cv2.rectangle(imgHSV, (50, 50), (10000, 10000), (100, 0, 0), 5)
# imgAfter = cv2.cvtColor(imgHSV, cv2.COLOR_HSV2BGR)
# cv2.imshow("after", imgAfter)

cv2.waitKey(0)
cv2.destroyAllWindows()
