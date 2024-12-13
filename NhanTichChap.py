import cv2
import numpy as np
import matplotlib as plt

def cov(img, k):
    I = np.zeros(img.shape)
    k = np.array(k)
    iw, ih = img.shape
    kw, kh = k.shape
    for i in range(iw):
        for j in range(ih):
            sum = 0
            for u in range(kw):
                for v in range(kh):
                    uu = u + i - kw // 2
                    vv = v + j - kh // 2
                    if uu < 0 or vv < 0 or uu >= iw or vv >= ih: continue
                    sum += k[u][v] * img[uu][vv]
            I[i][j] = sum
    return I

k1 = [[-1, 0, 1],
      [-2, 0, 2],
      [-1, 0, 1]]

k2 = [[-1, -2, -1],
      [0, 0, 0],
      [1, 2, 1]]

imgGRAY = cv2.imread("tan.jpg", cv2.IMREAD_GRAYSCALE)
imgGRAY = cv2.resize(imgGRAY, (500, 500))
g1 = cov(imgGRAY, k1) * cov(imgGRAY, k1) 
g2 = cov(imgGRAY, k2) * cov(imgGRAY, k2) 
sobel = np.sqrt(g1 + g2)
cv2.imshow("GRAY", imgGRAY)
cv2.imshow("G1", cv2.convertScaleAbs(g1))
cv2.imshow("G2", cv2.convertScaleAbs(g2))
cv2.imshow("SOBEL", cv2.convertScaleAbs(sobel))

cv2.waitKey(0)
cv2.destroyAllWindows()