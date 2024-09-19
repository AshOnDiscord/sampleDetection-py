import cv2 as cv
import numpy as np
import random as rng

img = cv.rotate(cv.resize(cv.imread("pit.jpg"), (0, 0), fx = 0.2, fy = 0.2), cv.ROTATE_90_COUNTERCLOCKWISE)
# keep only bright yellow
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
ycrcb = cv.cvtColor(img, cv.COLOR_BGR2YCrCb)
cbChannel = ycrcb[:, :, 2]
cbZeros = np.zeros(cbChannel.shape, np.uint8)
cbMat = cv.merge((cbZeros, cbZeros, cbChannel))
ret, ranged = cv.threshold(cbMat, 57, 255, cv.THRESH_BINARY_INV)
mask = cv.inRange(ranged, (254, 254, 254), (255, 255, 255))
masked = cv.bitwise_and(img, img, mask = mask)
denoiseSize = np.ones((10, 10), np.uint8)
denoised = cv.erode(mask, denoiseSize, iterations = 1)
denoised = cv.dilate(denoised, denoiseSize, iterations = 1)
denoised = cv.erode(denoised, np.ones((5, 5), np.uint8), iterations = 1)
denoised = cv.morphologyEx(denoised, cv.MORPH_OPEN, np.ones((10, 10), np.uint8))
contours, heirachy = cv.findContours(denoised, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
contoursImg = cv.drawContours(img.copy(), contours, -1, (0, 255, 0), 2)
# also draw bounding, prune any small or children contours
for (i, c) in enumerate(contours):
    if heirachy[0, i, 3] != -1:
        continue
    x, y, w, h = cv.boundingRect(c)
    if w < 20 or h < 20:
        continue
    rect = cv.minAreaRect(c)
    v = np.intp(cv.boxPoints(rect))
    for j in range(4):
        cv.line(contoursImg, tuple(v[j]), tuple(v[(j+1)%4]), (255, 255, 0), 2)
        
cv.imshow("Denoised", denoised)
cv.imshow("Contours", contoursImg)

while True:
    if cv.waitKey(1) & 0xFF == ord('q'):
        break;
cv.destroyAllWindows();