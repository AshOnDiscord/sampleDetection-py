import cv2 as cv
import numpy as np
import random as rng

def getCountours(mask):
  denoiseSize = np.ones((10, 10), np.uint8)
  denoised = cv.erode(mask, denoiseSize, iterations = 1)
  denoised = cv.dilate(denoised, denoiseSize, iterations = 1)
  denoised = cv.erode(denoised, np.ones((5, 5), np.uint8), iterations = 1)
  # denoised = cv.morphologyEx(denoised, cv.MORPH_OPEN, np.ones((10, 10), np.uint8))
  contours, heirachy = cv.findContours(denoised, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
  return (denoised, contours, heirachy)

def drawContours(img, contours, heirachy, contourColor, boxColor):
  cv.drawContours(img, contours, -1, contourColor, 2)
  for (i, c) in enumerate(contours):
    if heirachy[0, i, 3] != -1:
      continue
    _, _, w, h = cv.boundingRect(c)
    if w < 20 or h < 20:
      continue
    rect = cv.minAreaRect(c)
    v = np.intp(cv.boxPoints(rect))
    for j in range(4):
        cv.line(img, tuple(v[j]), tuple(v[(j+1)%4]), boxColor, 2)
  return img

img = cv.rotate(cv.resize(cv.imread("pit.jpg"), (0, 0), fx = 0.2, fy = 0.2), cv.ROTATE_90_COUNTERCLOCKWISE)
# keep only bright yellow
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
ycrcb = cv.cvtColor(img, cv.COLOR_BGR2YCrCb)

crChannel = ycrcb[:, :, 1]
crZeros = np.zeros(crChannel.shape, np.uint8)
crMat = cv.merge((crZeros, crChannel, crZeros))

cbChannel = ycrcb[:, :, 2]
cbZeros = np.zeros(cbChannel.shape, np.uint8)
cbMat = cv.merge((cbZeros, cbZeros, cbChannel))

contoursImg = img.copy()

_, thresholdY = cv.threshold(cbMat, 57, 255, cv.THRESH_BINARY_INV)
maskY = cv.inRange(thresholdY, (254, 254, 254), (255, 255, 255))
denoisedY, contoursY, heirachyY = getCountours(maskY)
# contoursImg = drawContours(img.copy(), contoursY, heirachyY, (255, 255, 0), (0, 255, 255))

_, thresholdB = cv.threshold(cbMat, 150, 255, cv.THRESH_BINARY)
maskB = cv.inRange(thresholdB, (0, 0, 254), (0, 0, 255))
denoisedB, contoursB, heirachyB = getCountours(maskB)
# contoursImg = drawContours(contoursImg, contoursB, heirachyB, (255, 255, 0), (255, 0, 0))

_, thresholdR = cv.threshold(crMat, 198, 255, cv.THRESH_BINARY)
maskR = cv.inRange(thresholdR, (0, 254, 0), (0, 255, 0))
denoisedR, contoursR, heirachyR = getCountours(maskR)
contoursImg = drawContours(contoursImg, contoursR, heirachyR, (255, 255, 0), (0, 0, 255))

# cv.imshow("DenoisedY", denoisedY)
# cv.imshow("DenoisedB", denoisedB)
cv.imshow("DenoisedR", denoisedR)
# cv.imshow("ThresholdR", thresholdR)
cv.imshow("MaskR", maskR)
# cv.imshow("Contours", contoursImg)

while True:
    if cv.waitKey(1) & 0xFF == ord('q'):
        break;
cv.destroyAllWindows();