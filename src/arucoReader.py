import cv2
from cv2 import aruco
import numpy as np
import morph


def readImage(image):
    dict_aruco = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    parameters = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(dict_aruco, parameters)

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    corners, ids, rejected = detector.detectMarkers(gray)

    if ids is None:
        return None, None

    ids = ids.flatten()

    return corners, ids


def generateLastPoint(points):
    pt1 = np.array(points[0])
    pt2 = np.array(points[1])
    pt3 = np.array(points[2])

    pts = [pt1, pt2, pt3]

    d12 = np.linalg.norm(pt1[0] - pt2[0])
    d23 = np.linalg.norm(pt2[0] - pt3[0])
    d13 = np.linalg.norm(pt1[0] - pt3[0])

    if d12 > d23 and d12 > d13:
        temp = np.copy(pt3)
        pt3 = np.copy(pt1)
        pt1 = np.copy(temp)
    elif d13 > d12 and d13 > d23:
        temp = np.copy(pt2)
        pt2 = np.copy(pt1)
        pt1 = np.copy(temp)

    v1 = pt3 - pt1
    v2 = pt2 - pt1

    pt4 = pt1 + v1 + v2

    # print(pt1)
    # print(pt2)
    # print(pt3)
    # print(v1, v2)
    # print(pt4)

    return pt4


def extractPaperBounds(image):
    corners, ids = readImage(image)

    if ids is None:
        return None

    ncorners = []
    bounds = np.zeros((4, 2), dtype="float32")
    spoints = np.zeros((4, 2), dtype="float32")
    flag = np.zeros(4, dtype="bool")

    for i in range(len(ids)):
        if ids[i] == 0 or ids[i] == 1 or ids[i] == 2 or ids[i] == 3:
            flag[ids[i]] = True
            ncorners.append(corners[i][0])

    if len(ncorners) == 3:
        pt4 = np.copy(generateLastPoint(ncorners))
        ncorners.append(pt4)
        for i in range(4):
            flag[i] = True

    if len(ncorners) != 4:
        return None

    for v in flag:
        if not v:
            # print(v)
            return None

    spoints = [p[0] for p in ncorners]
    spoints = np.copy(morph.sortPoints(spoints))

    # to solve the instances where paper is rotated around 60 degress and is perceived as horizontal
    # for i in range(4):
    #     ncorners[i] = np.copy(morph.sortPoints(ncorners[i]))

    # print(ncorners)
    # print("yigit")

    for i in range(4):
        for j in range(4):
            if ncorners[j][0][0] == spoints[i][0] and ncorners[j][0][1] == spoints[i][1]:
                bounds[i] = ncorners[j][i]

    # print(bounds)
    # print("Furkan")

    return bounds


def warpArucoImg(image):
    bounds = extractPaperBounds(image)

    if bounds is None:
        return None

    result = morph.four_point_transform_image(image, bounds)
    return result


def getTransformMatrix(image):
    bounds = extractPaperBounds(image)

    if bounds is None:
        return None, None, None

    M, w, h = morph.getTransformMatrix(bounds)
    return M, w, h


def warpArucoPoint(image, x, y):
    bounds = extractPaperBounds(image)

    if bounds is None:
        return None

    x_res, y_res = morph.four_point_transform(bounds, x, y)

    return x_res, y_res


def findPage(image):
    corners, ids = readImage(image)

    if ids is None:
        return None

    for ucoid in ids:
        if ucoid > 3:
            return ucoid - 4
    return None

# image = cv2.imread("arucosample.jpg")
#
# cv2.imshow("hold the line", warpArucoImg(image))
# cv2.waitKey(0)
# cv2.destroyAllWindows()
