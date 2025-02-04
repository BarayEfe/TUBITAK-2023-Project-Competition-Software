import math
import cv2 as cv
import numpy as np
from pyzbar import pyzbar

qrdetector = cv.QRCodeDetector()

# image = cv.imread("unnamed2.jpg")

# 0 --> Top Left | 1 --> Top Right | 2 --> Bottom Left | 3 --> Bottom Right

def findBottomQR(top_qr):
    if len(top_qr) != 2:
        return None

    pos = getPosPyzbar(top_qr)
    pos[0] = sortPoints(pos[0])
    pos[1] = sortPoints(pos[1])

    x1 = pos[0][0][0]
    x2 = pos[1][1][0]

    y1 = pos[0][0][1]
    y2 = pos[1][1][1]

    angle = np.arctan2(x1 - x2, y1 - y2)
    dist_between = (np.sqrt(np.square(x1 - x2) + np.square(y1 - y2)))
    dist_to_bottom = dist_between * np.sqrt(2) # the ratio of a4 papers



def find_QR_pyzbar(img):
    gray_image = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    _, thresh = cv.threshold(gray_image, 127, 255, cv.THRESH_BINARY)
    return pyzbar.decode(gray_image)


def getDataPyzbar(qr_codes):
    data = []

    if qr_codes:
        for qr_code in qr_codes:
            data.append(qr_code.data.decode('utf-8'))
        return data

    return None


def getPosPyzbar(qr_codes):
    pos = []

    if qr_codes:
        for qr_code in qr_codes:
            poly = []
            for pts in qr_code.polygon:
                point = [pts.x, pts.y]
                poly.append(point)
            pos.append(poly)
        return pos

    return None


def find_QR_cv(img_path):
    image = cv.imread(img_path)
    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    return qrdetector.detectAndDecodeMulti(gray_image)


def sortPoints(points):
    centroid = [sum(p[0] for p in points) / len(points), sum(p[1] for p in points) / len(points)]
    angles = [math.atan2(p[1] - centroid[1], p[0] - centroid[0]) for p in points]
    sorted_points = [p for _, p in sorted(zip(angles, points), key=lambda x: x[0])]

    return sorted_points


def findBounds(qr_data, pos):
    bounds = np.zeros((4, 2), dtype="float32")
    points = np.zeros((4, 4, 2), dtype="float32")
    spoints = np.zeros((4, 2), dtype="float32")
    flag = [False, False, False, False]

    for i in range(len(qr_data)):
        # print("(" + qr_data[i] + ")")
        # print(pos[i])

        if qr_data[i] == "TL":
            points[0] = pos[i]
            flag[0] = True
        elif qr_data[i] == "TR":
            points[1] = pos[i]
            flag[1] = True
        elif qr_data[i] == "BR":
            points[2] = pos[i]
            flag[2] = True
        elif qr_data[i] == "BL":
            points[3] = pos[i]
            flag[3] = True

    for check in flag:
        if not check:
            return None

    spoints = [p[0] for p in points]
    sorted_points = sortPoints(spoints)

    for i in range(4):
        points[i] = np.copy(sortPoints(points[i]))

    for i in range(4):
        for j in range(4):
            if points[j][0][0] == sorted_points[i][0] and points[j][0][1] == sorted_points[i][1]:
                bounds[i] = np.copy(points[j][i])

    return bounds


def getTransformMatrix(pts):
    # obtain a consistent order of the points and unpack them
    # individually
    tl = pts[0]
    tr = pts[1]
    br = pts[2]
    bl = pts[3]

    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordinates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # compute the perspective transform matrix and then apply it
    M = cv.getPerspectiveTransform(pts, dst)

    return M, maxWidth, maxHeight


def four_point_transform_image(img, pts):
    # obtain a consistent order of the points and unpack them
    # individually
    tl = pts[0]
    tr = pts[1]
    br = pts[2]
    bl = pts[3]

    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordinates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # compute the perspective transform matrix and then apply it
    M = cv.getPerspectiveTransform(pts, dst)
    warped = cv.warpPerspective(img, M, (maxWidth, maxHeight))

    # return the warped image
    return warped


def four_point_transform(img, pts, x, y):
    # obtain a consistent order of the points and unpack them
    # individually
    tl = pts[0]
    tr = pts[1]
    br = pts[2]
    bl = pts[3]

    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordinates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # compute the perspective transform matrix and then apply it
    M = cv.getPerspectiveTransform(pts, dst)
    warped = cv.warpPerspective(img, M, (maxWidth, maxHeight))

    # Transform the point using the perspective transformation matrix
    transformed_point = cv.perspectiveTransform(np.array([[[x, y]]], dtype=np.float32), M)

    # Extract the transformed coordinates
    transformed_x, transformed_y = transformed_point[0][0]

    # return the warped image and warped point
    return warped, transformed_x, transformed_y


def four_point_transform(pts, x, y):
    # obtain a consistent order of the points and unpack them
    # individually
    tl = pts[0]
    tr = pts[1]
    br = pts[2]
    bl = pts[3]

    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordinates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # compute the perspective transform matrix and then apply it
    M = cv.getPerspectiveTransform(pts, dst)

    # Transform the point using the perspective transformation matrix
    transformed_point = cv.perspectiveTransform(np.array([[[x, y]]], dtype=np.float32), M)

    # Extract the transformed coordinates
    transformed_x, transformed_y = transformed_point[0][0]

    # return the warped image and warped point
    return transformed_x, transformed_y


def warpImage(img):
    qr_codes = find_QR_pyzbar(img)

    if qr_codes is None or qr_codes == []:
        return None

    print(qr_codes)

    qr_data = getDataPyzbar(qr_codes)
    qr_pos = getPosPyzbar(qr_codes)
    paper_bounds = findBounds(qr_data, qr_pos)

    if paper_bounds is None:
        return None

    print(paper_bounds)

    result_img = four_point_transform_image(img, paper_bounds)
    return result_img


def warpImagePoint(img, x, y):
    qr_codes = find_QR_pyzbar(img)

    if qr_codes is None or qr_codes == []:
        return None

    qr_data = getDataPyzbar(qr_codes)
    qr_pos = getPosPyzbar(qr_codes)
    paper_bounds = findBounds(qr_data, qr_pos)

    if paper_bounds is None:
        return None

    result_img, res_x, res_y = four_point_transform(img, paper_bounds, x, y)
    return result_img, res_x, res_y


def warpPoint(img, x, y):
    qr_codes = find_QR_pyzbar(img)

    if qr_codes is None or qr_codes == []:
        return None

    qr_data = getDataPyzbar(qr_codes)
    qr_pos = getPosPyzbar(qr_codes)
    paper_bounds = findBounds(qr_data, qr_pos)

    if paper_bounds is None:
        return None

    res_x, res_y = four_point_transform(paper_bounds, x, y)
    return res_x, res_y
