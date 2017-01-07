import numpy as np
import cv2
from matplotlib import pyplot as plt
from skimage.measure import label, regionprops
from skimage.color import label2rgb
from skimage import img_as_ubyte

data_folder = "../data/"
results_path = "results/"
image_names = ["Klad00.jpg", "Klad01.jpg", "Klad02.jpg"]


def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 0.5)
    result = cv2.warpAffine(
        image, rot_mat, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR)
    return result


def get_direction(arrow_img):
    """ in radians """
    orientation = regionprops(label(arrow_img))[0].orientation
    angle = -(orientation) * 180 / np.pi
    rotated = rotate_image(arrow_img, angle)
    regions = regionprops(label(rotated))
    areas = [item.area for item in regions]
    r_region = regions[areas.index(max(areas))].filled_image
    size = 10
    left = r_region[:, :size].astype(int)
    right = r_region[:, -size:].astype(int)
    left = np.abs(left[:, :int(len(left[0]) / 2)] -
                  left[:, -int(len(left[0]) / 2):])
    right = np.abs(right[:, :int(len(right[0]) / 2)] -
                   right[:, -int(len(right[0]) / 2):])
    direction = orientation
    if np.mean(left) > np.mean(right):
        direction += np.pi

    # cv2.imshow("source", arrow_img)
    # cv2.imshow("rotated", rotated)
    # cv2.imshow("rotated_region", img_as_ubyte(r_region))
    # cv2.waitKey(0)
    if direction < 0:
        direction += np.pi * 2
    elif direction > np.pi * 2:
        direction -= np.pi * 2
    return direction


def get_objects(img):
    ret, th = cv2.threshold(img, 10, 255, cv2.THRESH_BINARY)
    l, num = label(th, return_num=True, background=0)
    # c = [item.filled_image for item in r]
    return regionprops(l, intensity_image=img)


def is_arrow(obj):
    arrow_eccentricity = 0.857
    delta = 0.02
    return abs(obj.eccentricity - arrow_eccentricity) < delta


def filter_arrows(objects):
    return list(filter(lambda item: is_arrow(item), objects))


def get_next_point(start, current, angle, step):
    cx = np.round(current[0])
    cy = np.round(current[1])
    sx = np.round(start[0])
    sy = np.round(start[1])
    angle_delta = 10
    angle_deg = angle * 180 / np.pi
    points = [
        (0, (cx, cy + step)),
        (360, (cx, cy + step)),
        (90, (cx - step, cy)),
        (45, (cx - step, cy + step)),
        (135, (cx - step, cy - step)),
        (180, (cx, cy - step)),
        (225, (cx + step, cy - step)),
        (270, (cx + step, cy)),
        (315, (cx + step, cy + step))
    ]

    distances = [(np.abs(a - angle_deg), p) for a, p in points]
    for d, p in distances:
        if d < angle_delta:
            return p
    min_1 = min(distances, key=lambda item: item[0])
    del distances[distances.index(min_1)]
    min_2 = min(distances, key=lambda item: item[0])
    # del distances[distances.index(min_2)]
    # min_3 = min(distances, key = lambda item: item[0])
    mins = [min_1, min_2]
    # m = max(mins, key = lambda item: item[0])[0]
    # print(m)
    # mins = [((m - item[0]) / m, item[1]) for item in mins]
    total_dist = 0
    for item in mins:
        total_dist += item[0]
    probs = [(total_dist - item[0]) / total_dist for item in mins]
    # print(probs)
    r_index = np.random.choice(len(mins), 1, p=probs)[0]
    return mins[r_index][1]


def find_route(img, objects, start):
    current = start
    while is_arrow(current):
        x, y = current.centroid
        except_current = list(filter(lambda item: item != current, objects))
        img_arrow = img_as_ubyte(current.filled_image)
        direction = get_direction(img_arrow)
        step = 2
        i = 0
        while True:
            i += 1
            x, y = get_next_point(current.centroid, (x, y), direction, step)

            img[int(x)][int(y)] = 128
            # if not i%10:
            # 	cv2.imshow("", img)
            # 	cv2.waitKey(0)

            point = [int(x), int(y)]
            current_changed = False
            for obj in except_current:
                # print(point)
                # print(obj.coords)
                if point in np.ndarray.tolist(obj.coords):
                    # print(obj.label)
                    current = obj
                    current_changed = True
                    break
            if current_changed:
                break
                # sys.exit()
    return img


def get_red_arrow(arrows):
    return min(arrows, key=lambda item: item.mean_intensity)


def main(image_name):
    img_path = data_folder + image_name
    img = cv2.imread(img_path, 0)
    img = cv2.medianBlur(img, 5)

    objects = get_objects(img)
    arrows = filter_arrows(objects)
    start = get_red_arrow(arrows)

    return find_route(img, objects, start)

for i_name in image_names:
    print(i_name)
    res = main(i_name)
    cv2.imwrite(results_path + i_name, res)
