from xml.etree.ElementTree import parse
import cv2
import os
import math
import natsort
import numpy as np

# 한 이미지 리드하는 코드 추가
def ko_imread(filename, flags=cv2.IMREAD_COLOR, dtype=np.uint8):
    try:
        n = np.fromfile(filename, dtype)
        img = cv2.imdecode(n, flags)
        return img
    except Exception as e:
        return None
# 한국어 이미지 저장 코드
def ko_imwrite(filename, img, params=None):
    try:
        ext = os.path.splitext(filename)[1]
        result, n = cv2.imencode(ext, img, params)

        if result:
            with open(filename, mode='w+b') as f:
                n.tofile(f)
            return True
        else:
            return False
    except Exception as e:
               return False


def Find_all_file(folder_path):
    all_root = []
    for (path, dir, files) in os.walk(folder_path):
        for filename in files:
            filename = os.path.join(path, filename)
            all_root.append(filename)
    return all_root


paths = Find_all_file("xml")
img_paths = Find_all_file("img")
print(len(img_paths))

touch_count = 0
peep_count = 0
move_count = 0
full_count = 0
stand_count = 0
False_count = 0
b_count = 0

for path in paths:

    image_path = path.split("\\")
    image_path[0] = "img"
    image_path.remove(image_path[-1])
    #image_path[-1] = image_path[-1].split(".")[0]
    img_path = "\\".join(image_path)
    folders = os.listdir(img_path)
    for folder in folders:
        img_path = os.path.join(img_path,folder)
        if os.path.isdir(img_path) == False:
            print("경로문제",img_path)
            print("-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=")
            continue
        img_list = os.listdir(img_path)
        path = parse(path)
        root = path.getroot()
        image = root.findall("image")



        if len(image) != len(img_list):
            print(img_path)
            print("xml", len(image))
            print("이미지", len(img_list))
            print("no!!")
            print("-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=")
            continue

        for i, x in enumerate(image):
            name = x.attrib['name']
            img = ko_imread(os.path.join(img_path, name))

            if os.path.isfile(os.path.join(img_path, name)) == False:
                print("파일문제", os.path.join(img_path, name))
                print("-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=")
                False_count += 1
                continue

            box_count = 0


            for box in x.findall("box"):

                xbr = float(box.attrib["xbr"])
                ybr = float(box.attrib["ybr"])
                xtl = float(box.attrib["xtl"])
                ytl = float(box.attrib["ytl"])
                label = box.attrib["label"]

                # if os.path.isfile(os.path.join(img_path, name)) == False:
                #     print(os.path.join(img_path, name))
                #     print(label)
                #     continue

                if label == "Stand" or label =="stand":
                    #os.remove(os.path.join(img_path, name))
                    stand_count += 1
                    #print("delete")

                if label == "touch_bell" or label == "Touch_bell":
                    touch_count += 1
                    box_count += 1

                if label == "Move" or label == "move" or label == "MOve":
                    move_count += 1
                    box_count += 1

                if label == "peep" or label == "Peep":
                    peep_count += 1
                    box_count += 1


                # if box_count == 2:
                #     print(img_path, name)
                #     b_count += 1





                img = cv2.rectangle(img, (math.ceil(xbr), math.ceil(ybr)), (math.ceil(xtl), math.ceil(ytl)), (255, 0, 0), 3)
                img = cv2.putText(img, label, (math.ceil(xtl), math.ceil(ybr)), 1, 3, (255, 255, 0), 2)
                img = cv2.putText(img, img_path, (0, 30), 1, 2, (0, 255, 0), 2)
                img = cv2.putText(img, name, (0, 100), 1, 2, (0, 255, 0), 2)

            full_count += 1
            #ko_imwrite(os.path.join("test", img_path, str(i)) + ".jpg", img)

            cv2.imshow("11", img)
            cv2.waitKey(0)

print("peep", peep_count)
print("touch", touch_count)
print("move", move_count)
print("stand",stand_count)
print("full", full_count)
print("False", False_count)
print("b_count", b_count)




"""
root = xml_path.getroot()
#image = natsort.natsorted(root.findall("image"))
image = root.findall("image")
img_list = natsort.natsorted(os.listdir("1"))
if len(image) != len(img_list):
    print("no!!")
    exit()


for i, x in enumerate(image):
    name = x.attrib['name']
    img = cv2.imread(os.path.join("1", img_list[i]))
    print(name)
    print(img_list[i])




    for box in x.findall("box"):
        xbr = float(box.attrib["xbr"])
        ybr = float(box.attrib["ybr"])
        xtl = float(box.attrib["xtl"])
        ytl = float(box.attrib["ytl"])
        label = box.attrib["label"]
        print(xbr)

        img = cv2.rectangle(img, (math.ceil(xbr), math.ceil(ybr)), (math.ceil(xtl), math.ceil(ytl)), (255, 0, 0), 3)
        img = cv2.putText(img, label, (math.ceil(xtl), math.ceil(ybr)), 1, 3, (255, 255, 0), 2)


    cv2.imshow('test', img)
    cv2.waitKey(0)
    #cv2.imwrite(f"error_knock_apt/{i}.png", img)
"""





