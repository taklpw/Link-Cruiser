import numpy as np
import cv2


classes = None
with open('yolov3.txt', 'r') as f:
    classes = [line.strip() for line in f.readlines()]
COLORS = np.random.uniform(0, 255, size=(len(classes), 3))
# net = cv2.dnn.readNet('yolov3-tiny.weights', 'yolov3-tiny.cfg')
net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')


def get_output_layers():
    global net
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return output_layers


def yolo_mask_create(image):
    global net
    global classes
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_img = np.array(gray_img, dtype=np.uint8)

    width = image.shape[1]
    height = image.shape[0]
    scale = 0.00392

    blob = cv2.dnn.blobFromImage(image, scale, (416, 416), (0, 0, 0), True, crop=False)

    net.setInput(blob)

    outs = net.forward(get_output_layers())

    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.3
    nms_threshold = 0.1

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.1 and classes[class_id] == 'person':
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    img_mask = np.ones(gray_img.shape)

    for index in indices:
        i = index[0]
        box = boxes[i]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        image[round(y):round(y + h), round(x):round(x + w)] = 0
        # Create a mask for the image
        img_mask[round(y):round(y + h), round(x):round(x + w)] = 0

    return img_mask


def mask_people(input_image):
    people_mask = yolo_mask_create(input_image)
    return people_mask


if __name__ == '__main__':
    image = cv2.imread('class.jpg')
    mask = mask_people(image)
    idx = (mask == 0)
    masked_image = image
    masked_image[idx] = image[idx]
    cv2.imshow("mask ", masked_image)
    cv2.waitKey(0)
