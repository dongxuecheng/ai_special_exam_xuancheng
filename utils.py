def IoU(box1, box2):
    '''
    计算两个矩形框的交并比
    :param box1: list,第一个矩形框的左上角和右下角坐标
    :param box2: list,第二个矩形框的左上角和右下角坐标
    :return: 两个矩形框的交并比iou
    '''
    x1 = max(box1[0], box2[0])   # 交集左上角x
    x2 = min(box1[2], box2[2])   # 交集右下角x
    y1 = max(box1[1], box2[1])   # 交集左上角y
    y2 = min(box1[3], box2[3])   # 交集右下角y

    overlap = max(0., x2-x1) * max(0., y2-y1)
    union = (box1[2]-box1[0]) * (box1[3]-box1[1]) \
            + (box2[2]-box2[0]) * (box2[3]-box2[1]) \
            - overlap

    if union == 0:
        return 0.0

    return overlap / union