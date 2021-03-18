import numpy as np
def NMS(nums, threshold):
    x0 = nums[:, 0]
    y0 = nums[:, 1]
    x1 = nums[:, 2]
    y1 = nums[:, 3]
    score = nums[:, 4]
    area = (x1 - x0) * (y1 - y0)
    score = np.argsort(score)[::-1]
    print(score)
    keep = []
    while score.size > 0:
        #         print("score=",score)
        keep.append(score[0])
        print(keep)
        # 计算次大于score的所有框子与最大的score的IOU
        left_top_x = np.maximum(x0[score[0]], x0[score[1:]])  # 注意：这里是一个迭代器 for
        left_top_y = np.maximum(y0[score[0]], y0[score[1:]])
        right_bottom_x = np.minimum(x1[score[0]], x1[score[1:]])
        right_bottom_y = np.minimum(y1[score[0]], y1[score[1:]])
        w = np.maximum(0.0, right_bottom_x - left_top_x)
        h = np.maximum(0.0, right_bottom_y - left_top_y)
        # IOU
        overlap = w * h
        iou = overlap / (area[score[0]] + area[score[1:]] - overlap)
        #         print("iou=",iou)
        iou_low = np.where(iou <= threshold)[0]
        iou_low = iou_low  # 返回的是所有与最大的score相比的<threshold 的下标
        #         print("找到重叠度不高于阈值的矩形框索引:",iou_low)
        score = score[iou_low + 1]
        print(score)
    return keep
score=[0.9,0.7,0.8,0.85]
rec1=[0,0,100,100]
rec2=[50,50,150,150]
rec3=[60,60,170,170]
rec4=[150,150,200,200]
nums=np.array([rec1+[score[0]],rec2+[score[1]],rec3+[score[2]],rec4+[score[3]]])
assert nums.shape==(4,5),'value error'
print(nums)
keep=NMS(nums,0.5)
print(keep)