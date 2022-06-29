import numpy as np


def get_src_dst_index_tl(margin):
    if margin>=0:
        src = margin
        dst = 0
    else:
        src = 0
        dst = -margin
    return int(src), int(dst)

def get_x2_box(face_box, image):
    height, width, _ = image.shape
    x1, y1, x2, y2 = face_box[:4]


    h = (y2-y1)
    w = (x2-x1)
    result_image = np.zeros((2*h, 2*w, 3), dtype=np.uint8)

    src_x1, dst_x1 = get_src_dst_index_tl(x1-w/2)
    src_y1, dst_y1 = get_src_dst_index_tl(y1-h/2)

    if(x2 + w/2) <= width:
        src_x2 = int(x2 + w/2)
    else:
        src_x2 = width

    if(y2+h/2) <= height:
        src_y2 = int(y2+h/2)
    else:
        src_y2 = height
    
    cropped_box = [src_x1, src_y1, src_x2, src_y2]
    cropped_width = src_x2 - src_x1
    cropped_height = src_y2-src_y1

    result_image[dst_y1:(dst_y1+cropped_height),dst_x1:(dst_x1+cropped_width), :] = image[src_y1:src_y2,src_x1:src_x2,:]

    return result_image