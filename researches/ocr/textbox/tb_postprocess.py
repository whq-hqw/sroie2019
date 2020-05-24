import os, datetime
import numpy as np
from researches.ocr.textbox.tb_utils import *
import omni_torch.visualize.basic as vb


def combine_boxes(prediction, img, h_thres_pct = 1.5, y_thres_pct=1, combine_thres=0.7,
                  overlap_thres=0.0, verbose=False):
    save_dir = os.path.expanduser("~/Pictures/")
    #print_box(red_boxes=prediction, shape=(h, w), step_by_step_r=True, save_dir=save_dir)
    w = img.size(3)
    h = img.size(2)
    output_box = []
    _scale = torch.Tensor([w, h, w, h])
    if prediction.is_cuda:
        _scale = _scale.cuda()
    scale = _scale.unsqueeze(0).repeat(prediction.size(0), 1)
    prediction = prediction * scale
    
    # Eliminate White Boxes
    # Method 1: eliminate by color
    qualified_boxes=[]
    for _, pred in enumerate(prediction):
        cropped = img[:, :, int(pred[1]): int(pred[3]), int(pred[0]): int(pred[2])]
        avg_value = 255 * (float(torch.sum(cropped) / cropped.nelement()) + 0.5)
        if avg_value > 245:
            continue
            dt = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
            std_black = int(torch.std(torch.histc((255 * (cropped + 0.5)).cpu(), bins=50, min=0, max=85)))
            std_white = int(torch.std(torch.histc((255 * (cropped + 0.5)).cpu(), bins=50, min=170, max=255)))
            vb.plot_tensor(None, cropped, margin=5, bg_color=128,
                           path="/home/wang/Pictures/%d_%d_%d_%s_%d.jpg"%(round(avg_value), std_black, std_white, dt, _))
        else:
            #pass
            qualified_boxes.append(pred)
    prediction = torch.stack(qualified_boxes, dim=0)
    
    # Method 2: eliminate by histogram
    # Eliminate by variance
    #torch.histc(input, bins=100, min=0, max=0, out=None)

    # Merge the boxes contained in other boxes
    merged_boxes = []
    before_merge = prediction.size(0)
    unmerge_idx = torch.ones(prediction.size(0)).byte()
    inter = intersect(prediction, prediction)
    pred_size = get_box_size(prediction).unsqueeze(0).expand_as(inter)
    #indicator = 2 / (inter / pred_size + pred_size / inter) - identity
    indicator = (inter / pred_size) > combine_thres
    for idctr in indicator:
        if torch.sum(idctr) <= 1:
            # the element on the diagonal is 1
            continue
        # eliminate the index of predicted boxes that need to be merged
        unmerge_idx[idctr] = 0
        merged_boxes.append(
            torch.cat([torch.min(prediction[idctr][:, :2], dim=0)[0], torch.max(prediction[idctr][:, 2:], dim=0)[0]])
        )
        idx = idctr.unsqueeze(0).expand_as(indicator)
        # once a box is merged, it does not need tp be merged or calculated again
        indicator[idx] = 0
    if len(merged_boxes) > 0:
        merged_boxes = torch.stack(merged_boxes, dim=0)
        unmerged_boxes = prediction[unmerge_idx]
        prediction = torch.cat([unmerged_boxes, merged_boxes], dim=0)
    after_merge = prediction.size(0)
    if before_merge > after_merge and verbose:
        print("merged %d boxes"%(before_merge - after_merge))

    # Find boxes with similar height
    vertical_height = (prediction[:, 3] - prediction[:, 1])
    vertical_height = vertical_height.unsqueeze(0).repeat(vertical_height.size(0), 1)
    dis_matrix = torch.abs(vertical_height - vertical_height.permute(1, 0))
    idx_h = dis_matrix < (h_thres_pct * h / 100)
    # Find boxes at almost same height
    vertical_height = (prediction[:, 3] + prediction[:, 1]) / 2
    vertical_height = vertical_height.unsqueeze(0).repeat(vertical_height.size(0), 1)
    dis_matrix = torch.abs(vertical_height - vertical_height.permute(1, 0))
    idx_v = dis_matrix < (y_thres_pct * h / 100)
    idx = idx_h * idx_v
    # Iterate idx in axis=0
    eliminated_box_id = set([])
    for i, box_id in enumerate(idx):
        #print(i)
        if i in eliminated_box_id:
            continue
        if int(torch.sum(box_id[i:])) == 1:
            output_box.append(prediction[i, :] / _scale)
            eliminated_box_id.add(i)
        else:
            # boxes that have the potential to be connected
            _box_id = np.where(box_id.cpu().numpy() == 1)[0]
            qualify_box = prediction[box_id]
            overlaps = jaccard(qualify_box, qualify_box)
            similar_boxes = overlaps > overlap_thres
            for j, similar_id in enumerate(similar_boxes):
                #print(similar_boxes)
                # Make the lower triangle part to be 0
                similar_id[:j] = 0
                if int(torch.sum(similar_id)) == 0:
                    continue
                elif int(torch.sum(similar_id)) == 1:
                    # this box has no intersecting boxes
                    if int(_box_id[(similar_id > 0).nonzero().squeeze()]) in eliminated_box_id:
                        continue
                    eliminated_box_id.add(_box_id[(similar_id > 0).nonzero().squeeze()])
                    output_box.append(qualify_box[similar_id].squeeze() / _scale)
                else:
                    comb_boxes = qualify_box[similar_id]
                    # Combine comb_boxes
                    new_box = torch.cat([torch.min(comb_boxes[:, :2], dim=0)[0], torch.max(comb_boxes[:, 2:], dim=0)[0]])
                    output_box.append(new_box / _scale)
                    eliminated_box_id = eliminated_box_id.union(set(_box_id[(similar_id.cpu() > 0).nonzero().squeeze()]))
                # Eliminate the boxes that already been combined
                zero_id = similar_id.unsqueeze(0).repeat(similar_boxes.size(0), 1)
                similar_boxes[zero_id] = 0
    output = torch.stack(output_box, dim=0)
    after_combine = output.size(0)
    if after_merge > after_combine and verbose:
        print("Combined %d boxes"%(after_merge - after_combine))
    return output



if __name__ == "__main__":
    box_num = 512
    square = 1024
    origin = torch.empty(box_num, 2).uniform_(0, 1)
    delta = torch.empty(box_num, 2).uniform_(0, 0.2)
    pred = torch.cat([origin, origin + delta], dim=1).clamp_(min=0, max=1).cuda()
    combinition = combine_boxes(pred, square, square)
    #print_box(combinition, shape=(square, square))
