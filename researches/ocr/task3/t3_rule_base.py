import imgaug
from imgaug import augmenters
from researches.ocr.task3.t3_util import *
import torch

month = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN",
         "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
price_char = set(['0', 'R', '1', '3', ' ', '7', '.', '5', 'M', '2', ',', '8', '-', '4', '6', '$', '9'])

def is_date(text):
    def date_judge(split):
        if any([char in [",", "."] for char in split[1]]):
            return False
        if is_number(split[0][-2:]) and is_number(split[-1][:2]):
            pass
        else:
            return False
        if split[1].upper() in month:
            pass
        elif is_number(split[1]):
            if int(split[1]) > 12:
                # Month can not be larger than 12
                return False
            pass
        else:
            return False
        return True
        
    ind_text = text.replace("[", "").replace("]", "").replace(":", "") \
        .replace(" ", "")
    extra_indicator = len(ind_text.split("/")) == 3 or \
                      len(ind_text.split("-")) == 3
    if extra_indicator:
        if len(ind_text.split("/")) == 3:
            split = ind_text.split("/")
            slash_is_date = date_judge(split)
        else:
            slash_is_date = False
        if len(ind_text.split("-")) == 3:
            split = ind_text.split("-")
            hyphen_is_date = date_judge(split)
        else:
            hyphen_is_date = False
        if hyphen_is_date or slash_is_date:
            pass
        else:
            return False
        if len(ind_text.split("-")) > 1:
            splited = ind_text.split("-")
            # make sure telphone number xx-xxxx-xxxx will be filtered out
            if all([num.isdigit() for num in splited[1]]) and len(splited[1]) == 4:
                return False
    dot_split = ind_text.split(".")
    if len(dot_split) == 3 and len(dot_split[1]) > 1:
        if all([char.isdigit() for char in dot_split[1]]) \
                or dot_split[1].upper() in month:
            extra_indicator = True
    space_split = ind_text.split(" ")
    if any([split.upper() in month for split in space_split]):
        extra_indicator = True
    return extra_indicator


def crop_number_from_str(text):
    num_id = [i for i, char in enumerate(text) if char.isdigit()]
    start, end = min(num_id), max(num_id) + 1
    if start > 0 and  text[start - 1] == "-":
        start -= 1
    return start, end


def number_form(text):
    if type(text) is str:
        start, end = crop_number_from_str(text)
        return float(text[start: end])
    else:
        return text
    
    
def format_price(text):
    if type(text) in [float, int]:
        text = str(text)
    idx = text.rfind(".")
    if idx == -1:
        # cannot find "." in the string
        text += ".00"
    else:
        if idx + 1 == len(text):
            # "." is the last character of the string
            text += "00"
        elif idx + 2 == len(text):
            # "XXX.X" was the string format
            text += "0"
        elif idx + 3 < len(text):
            # "XXX.XXX..." was the string format
            text = text[:idx+3]
    return text
    

def price_form(text, price):
    start, end = crop_number_from_str(text)
    new_text = text[:start] + str(price) + text[end:]
    return new_text


def calculate_price(price_type, ver_coord_price, height_thres):
    total_tensor = torch.Tensor(price_type).unsqueeze(1).repeat(1, len(ver_coord_price), 1)
    coords_tensor = torch.Tensor(ver_coord_price).unsqueeze(0).repeat(len(price_type), 1, 1)
    diss_total = torch.abs(coords_tensor - total_tensor)
    coord = (diss_total[:, :, 1] < height_thres).nonzero()[:, 1]
    return coord


def is_price(text):
    if any([char not in price_char for char in text]):
        return False
    if not has_number(text):
        return False
    start, end = crop_number_from_str(text)
    if end - start > 6:
        return False
    try:
        num = float(text[start: end])
    except ValueError:
        return False
    if num > 2000:
        return False
    return True


def get_max_price(price_list):
    max_value = 0.0
    max_id = 0
    for i, price in enumerate(price_list):
        if not is_price(price):
            continue
        start, end = crop_number_from_str(price)
        if float(price[start: end]) > max_value:
            max_id = i
            max_value = float(price[start: end])
    return price_list[max_id]


def get_min_price(price_list):
    min_value = 99999999.9
    min_id = 0
    for i, price in enumerate(price_list):
        if not is_price(price):
            continue
        start, end = crop_number_from_str(price)
        if float(price[start: end]) < min_value:
            min_id = i
            min_value = float(price[start: end])
    return price_list[min_id]


def correct_date(text):
    def get_year(_split):
        if len(_split[2]) >= 4 and is_number(_split[2][:4]):
            year = _split[2][:4]
        elif len(_split[2]) < 4 and is_number(_split[2][:2]):
            year = _split[2][:2]
        else:
            if len(_split[2].split(" ")) > 1:
                if is_number(_split[2].split(" ")[0]):
                    year = _split[2].split(" ")[0]
                else:
                    year = ""
            else:
                year = ""
        return year
    if len(text.split("/")) == 3:
        split = text.split("/")
        year = get_year(split)
        return "/".join([split[0][-2:], split[1], year])
    elif len(text.split("-")) == 3:
        split = text.split("-")
        year = get_year(split)
        return "-".join([split[0][-2:], split[1], year])
    elif len(text.split(".")) >= 3:
        split = text.split(".")
        year = get_year(split)
        return ".".join([split[0][-2:], split[1], year])
    elif len(text.split(" ")) >= 3:
        split = text.split(" ")
        for i, string in enumerate(split):
            if string.upper() in month:
                return " ".join([split[i - 1], split[i], split[i + 1]])
        return text
    else:
        print("Invalid date format: %s" % text)
        return text


def correct_company(text):
    # Eliminate the (xxxxxxxxx) at the end
    text = text.replace(",", "").replace("  ", "")
    text = text.strip(" ")
    if len(text.split("(")) > 1:
        latter = text.split("(")[-1]
        if len(latter.split(" ")) == 1:
            # means (xxxxxxxxx) at the end
            return text.split("(")[0].strip(" ")
        else:
            return text
    else:
        return text


def get_height_of_line(text):
    coord = text.strip().split(",")[:8]
    try:
        coord = [int(c) for c in coord]
    except ValueError:
        x=0
    #x1, x2 = min(coord[::2]), max(coord[::2])
    y1, y2 = min(coord[1::2]), max(coord[1::2])
    return y1, y2


def rotate_coords(text_lines, angle, h, w):
    rotation = angle * 90
    rot_aug = augmenters.Affine(rotate=rotation)
    coords = []
    labels = []
    for line in text_lines:
        coord = line.strip().split(",")[:8]
        coord = [int(c) for c in coord]
        coords.append(coord)
        labels.append(",".join(line.strip().split(",")[8:]))
    BBox = []
    for coord in coords:
        x1, x2 = min(coord[::2]), max(coord[::2])
        y1, y2 = min(coord[1::2]), max(coord[1::2])
        if abs(x2 - x1) * abs(y2 - y1) <= 0.005 * h * w / 100:
            # Skip a bbox which is smaller than a certain percentage of the total size
            continue
        BBox.append(imgaug.imgaug.BoundingBox(x1, y1, x2, y2))
    BBoxes = imgaug.imgaug.BoundingBoxesOnImage(BBox, shape=(h, w))
    bbox = rot_aug.augment_bounding_boxes([BBoxes])[0]
    new_text_lines = []
    for i, box in enumerate(bbox.bounding_boxes):
        x1, y1, x2, y2 = int(round(box.x1)), int(round(box.y1)), int(round(box.x2)), int(round(box.y2))
        # box_tensors.append(torch.tensor([x1, y1, x2, y2]))
        # 4-point to 8-point: x1, y1, x2, y1, x2, y2, x1, y2
        coord = "%d,%d,%d,%d,%d,%d,%d,%d," % (x1, y1, x2, y1, x2, y2, x1, y2)
        new_text_lines.append(coord + labels[i] + "\n")
    return new_text_lines


if __name__ == "__main__":
    print(is_date('RC22-32 - 36'))