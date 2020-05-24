import os, glob, argparse
import cv2, torch
from os.path import *
import numpy as np
import scipy.io as sio
from researches.ocr.task3.t3_make_data import make_data
from researches.ocr.task3.t3_rule_base import *
from researches.ocr.textbox.tb_preprocess import detect_angle, rotate_image


def parse_arguments():
    parser = argparse.ArgumentParser(description='Task 3 settings')
    parser.add_argument(
        "--make_data",
        action="store_true"
    )
    parser.add_argument(
        "--do_predict",
        action="store_true",
    )
    parser.add_argument(
        "-br",
        "--bert_root",
        type = str,
        default = "~/Documents/bert"
    )
    parser.add_argument(
        "-bm",
        "--bert_model",
        type=str,
        default="uncased_L-12_H-768_A-12"
    )
    parser.add_argument(
        "-tef",
        "--test_folder",
        type=str,
        default="~/Pictures/dataset/ocr/SROIE2019_test_t3"
    )
    parser.add_argument(
        "-tet",
        "--test_text",
        type=str,
        default="~/Downloads/task_3_label"
    )
    parser.add_argument(
        "-trt",
        "--train_text",
        type=str,
        default="~/Pictures/dataset/ocr/SROIE2019"
    )
    parser.add_argument(
        "-trki",
        "--train_key_info",
        type=str,
        default="~/Downloads/task_1_2_label"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="~/Downloads/task_3_result"
    )
    parser.add_argument(
        "--sequence",
        nargs='+',
        help="a list folder/folders to use as training set",
        default=["address", "company", "date"]
    )
    args = parser.parse_args()
    return args


def do_predict(bert_root):
    os.chdir(bert_root)
    unchange_command = "python3 run_classifier.py --task_name=CoLA --do_predict=true " \
                       "--max_seq_length=128 --train_batch_size=32 --learning_rate=2e-5 " \
                       "--num_train_epochs=3.0"
    vocab_file = "--vocab_file=%s"%(join(bert_root, bert_model, "vocab.txt"))
    config_file = "-bert_config_file=%s"%(join(bert_root, bert_model, "bert_config.json"))
    checkpoint = "--init_checkpoint=%s"%(join(bert_root, bert_model, "bert_model.ckpt"))
    
    # perform prediction using BERT's CoLA Mode
    for task in args.sequence:
        data_dir = "--data_dir=%s/sroie_%s"%(bert_root, task)
        out_dir = "--output_dir=%s/tmp/cola_%s" % (bert_root, task)
        bert_predict_command = " ".join([unchange_command, vocab_file, config_file, checkpoint, data_dir, out_dir])
        os.system(bert_predict_command)
    
    print("##################################")
    print("#                                #")
    print("#      PREDICTION COMPLETED      #")
    print("#                                #")
    print("#   generating the json file...  #")
    print("#                                #")
    print("##################################")



args = parse_arguments()
if __name__ == "__main__":
    bert_root = expanduser(args.bert_root)
    bert_model = args.bert_model
    task_1_2_text_root = expanduser(args.train_text)
    task_1_2_label_root = expanduser(args.train_key_info)
    task_3_text = expanduser(args.test_text)
    task_3_img_root = expanduser(args.test_folder)
    output_dir = expanduser(args.output_dir)
    if not exists(output_dir):
        os.mkdir(output_dir)
        
    month = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN",
             "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
    company_key = set(["SDN", "SDN.", "BHD", "BHD.", "S/B", "S/B.", "S./B."])
    un_company_key = set(["TAX", "RECIEPT", "INVOICE"])
    
    # generate data
    if args.make_data:
        make_data(task_1_2_text_root, task_1_2_label_root, task_3_text, bert_root)
    
    # Prepare command for test to run
    if args.do_predict:
        do_predict(bert_root)
    
    # Get line number from ground truth text label of test data
    num_list = []
    all_txt_lines = []
    size_dict = {}
    size_dict_exist = exists(join(task_3_text, "size_dict.json"))
    if size_dict_exist:
        print("size_dict loaded from %s" % join(task_3_text, "size_dict.json"))
        size_dict = json.load(open(join(task_3_text, "size_dict.json"), "r"))
    text_files = sorted(glob.glob(task_3_text + "/*.txt"))
    for i, text_file in enumerate(text_files):
        name = text_file[text_file.rfind("/") + 1 : -4]
        img_path = join(task_3_img_root, name + ".jpg")
        if exists(img_path) and not size_dict_exist:
            size = cv2.imread(img_path, 0).shape
            size_dict.update({name: size})
        txt_lines = open(text_file, "r").readlines()
        num_list.append(len(txt_lines))
        all_txt_lines.append(txt_lines)
    if not size_dict_exist:
        with open(join(task_3_text, "size_dict.json"), "w") as file:
            json.dump(size_dict, file)
        print("size_dict saved to %s" % join(task_3_text, "size_dict.json"))
    
    # Get the test result by BERT
    output = {}
    for task in args.sequence:
        output_file = join(bert_root, "tmp", "cola_%s"%task, "test_results.tsv")
        output_lines = open(output_file, "r").readlines()
        assert len(output_lines) == sum(num_list)
        output.update({task: output_lines})
    
    # Get word frequency for company key info from train data
    company_set = get_key_info(task_1_2_label_root, keys=["company"],
                               split_word=True)[0]
    sort_dict = sorted(company_set.items(), key=lambda kv: (kv[1], kv[0]),
                       reverse=False)
    company_set = set([pair[0] for pair in sort_dict[300:]])
    
    # Generate JSON file
    start = 0
    for i, num in enumerate(num_list):
        text_file_name = text_files[i][text_files[i].rfind("/") + 1 : -4]
        if text_file_name not in size_dict:
            start += num
            # this sample does not exist in the task 3 test set
            continue
        print("%d: %s"%(i, text_file_name))
        if i in [82]:
            print("problem!")
        #print("%d-th reciept:"%i)
        tmp_lines = [[[float(v) for v in val.strip().split("\t")]
                      for val in output[task][start : start + num]]
                     for task in args.sequence]
        idx = [np.argmax(np.asarray(lines), axis=1) for lines in tmp_lines]
        result = {}
        for j, task in enumerate(args.sequence):
            #print("Task: %s matched %d."%(task, np.sum(idx[j])))
            #if task == "company" and i > 81:
                # force the algorithm to use rule-based method to find company name
                #idx[j] = [0]
            if np.sum(idx[j]) == 0:
                # Use rule-based method
                if task == "date":
                    for text in all_txt_lines[i]:
                        _text = ",".join(text.strip().split(",")[8:])
                        if is_date(_text):
                            if "date" in result:
                                print("Another date: %s" % _text)
                            else:
                                result.update({"date": correct_date(_text)})
                    if "date" not in result:
                        # Split by dot
                        for text in all_txt_lines[i]:
                            dot_split = ",".join(text.strip().split(",")[8:]).split(".")
                            if len(dot_split) == 3 and len(dot_split[1]) > 1:
                                if all([char.isdigit() for char in dot_split[1]]) \
                                        or dot_split[1].upper() in month:
                                    date = correct_date(",".join(text.strip().split(",")[8:]))
                                    result.update({"date": date})
                    if "date" not in result:
                        # split by space
                        for text in all_txt_lines[i]:
                            space_split = ",".join(text.strip().split(",")[8:]).split(" ")
                            if any([split.upper() in month for split in space_split]):
                                date = correct_date(",".join(text.strip().split(",")[8:]))
                                result.update({"date": date})
                                #result.update({"date": ",".join(text.strip().split(",")[8:])})
                    if "date" not in result:
                        print("%s: Nothing Detected" % (task))
                elif task == "company":
                    # has 81.83% accuracy that the first line of the reciept
                    # is the company name according to train set
                    for line_num in range(5):
                        cmpy_name = ",".join(all_txt_lines[i][line_num].strip().split(",")[8:])
                        if is_number(cmpy_name):
                            # The first line is a number
                            continue
                        elif any([word in un_company_key for word in cmpy_name]):
                            continue
                        else:
                            result.update({"company": correct_company(cmpy_name)})
                            break
                else:
                    print("%s: Nothing Detected" % (task))
            else:
                # Use the prediction result of BERT
                company_names = []
                for k, indicator in enumerate(idx[j]):
                    key_info = ",".join(all_txt_lines[i][k].strip().split(",")[8:])
                    if indicator == 1:
                        if task == "company":
                            cmpy_name = correct_company(key_info)
                            if "&" in cmpy_name[-2:]:
                                cmpy_name += (" " + ",".join(all_txt_lines[i][k+1].strip().split(",")[8:]))
                            company_names.append(cmpy_name)
                            continue
                        if task in result:
                            if task == "date":
                                if is_date(key_info):
                                    print("Another date: %s"%key_info)
                                pass
                            else:
                                result[task] += (" " + key_info)
                        else:
                            if task == "date":
                                if not is_date(key_info):
                                    continue
                                _key_info = correct_date(key_info)
                            else:
                                _key_info = key_info
                            result.update({task: _key_info})
                for cmpy_name in company_names:
                    hit = sum([1 for word in cmpy_name.split(" ") if word in company_key])
                    if len(cmpy_name.split(" ")) > hit and hit > 0:
                        result.update({"company": cmpy_name})
                        break
                if "company" not in result and task == "company":
                    result.update({"company": company_names[0]})
        if "date" not in result:
            for text in all_txt_lines[i]:
                _text = ",".join(text.strip().split(",")[8:])
                if is_date(_text):
                    if "date" in result:
                        print("Another date: %s" % _text)
                    else:
                        result.update({"date": correct_date(_text)})
        if "address" not in result:
            print("BAD ADDRESS")
            result.update({"address": " ".join([",".join(text.strip().split(",")[8:]) for text in all_txt_lines[i][1:4]])})
        # Get total price:
        number_with_total = []
        ver_coords_total = []
        ver_coord_price = []
        ver_coords_rounding = []
        ver_coords_cash = []
        ver_coords_change = []
        height, width = size_dict[text_file_name]
        price = []
        # 1% of the total height
        height_thres = height / 100
        img = cv2.imread(join(task_3_img_root, text_file_name + ".jpg"))
        angle = detect_angle(img)
        if angle is not None and abs(angle) * 90 > 1:
            #img = rotate_image(img, angle)
            new_text_lines = rotate_coords(all_txt_lines[i], angle, height, width)
        else:
            new_text_lines = all_txt_lines[i]
        for line in new_text_lines:
            if len(line.strip().split(",")) < 9:
                print("Error in lines!!!!!!!!!!!!!!")
                continue
            y1, y2 = get_height_of_line(line)
            height, h_coord = (y2 - y1), (y1 + y2) / 2
            if "TOTAL" in line and "QTY" not in line and "QUANTITY" not in line:
                ver_coords_total.append([height, h_coord])
            elif "ROUND" in line:
                ver_coords_rounding.append([height, h_coord])
            elif "CASH" in line or "PAY" in line or "PAID" in line:
                ver_coords_cash.append([height, h_coord])
            elif "CHANG" in line:
                # Sometime changes, change, changing
                ver_coords_change.append([height, h_coord])
            else:
                if is_price(",".join(line.strip().split(",")[8:])):
                    ver_coord_price.append([height, h_coord])
                    price.append(",".join(line.strip().split(",")[8:]))
        # Convert toi np array to do the calculation
        if len(ver_coords_total) > 0:
            total_coord = calculate_price(ver_coords_total, ver_coord_price, height_thres)
            # Calculate Total
            if len(total_coord) > 0:
                total_price = get_max_price([price[int(price_idx)] for price_idx in total_coord])
            else:
                total_price = get_max_price(price)
        else:
            total_price = get_max_price(price)
        # Calculate Cash
        if len(ver_coords_cash):
            cash_coord = calculate_price(ver_coords_cash, ver_coord_price, height_thres)
            if len(cash_coord) > 0:
                total_cash = get_max_price([price[int(price_idx)] for price_idx in cash_coord])
            else:
                total_cash = 0
        else:
            total_cash = 0
        # Calculate Change
        if len(ver_coords_change):
            change_coord = calculate_price(ver_coords_change, ver_coord_price, height_thres)
            if len(change_coord) > 0:
                total_change = get_max_price([price[int(price_idx)] for price_idx in change_coord])
            else:
                total_change = 0
        else:
            total_change = 0
        # Calculate Change
        if len(ver_coords_rounding) > 0:
            round_coord = calculate_price(ver_coords_rounding, ver_coord_price, height_thres)
            if len(round_coord) > 0:
                round_price = get_max_price([price[int(price_idx)] for price_idx in round_coord])
                round_adj = get_min_price([price[int(price_idx)] for price_idx in round_coord])
            else:
                round_price, round_adj = 0, 0
        else:
            round_price, round_adj = 0, 0
        
        # Calculate final price
        calculated_total = number_form(total_cash) - number_form(total_change)
        if calculated_total > 0 and abs(calculated_total - number_form(total_price)) < 0.1 and len(ver_coords_rounding) > 0:
            total_price = price_form(total_price, calculated_total)
        elif len(ver_coords_total) > 4 and len(total_coord) > 5 and calculated_total < number_form((total_price)):
            total_price = calculated_total
        
        result.update({"total": format_price(total_price)})
        final_txt = ["{\n"]
        for key in ["company", "date", "address", "total"]:
            try:
                content = result[key]
            except KeyError:
                content = " "
            if key == "total":
                final_txt.append('    "%s": "%s"\n' % (key, content))
            else:
                final_txt.append('    "%s": "%s",\n' % (key, content))
            print('\t%s: %s' % (key, content))
        final_txt.append("}\n")
        print("")
        
        # Write to file in JSON format
        with open(join(output_dir, text_file_name + ".txt"), "w") as file:
            for txt in final_txt:
                file.write(txt)
        file.close()
        #json.dump(result, file)
        start += num