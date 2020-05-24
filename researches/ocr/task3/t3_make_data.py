from os.path import *
import os, glob, json
import distance

from researches.ocr.task3.t3_util import *

def read_lines_from_txtfile(text_file):
    lines = []
    with open(text_file, "r") as txt_lines:
        for j, line in enumerate(txt_lines):
            line_element = line.split(",")
            text_label = ",".join(line_element[8:])
            lines.append(text_label.strip())
    return lines


def read_line_from_json(json_file):
    with open(json_file, "r") as file:
        data = json.load(file)
        comp = data["company"].strip()
        date = data["date"].strip()
        try:
            addr = data["address"].strip()
        except KeyError:
            print(data)
            return comp, '', ''
    return comp, addr, date


def create_bert_train_tsv(text_root, json_root, bert_root, task_for="company"):
    file_names = [file for file in os.listdir(json_root) if file.endswith("txt")]
    if not exists(join(bert_root, "sroie_%s"%(task_for))):
        os.mkdir(join(bert_root, "sroie_%s"%(task_for)))
    train = open(join(bert_root, "sroie_%s"%(task_for), "train.tsv"), "w")
    val = open(join(bert_root, "sroie_%s"%(task_for),  "dev.tsv"), "w")
    samples = len(file_names)
    write_lines = 0
    positive_samples = 0
    negative_samples = 0
    #text_file = get_task_word_freq(text_root)
    comp, addr, date = get_key_info(json_root, split_word=False)
    for i, file_name in enumerate(file_names):
        if i > round(samples * 0.9):
            train.close()
            write = val
        else:
            write = train
        if not exists(join(text_root, file_name)):
            continue
        text_file = read_lines_from_txtfile(join(text_root, file_name))
        #comp, addr, date = read_line_from_json(join(json_root, file_name))
        if task_for == "company":
            criterion = comp
        elif task_for == "address":
            criterion = addr
        elif task_for == "date":
            criterion = date
        else:
            raise TypeError("task_for parameter should be one of three options 'address', 'company' or 'date'")
        for j, text in enumerate(text_file):
            basic_indicator = any([True if text in key_info else False
                                   for key_info in criterion.keys()])
            if task_for == "date":
                ind_text = text.replace("[", "").replace("]", "").replace(":", "").replace(" ", "")
                extra_indicator = len(ind_text.split("/")) == 3 or \
                                  len(ind_text.split("-")) == 3
                if extra_indicator:
                    is_date_slash = sum([1 for t in ind_text.split("/") if t.isdigit()])
                    is_date_hyphen = sum([1 for t in ind_text.split("-") if t.isdigit()])
                    if is_date_slash < 2 and is_date_hyphen < 2:
                        extra_indicator = False
                true_state = extra_indicator or (basic_indicator and extra_indicator)
            else:
                extra_indicator = len(text.strip(" ").split(" ")) == 1 or \
                                   len(text.strip(" ")) <= 4
                true_state = not extra_indicator and basic_indicator
            if true_state:
            #if text in criterion:
                #label = 1
                write.write("data%d\t1\t \t%s\n" % (write_lines, text))
                positive_samples += 1
            else:
                #label = 0
                write.write("data%d\t0\t*\t%s\n" % (write_lines, text))
                negative_samples += 1
            #print("data%d  %d  *  %s" % (write_lines, label, text))
            #write.write("data%d\t%d\t*\t%s\n" % (write_lines, label, text))
            write_lines += 1
    print("Total %s samples contain %d positive and %d negative samples."
          % (write_lines, positive_samples, negative_samples))
    print("")
    val.close()


def create_bert_test_tsv(text_root, bert_root, task_for):
    test = open(join(bert_root, "sroie_%s" % (task_for), "test.tsv"), "w")
    task_3_text_lines = read_all_files_lines(text_root)
    test.write("index\tsentence\n")
    for i, line in enumerate(task_3_text_lines):
        test.write("%d\t%s"%(i, line))
    test.close()
    print("tsv file created for %s."%(task_for))


def make_data(task_1_2_text_root, task_1_2_label_root, task_3_text_root, bert_root):
    for task in ["company", "address", "date"]:
        create_bert_train_tsv(task_1_2_text_root, task_1_2_label_root, bert_root, task_for=task)
        create_bert_test_tsv(task_3_text_root, bert_root, task_for=task)

if __name__ is "__main__":
    bert_root = expanduser("~/Documents/bert")
    bert_model = "uncased_L-12_H-768_A-12"
    task_1_2_text_root = expanduser("~/Pictures/dataset/ocr/SROIE2019")
    task_1_2_label_root = expanduser("~/Downloads/task_1_2_label")
    task_3_text_root = expanduser("~/Downloads/task_3_label")
    make_data(task_1_2_text_root, task_1_2_label_root, task_3_text_root, bert_root)
