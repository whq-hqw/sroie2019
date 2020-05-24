import glob, os, json
from os.path import *
from researches.ocr.task3.t3_util import *

bert_root = expanduser("~/Documents/bert")
bert_model = "uncased_L-12_H-768_A-12"

task_1_2_text_root = expanduser("~/Pictures/dataset/ocr/SROIE2019")
task_1_2_label_root = expanduser("~/Downloads/task_1_2_label")
task_3_text_root = expanduser("~/Downloads/task_3_label")
month = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN",
         "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]

# Compare the vocabulary of train and test


# which line of the text info are company key info
text_files = sorted(glob.glob(task_1_2_text_root + "/*.txt"))
statistics = {}
char_list = set([])
for idx, text_file in enumerate(text_files):
    if idx == 81:
        x = 0
    name = text_file[text_file.rfind("/") + 1 :]
    key_info_file = join(task_1_2_label_root, name)
    try:
        key_info = json.load(open(key_info_file, "r"))
    except FileNotFoundError:
        continue
    company = key_info["company"]
    total = key_info["total"]
    print(total)
    for char in total:
        char_list.add(char)
    date = key_info["date"]
    text_lines = open(text_file, "r").readlines()
    for i, line in enumerate(text_lines):
        if any([m in date.upper() for m in month]):
            #print(name)
            pass
        if company in line:
            if str(i) not in statistics:
                statistics.update({str(i):1})
            else:
                statistics[str(i)] += 1
            continue
for key in sorted(statistics.keys()):
    print("%s => %d"%(key, statistics[key]))
print(statistics["0"]/sum(statistics[key] for key in statistics.keys()))
print(char_list)
