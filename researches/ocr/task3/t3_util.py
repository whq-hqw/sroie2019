from os.path import *
import glob, json


def has_number(inputString):
    return any(char.isdigit() for char in inputString)


def is_number(inputString):
    return all(char.isdigit() for char in inputString)


def get_word_frequency(root_path):
    # get top 1000 frequent words from task1_2
    text_files = sorted(glob.glob(root_path + "/*.txt"))
    word_freq = {}
    print("Enumerating through %d files in %s"%(len(text_files), root_path))
    for i, text_file in enumerate(text_files):
        with open(text_file, "r") as txt_lines:
            for j, line in enumerate(txt_lines):
                line_element = line.split(",")
                text_label = ",".join(line_element[8:])
                words = text_label.strip().split(" ")
                for word in words:
                    if has_number(word):
                        continue
                    if word in word_freq:
                        word_freq[word] += 1
                    else:
                        word_freq.update({word: 1})
    print("word_freq for %s has %d keys."%(root_path, len(word_freq.keys())))
    return word_freq


def read_all_files_lines(root_path):
    # get top 1000 frequent words from task1_2
    text_files = sorted(glob.glob(root_path + "/*.txt"))
    line_list = []
    print("Enumerating through %d files in %s"%(len(text_files), root_path))
    for i, text_file in enumerate(text_files):
        with open(text_file, "r") as txt_lines:
            for j, line in enumerate(txt_lines):
                line_element = line.split(",")
                text_label = " ".join(line_element[8:])
                line_list.append(text_label)
    return line_list


def get_key_info(task_1_2_label_root, keys=("company", "address", "date"),
                 split_word=True):
    # get top 1000 frequent words from task1_2
    text_files = sorted(glob.glob(task_1_2_label_root + "/*.txt"))
    key_dict = []
    for i in range(len(keys)):
        key_dict.append({})
    print("Enumerating through %d files in %s"%(len(text_files), task_1_2_label_root))
    for j, key in enumerate(keys):
        for i, text_file in enumerate(text_files):
            with open(text_file, "r") as file:
                data = json.load(file)
                try:
                    words = data[key]
                except KeyError:
                    #print(data)
                    continue
                if split_word:
                    words = words.strip().split(" ")
                    for word in words:
                        if has_number(word):
                            continue
                        word = word.replace(".", "").replace("(", "").replace(")", "") \
                            .replace(",", "").replace("[", "").replace("]", "")
                        if word in key_dict[j]:
                            key_dict[j][word] += 1
                        else:
                            key_dict[j].update({word: 1})
                else:
                    word = words.strip()
                    if word in key_dict[j]:
                        key_dict[j][word] += 1
                    else:
                        key_dict[j].update({word: 1})
        print("%s_freq for task_1_2_text has %d keys." %
              (keys[j], len(key_dict[j].keys())))
    return key_dict

def get_bert_model_vocab(bert_root, bert_model):
    vocab_text_file = join(bert_root, bert_model, "vocab.txt")
    vocabulary = set([])
    with open(vocab_text_file, "r", encoding="utf-8") as vocab_lines:
        for i, line in enumerate(vocab_lines):
            if i <= 999:
                continue
            vocabulary.add(line.strip().lower())
    return vocabulary

def clean_existed_vocab(dict_list, vocabulary):
    output_dict = []
    for freq_dict in dict_list:
        length = len(freq_dict.keys())
        del_list = []
        for i, key in enumerate(freq_dict.keys()):
            if key.lower() in vocabulary:
                del_list.append(key)
        for key in del_list:
            freq_dict.pop(key)
        print("After cleaning, keys in dict reduced from %d to %d."%(length, len(freq_dict.keys())))
        output_dict.append(freq_dict)
    return output_dict

def compare_two_vocab(vocab_1, vocab_2):
    total_vocab = set(list(vocab_1.keys()) + list(vocab_2.keys()))
    vocab_1_unique = []
    vocab_2_unique = []
    intersect = []
    for word in total_vocab:
        if word in vocab_1 and word in vocab_2:
            intersect.append(word)
        if word not in vocab_1:
            vocab_2_unique.append(word)
        if word not in vocab_2:
            vocab_1_unique.append(word)
    return vocab_1_unique, vocab_2_unique, intersect

def combine_multi_vocabs(list_of_vocab):
    outut_vocab = {}
    for vocab in list_of_vocab:
        for key in vocab.keys():
            if key in outut_vocab:
                outut_vocab[key] += vocab[key]
            else:
                outut_vocab.update({key: vocab[key]})
    return outut_vocab


if __name__ is "__main__":
    bert_root = expanduser("~/Documents/bert")
    bert_model = "uncased_L-12_H-768_A-12"
    task_1_2_text_root = expanduser("~/Pictures/dataset/ocr/SROIE2019")
    task_1_2_label_root = expanduser("~/Downloads/task_1_2_label")
    task_3_text_root = expanduser("~/Downloads/task_3_label")

    vocabulary = get_bert_model_vocab()
    task_1_2_vocab = get_word_frequency(task_1_2_text_root)
    task_3_vocab = get_word_frequency(task_3_text_root)
    cleaned_vocab = clean_existed_vocab([task_1_2_vocab, task_3_vocab], vocabulary)
    task_1_2_unique, task_3_unique, intersect = compare_two_vocab(cleaned_vocab[0], cleaned_vocab[1])
    print("Word only exist in task_1_2 are: %d"%(len(task_1_2_unique)))
    print("Word only exist in task_3 are: %d" % (len(task_3_unique)))
    print("Intersection in both vocab is: %d" % (len(intersect)))
    combined_vocab = combine_multi_vocabs([task_1_2_vocab, task_3_vocab])
    cleaned_vocab = clean_existed_vocab([combined_vocab], vocabulary)[0]


    frequency_com = sorted(cleaned_vocab.items(), key=lambda kv: (kv[1], kv[0]), reverse=False)
    print("Finish")
    #company_freq, address_freq = get_task_1_2_key_info()
    #display_uncontained_word_freq([task_1_2_vocab])