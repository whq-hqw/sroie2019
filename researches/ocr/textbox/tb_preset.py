def GeneralPattern(args):
    args.path = "~/Downloads/dataset/ocr"
    # this will create a folder named "_text_detection" under "~/Pictures/dataset/ocr"
    args.code_name = "_text_detection"
    # Set it to True to make experiment result reproducible
    args.deterministic_train = False
    args.cudnn_benchmark = False
    # Random seed for everything
    # If deterministic_train is disabled, then it will have no meaning
    args.seed = 1
    # Training Hyperparameter
    args.learning_rate = 1e-4
    args.batch_size_per_gpu = 1
    args.loading_threads = 2
    args.img_channel = 3
    args.epoch_num = 2000
    args.finetune = True

    # Because augmentation operation is defined in tb_augment.py
    args.do_imgaug = False

    # Image Normalization
    args.img_mean = (0.5, 0.5, 0.5)
    args.img_std = (1.0, 1.0, 1.0)
    args.img_bias = (0.0, 0.0, 0.0)
    return args

def Unique_Patterns(args):
    args.train_sources = ["SROIE2019"]
    args.train_aux = [{"txt": "txt", "img": "jpg"}]
    args.fix_size = True
    return args


def Runtime_Patterns(args):
    args.model_prefix_finetune = "768",
    args.model_prefix = "768",
    return args


PRESET = {
    "general": GeneralPattern,
    "unique": Unique_Patterns,
    "runtime": Runtime_Patterns,
}