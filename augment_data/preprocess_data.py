import re
import csv
import pandas as pd
from sklearn.model_selection import train_test_split


correct_mapping = {
    "ship": "vận chuyển",
    "shop": "cửa hàng",
    "m": "mình",
    "mik": "mình",
    "ko": "không",
    "k": " không ",
    "kh": "không",
    "khong": "không",
    "kg": "không",
    "khg": "không",
    "tl": "trả lời",
    "r": "rồi",
    "fb": "mạng xã hội", # facebook
    "face": "mạng xã hội",
    "thanks": "cảm ơn",
    "thank": "cảm ơn",
    "tks": "cảm ơn",
    "tk": "cảm ơn",
    "ok": "tốt",
    "dc": "được",
    "vs": "với",
    "đt": "điện thoại",
    "thjk": "thích",
    "qá": "quá",
    "trể": "trễ",
    "bgjo": "bao giờ"
}


def tokmap(tok):
    if tok.lower() in correct_mapping:
        return correct_mapping[tok.lower()]
    else:
        return tok


def preprocess(review):
    tokens = review.split()
    tokens = map(tokmap, tokens)
    return " ".join(tokens)


def load_data(filepath, is_train=True):
    regex = 'train_'
    if not is_train:
        regex = 'test_'

    a = []
    b = []

    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if regex in line:
                b.append(a)
                a = [line]
            elif line != "":
                a.append(line)

        b.append(a)

    b = b[1:]
    lst = []
    for tp in b:
        idx = tp[0]
        if is_train:
            lb = int(tp.pop(-1))
        else:
            lb = "0"
        review = " ".join(tp[1:])
        review = re.sub(r"^\"*", "", review)
        review = re.sub(r"\"*$", "", review)
        review_ = preprocess(review)
        lst.append([idx, review, review_, lb])
    return lst


if __name__ == "__main__":
    TRAIN_FILE = "./train.crash.txt"
    TRAIN_CSV = "./train_augment.csv"
    train_data = load_data(TRAIN_FILE)
    print("# Loaded training samples: {}".format(len(train_data)))

    cols = ["id", "text", "text_ws", "label"]
    df_train = pd.DataFrame(data=train_data, columns=cols)

    df_train = df_train[['label', 'text_ws']]
    df_train.rename({'label': 'Class', 'text_ws': 'Data'}, axis=1, inplace=True)
    df_train['Class'] = df_train['Class'].map({0: 1, 1: -1})
    df_train.to_csv(TRAIN_CSV, index=False)
