#coding:utf-8
from data_helpers import load_data_and_labels
positive_data_file = './mail_data/data_positive_str.txt'
negative_data_file = './mail_data/data_negative.txt'
positive_examples = list(open(positive_data_file,"r").readlines())
# for s in positive_examples:
#     print(s)

load_data_and_labels(positive_data_file, negative_data_file)
