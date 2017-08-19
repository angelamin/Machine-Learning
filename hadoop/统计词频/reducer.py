#!/usr/bin/env python
from operator import itemgetter
import sys
current_word = None
current_count = 0
word = None

for line in sys.stdin:
    line = line.strip()
    word,count = line.split('\t',1)
    try:
        count = int(count)
    except ValueError:
        continue
    if current_word == word:
        current_count += count
    else:
        if current_word:
            print "%s\t%s" % (current_word,current_count)
        current_count = count
        current_word = word

if word == current_word:
    print "%s\t%s" % (current_word,current_count)


#优化
# from operator import itemgetter
# from itertools import groupby
# import sys
#
# def read_mapper_output(file,separator='\t'):
#     for line in file:
#         yield line.rstrip().split(separator,1)
#
# def main(separator='\t'):
#     data = read_mapper_output(sys.stdin,separator=separator)
#     for current_word,group in groupby(data,itemgetter(0)):
#         try:
#             total_count = sum(int(count) for current_word,count in group)
#             print "%s%s%d" % (current_word,separator,total_count)
#         except ValueError
#             pass
#
# if __name__ == "__main__":
#     main()
