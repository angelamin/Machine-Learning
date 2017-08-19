#!/usr/bin/env python
import sys
for line in sys.stdin:
    line = line.strip()
    words = line.split()
    for word in words:
        print "%s\t%s" % (word,1)

#用iterator优化

# import sys
# def read_input(file):
#     for line in file:
#         yield line.split()
#
# def main(separator='\t'):
#     data = read_input(sys.stdin)
#     for words in data:
#         for word in words:
#             print "%s%s%d" % (word,separator,1)
#
#
# if __name__ == "__main__":
#     main()
