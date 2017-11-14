# -*- coding: UTF-8 -*-
from sklearn.naive_bayes import MultinomialNB
import matplotlib.pyplot as plt
import os
import random
import jieba
import cPickle
"""
函数说明:中文文本处理，，得到训练集、测试集以及所有词排序结果
Parameters:
	folder_path - 文本存放的路径
	test_size - 测试集占比，默认占所有数据集的百分之20
Returns:
	all_words_list - 按词频降序排序的训练集列表
	train_data_list - 训练集列表
	test_data_list - 测试集列表
	train_class_list - 训练集标签列表
	test_class_list - 测试集标签列表
"""
def TextProcessing(folder_path, test_size = 0.2):
	print('处理样本数据.......')
	folder_list = os.listdir(folder_path)						#查看folder_path下的文件
	data_list = []												#数据集数据
	class_list = []												#数据集类别
	batch_tmp = []
	files = os.listdir(folder_path)
	for file in files:
		file_path = os.path.join(folder_path,file)
		fin = open(file_path,'r')
		i = 0
		for raw in fin:
			i += 1
			raw = raw.strip()
			raw = raw.replace(' ','')
			word_cut = jieba.cut(raw,cut_all = False)
			word_list = list(word_cut)
			batch_tmp.extend(word_list)
			if i%1 == 0:  #对一行进行处理
				data_list.append(batch_tmp)						#添加数据集数据
				if file == 'ad':
					class_list.append('ad')							#添加数据集类别
				else:
					class_list.append('not_ad')
				batch_tmp = []

		if len(batch_tmp) != 0:
			data_list.append(batch_tmp)						#添加数据集数据
			if file == 'ad':
				class_list.append('ad')							#添加数据集类别
			else:
				class_list.append('not_ad')
			batch_tmp = []

	data_class_list = list(zip(data_list, class_list))			#zip压缩合并，将数据与标签对应压缩
	random.shuffle(data_class_list)								#将data_class_list乱序
	# print(len(data_class_list))	 #2								#将data_class_list乱序
	index = int(len(data_class_list) * test_size) + 1			#训练集和测试集切分的索引值
	# print(index)    #1
	train_list = data_class_list[index:]						#训练集
	test_list = data_class_list[:index]							#测试集
	train_data_list, train_class_list = zip(*train_list)		#训练集解压缩
	test_data_list, test_class_list = zip(*test_list)			#测试集解压缩

	all_words_dict = {}											#统计训练集词频
	for word_list in train_data_list:
		for word in word_list:
			if word in all_words_dict.keys():
				all_words_dict[word] += 1
			else:
				all_words_dict[word] = 1

	#根据键的值倒序排序
	all_words_tuple_list = sorted(all_words_dict.items(), key = lambda f:f[1], reverse = True)
	all_words_list, all_words_nums = zip(*all_words_tuple_list)	#解压缩
	all_words_list = list(all_words_list)						#转换成列表

	#保存所有词汇
	all_words_list_file = open('./process_files/all_words_list','w')
	for word in all_words_list:
		all_words_list_file.write(word.encode('utf-8'))
		all_words_list_file.write('\n')

	return all_words_list, train_data_list, test_data_list, train_class_list, test_class_list

"""
函数说明:读取文件里的内容，并去重
Parameters:
	words_file - 文件路径
Returns:
	words_set - 读取的内容的set集合
"""
def MakeWordsSet(words_file):
	print('读取文件内容.....')
	words_set = set()											#创建set集合
	with open(words_file, 'r') as f:		#打开文件
		for line in f.readlines():								#一行一行读取
			word = line.strip()									#去回车
			if len(word) > 0:									#有文本，则添加到words_set中
				words_set.add(word)
	return words_set 											#返回处理结果

"""
函数说明:根据feature_words将文本向量化
Parameters:
	train_data_list - 训练集
	test_data_list - 测试集
	feature_words - 特征集
Returns:
	train_feature_list - 训练集向量化列表
	test_feature_list - 测试集向量化列表
"""
def TextFeatures(train_data_list, test_data_list, feature_words):
	print('文本向量化过程中.....')
	def text_features(text, feature_words):						#出现在特征集中，则置1
		text_words = set(text)
		features = [1 if word in text_words else 0 for word in feature_words]
		return features
	train_feature_list = [text_features(text, feature_words) for text in train_data_list]
	test_feature_list = [text_features(text, feature_words) for text in test_data_list]
	# print('-------------------------')
	# print(train_feature_list)
	return train_feature_list, test_feature_list				#返回结果

"""
函数说明:文本特征选取，得到所有的词去停用词之后的特征集
Parameters:
	all_words_list - 训练集所有文本列表
	stopwords_set - 指定的结束语
Returns:
	feature_words - 特征集
"""
def words_dict(all_words_list, stopwords_set = set()):
	ignored_words_file = open('./process_files/ignored_feature_words','w')
	selected_feature_words_file = open('./process_files/selected_feature_words','w')
	detected_stopwords_file = open('./process_files/detected_stopwords','w')

	print('生成特征集过程中........')
	feature_words = []							#特征列表
	special_feature_words = ['theend','TheEND','THEEND','Theend','TheEnd','完'.decode('utf-8'),'end','End','END']	#特殊特征集
	n = 1
	for t in range(len(all_words_list)):
		if n > 1000:							#feature_words的维度为1000     可改进
			ignored_words_file.write(all_words_list[t].encode('utf-8'))
			ignored_words_file.write('\n')

		#如果这个词不是数字，并且不是指定的结束语，并且单词长度大于1小于5，那么这个词就可以作为特征词
		if not all_words_list[t].isdigit() and all_words_list[t] not in stopwords_set and 1 < len(all_words_list[t]) < 5:   #长度限制可改进
			#将特征集写入
			selected_feature_words_file.write(all_words_list[t].encode('utf-8'))
			selected_feature_words_file.write('\n')

			feature_words.append(all_words_list[t])
			n += 1
		#单独处理特殊情况的单个字符
		elif all_words_list[t] in special_feature_words:
			#将特征集写入
			selected_feature_words_file.write(all_words_list[t].encode('utf-8'))
			selected_feature_words_file.write('\n')

			feature_words.append(all_words_list[t])
			n += 1
		else:
			#保存生成特征集过程中去停用词
			detected_stopwords_file.write(all_words_list[t].encode('utf-8'))
			detected_stopwords_file.write('\n')

	return feature_words

"""
函数说明:新闻分类器
Parameters:
	train_feature_list - 训练集向量化的特征文本
	test_feature_list - 测试集向量化的特征文本
	train_class_list - 训练集分类标签
	test_class_list - 测试集分类标签
Returns:
	test_accuracy - 分类器精度
"""
def TextClassifier(train_feature_list, test_feature_list, train_class_list, test_class_list):
	print('分类器训练过程中......')
	gnb = MultinomialNB()
	classifier = gnb.fit(train_feature_list, train_class_list)
	#存储分类器数据
	with open('ad_classifier.pkl','wb') as fid:
		cPickle.dump(gnb,fid)

	#测试准确度
	test_accuracy = classifier.score(test_feature_list, test_class_list)
	print('test_accuracy')
	print(test_accuracy)

	return classifier
if __name__ == '__main__':
	#文本预处理
	folder_path = './ads/train'
	all_words_list, train_data_list, test_data_list, train_class_list, test_class_list = TextProcessing(folder_path, test_size=0.2)

	# 生成stopwords_set
	stopwords_file = './stopwords_cn.txt'
	stopwords_set = MakeWordsSet(stopwords_file)

	#特征集
	feature_words = words_dict(all_words_list, stopwords_set)

	#将feature_words存储
	feature_words_file = open('./process_files/feature_words.txt','wb')
	feature_words_file.write(str(feature_words))

	#将训练集、测试集向量化
	train_feature_list, test_feature_list = TextFeatures(train_data_list, test_data_list, feature_words)

	#训练并返回分类器
	classifier = TextClassifier(train_feature_list, test_feature_list, train_class_list, test_class_list)
