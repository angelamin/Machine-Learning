#coding:UTF-8
import cPickle
import jieba
import ast


'''
函数功能：将需要分类的数据根据特征集进行向量化
Returns: 向量化的结果
'''
def TextFeatures(data,feature_words):
	data = data.strip()
	data = data.replace(' ','') #去除空格
	data_list = []
	word_cut = jieba.cut(data)
	data_list = list(word_cut)
	def text_features(text, feature_words):						#出现在特征集中，则置1
		text_words = set(text)
		features = [1 if word in text_words else 0 for word in feature_words]
		return features
	data_feature_list = [text_features(data_list, feature_words)]
	return data_feature_list				#返回结果

def TextClassifing(classifier, data_feature_list):
	result = classifier.predict(data_feature_list)
	result1 = classifier.predict_log_proba(data_feature_list)
	result2 = classifier.predict_proba(data_feature_list)

	return result,result1,result2

with open('ad_classifier-11-12.pkl','rb') as fid:
    classifier = cPickle.load(fid)


#读取feature_words
feature_words_file = open('./process_files-11-12/feature_words.txt','r')
print('feature_words_file')
print(feature_words_file)
for line in feature_words_file:
    feature_words = ast.literal_eval(line)


#测试广告文件
# ads = open('./ads/Sample/ad','r')
ads = open('./ads/train/ad','r')
wrong_Ad_data = open('./results/wrong_Ad.txt','w')
right_Ad_data = open('./results/right_Ad.txt','w')
num_Ad = 0
for line in ads:
	data_feature_list = TextFeatures(line,feature_words)
	result,result1,result2 = TextClassifing(classifier,data_feature_list)
	if(result[0] == 'not_ad'):
		print('------错误检测广告')
		wrong_Ad_data.write(line)
		wrong_Ad_data.write('\n')
		wrong_Ad_data.write(result)
		wrong_Ad_data.write(str(result2))
	else:
		num_Ad += 1
		print('正确检测广告：：：：')
		right_Ad_data.write(line)
		right_Ad_data.write('\n')
		right_Ad_data.write(result)
		right_Ad_data.write(str(result2))
print('num_Ad')
print(num_Ad)

#测试非广告文件
# not_ads = open('./ads/Sample/not_ad','r')
not_ads = open('./ads/train/not_ad','r')
wrong_notAd_data = open('./results/wrong_notAd.txt','w')
right_notAd_data = open('./results/right_notAd.txt','w')
num_notAd = 0
for line in not_ads:
	data_feature_list = TextFeatures(line,feature_words)
	result,result1,result2 = TextClassifing(classifier,data_feature_list)
	if(result[0] == 'ad'):
		print('-----错误检测非广告')
		wrong_notAd_data.write(line)
		wrong_notAd_data.write('\n')
		wrong_notAd_data.write(result)
		wrong_notAd_data.write(str(result2))
	else:
		num_notAd += 1
		print('正确检测非广告：：：：')
		right_notAd_data.write(line)
		right_notAd_data.write('\n')
		right_notAd_data.write(result)
		right_notAd_data.write(str(result2))

print('num_notAd')
print(num_notAd)

# #测试数组
# test_array = ['中泰证券研究所机构销售通讯录','喜欢的朋友可以扫扫微信','每天看到我们的公众号内容，请关注“走着”','想更进一步了解文博会精彩内容？','只需扫扫二维码就可以把共享单车骑走']
# for test_data in test_array:
# 	print(test_data)
# 	data_feature_list = TextFeatures(test_data,feature_words)
# 	result,result1,result2 = TextClassifier1(train_feature_list, train_class_list,data_feature_list)
# 	print(result)
# 	print(result2)
