##å¹¿å‘Šè¿‡æ»¤
virtualenv -p python /Users/xiamin/virtualenv/tensorflow-py2                            

source /Users/xiamin/virtualenv/tensorflow-py2/bin/activate

./train.py

tensorboard --logdir /Users/xiamin/Desktop/TextCNN/runs/1510122993/summaries/

pip install --upgrade pip

./eval.py --eval_train --checkpoint_dir="./runs/1510220628/checkpoints/"

pip install jieba
sudo pip install gensim   

 pip install pandas

完整的g广告训练结果：1510122993
h少量数据训练结果：1510219633 1510220628
