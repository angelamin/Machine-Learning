## 权限
为了是脚本可执行，增加mapper.py的可执行权限


> chmod +x mapper.py

> chmod +x reducer.py

## 测试
先在本地测试map reduce代码

> echo "foo foo quux labs foo bar quux" | ./mapper.py

> echo "foo foo quux labs foo bar quux" | ./mapper.py | sort -k1,1 | ./reducer.py

sort -k1,1  参数何意？

-k, -key=POS1[,POS2]     键以pos1开始，以pos2结束

有时候经常使用sort来排序，需要预处理把需要排序的field语言在最前面。实际上这是

完全没有必要的，利用-k参数就足够了。

比如sort all

1 4
2 3
3 2
4 1
5 0
如果sort -k 2的话，那么执行结果就是

5 0
4 1
3 2
2 3
1 4

## 运行
hadoop所在路经
/usr/local/Cellar/hadoop/2.8.0/bin/hadoop


先创建hdfs_in
> hadoop fs -mkdir hdfs://localhost:9000/hdfs_in


把本地的数据文件拷贝到分布式文件系统HDFS中。（放到本地服务器上）


> hadoop dfs -copyFromLocal /Users/xiamin/Downloads/Machine-Learning/hadoop/统计词频/datas hdfs://localhost:9000/hdfs_in

查看

> hadoop fs -ls hdfs://localhost:9000/hdfs_in

执行MapReduce job

> xiamindeMacBook-Air:统计词频 xiamin$ hadoop jar /usr/local/Cellar/hadoop/2.8.0/libexec/share/hadoop/tools/lib/hadoop-\*streaming*.jar -file /Users/xiamin/Downloads/Machine-Learning/hadoop/统计词频/mapper.py -mapper /Users/xiamin/Downloads/Machine-Learning/hadoop/统计词频/mapper.py -file /Users/xiamin/Downloads/Machine-Learning/hadoop/统计词频/reducer.py -reducer /Users/xiamin/Downloads/Machine-Learning/hadoop/统计词频/reducer.py -input hdfs://localhost:9000/hdfs_in/* -output hdfs://localhost:9000/hdfs_out

查看输出结果是否在目标目录/user/rte/hdfs_out

> hadoop dfs -ls hdfs://localhost:9000/hdfs_out

查看输出内容

>  hadoop fs -cat hdfs://localhost:9000/hdfs_out/part-00000
