```
    训练尽量使用多点数据，训练4w+, validation_data：1.7w，数据过少导致一个问题：所有预测结果都是同一个值
    训练时间：机器是E5-2630 v2，24线程，跑了9h，一次epoch需要1080s（谁有GTX1080T可以试试效率提升多少）；
    测试：1416，正确：1357，准确率：88.78%，错误集中在0和O；(1 - 1.0 * 2 / 36) ** 4～～暂时先不管了
    之前有拿过一个知乎友人给的caffe训练模型，据说有98%（单个字符）
```

```
    图像文件夹定义
    train:训练
    test:验证
    check:测试
```

```
    show.py:本来只想用来看图的，写着写着写成了生成图了；所以后续产生数据直接用run.sh
    model.py:vgg
    pred.py:测试，使用了model.py的两个函数
```
