1. put the probabilities and ground truth in csv and  put them in './data/'
format like(參照隨便一份data裡面格式):
```
,name,prob,label
0,1216912_1.png,0.9868166042160015,1
1,1534242_1.png,0.8885622547129731,0
2,1638885_1.png,0.9214089110781455,1
3,1823594_1.png,0.9655023333513144,1
4,1960573_1.png,0.1449221957920446,1
5,1994086_1.png,0.3464823420512649,1
```

2. run the python to build  txt file, the result will be stored in './txt/'
``` python rebuild_all_into_txt.py```


3. execute the ROCKIT.exe and calculate the 'a' value and 'b' value, record them(參閱readme_單一.doc)

4. use PlotROC.xls to draw these curves.

注意事項：
1. 記得把生成的圖片座標值改為0~1，間距0.1
2. 找出一台window電腦
