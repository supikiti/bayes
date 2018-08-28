# bayes
# Linear_Regrettion
線形回帰のモデルをガウス分布を使って構築し, さらに係数パラメータの学習及び未観測データの予測を行えるやつ. 
今回は係数パラメータの学習， 未観測データに用いるデータは, 正弦関数からの乱数生成により得た. 

# 定義
M => 線形回帰にM-1次の多項式を仮定する. M=4ならば, 入力ベクトルは(1, x, x^2, x^3)となる.
ramda => ノイズ成分の1次元ガウス分布の既知の精度パラメータ.
rand_num => 学習データとなる乱数の個数 

```python
lr = Linear_Regrettion(M, ramda, rand_num)
```

# fit
コンストラクタで与えられたデータから学習し, パラメータの事後分布を生成する. 引数としてdata_x, data_y を渡すことで, モデルを学習させることができる. 

```python
lr.fit()
```

# plot_result
学習データに事後分布により得られたパラメータの平均をプロットする.

```python
lr.plot_result()
```

![学習データに対するプロット](https://user-images.githubusercontent.com/37444351/44082963-c043be52-9fed-11e8-9991-650293235d91.png)

# plot_test_result
テストデータに対しての予測結果をプロットする. 
引数に指定された値だけ, テストデータを生成し, 予測する．
分散値を点線で明記している.

![テストデータに対するプロット](https://user-images.githubusercontent.com/37444351/44082991-cd18b1fa-9fed-11e8-859a-8f6536d36dc6.png)

```python
lr.plot_test_result()
```

# plot_all_result
Mを適宜変えた場合の様子をプロットする.

![M=1,2,3,4,5,10に対する結果](https://user-images.githubusercontent.com/37444351/44629326-68945c80-a988-11e8-8f61-ab5b70aba127.png)
