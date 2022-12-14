# 2020-bigdata-competition
>IMBD2020 全國智慧製造大數據分析競賽初賽<br>
>以機台加工參數和加工品質預測加工的20項重要參數<br>
>共281項加工參數(137項鑽孔機共同參數及144項單一鑽孔機各別參數)，6項加工品質<br>
>測試95筆數據<br>

## 資料前處理
使用常用之數據處理方法，針對座標參數依照象限位置數據化，使數據符合模型訓練之需

- 去除單一數據
- 座標參數數據化
- 缺失值補眾數
- 資料標準化

## 演算法
透過隨機森林重要度分析找出重要參數，以交叉驗證方式訓練測試模型，模型使用Random Forest 和 Extra Tree演算法找出最佳參數，分別建立模型回歸結果

![image](https://user-images.githubusercontent.com/67943586/186482224-70d0bbef-72e3-4676-ab94-5030654bbd60.png)

## 預測結果
將數據分成90%訓練、10%test預測20項重要參數之方均根誤差

![image](https://user-images.githubusercontent.com/67943586/186482298-90fb5956-aca1-444c-9d2a-b55df26568f5.png)

## 最終結果
成功進入決賽

