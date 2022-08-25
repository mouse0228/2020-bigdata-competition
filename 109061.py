import pandas as pd
import warnings
import cmath

warnings.filterwarnings("ignore")

df = pd.read_csv('0714train.csv', header = 0)
test = pd.read_csv('0728test.csv' , header = 0)
df = df.drop(['Number'],axis=1)
test = test.drop(['Number'],axis=1)

#單一值去掉
df = df.drop(['Input_A1_010','Input_A2_008','Input_A2_010','Input_A4_010',
              'Input_A5_010','Input_A6_010','Input_C_002' ,'Input_C_003' ,
              'Input_C_006' ,'Input_C_132' ],axis = 1)

test = test.drop(['Input_A1_010','Input_A2_008','Input_A2_010','Input_A4_010',
              'Input_A5_010','Input_A6_010','Input_C_002' ,'Input_C_003' ,
              'Input_C_006' ,'Input_C_132' ],axis = 1)

#差異太小值去掉
df = df.drop(['Input_A1_008','Input_A3_008','Input_A3_010','Input_A4_008',
              'Input_A5_008','Input_A6_008','Input_C_004' ,'Input_C_009' ],axis = 1)

test = test.drop(['Input_A1_008','Input_A3_008','Input_A3_010','Input_A4_008',
              'Input_A5_008','Input_A6_008','Input_C_004' ,'Input_C_009' ],axis = 1)

#將偏移量的文字參數改為極座標
object_cols = list((df.dtypes == 'object')[(df.dtypes == 'object')].index)
df[object_cols] = df[object_cols].fillna('F;-1;F;-1')
test[object_cols] = test[object_cols].fillna('F;-1;F;-1')
def polor(df):
    for i in range(len(object_cols)):
        locals()['C%s_r' % (i)] = []
    for i in range(len(object_cols)):
        locals()['C%s_1' % (i)] = []
    for i in range(len(object_cols)):
        locals()['C%s_2' % (i)] = []
    for i in range(len(object_cols)):
        locals()['C%s_3' % (i)] = []
    for i in range(len(object_cols)):
        locals()['C%s_4' % (i)] = []
    j= 0
    for name in object_cols:
        for i in df.index:
            a = df[name][i].split(";",3)
            if( a[0]=='N' and a[2]=='N') or (a[1]=='0'and a[3]=='0'):
                locals()['C%s_1' % (j)].append(0)
                locals()['C%s_2' % (j)].append(0)
                locals()['C%s_3' % (j)].append(0)
                locals()['C%s_4' % (j)].append(0)
                x = 0
                y = 0
            elif (a[1]=='0' or a[0]=='N') and a[3]!='0':
                if a[2]=='R':
                    locals()['C%s_1' % (j)].append(1)
                    locals()['C%s_2' % (j)].append(0)
                    locals()['C%s_3' % (j)].append(0)
                    locals()['C%s_4' % (j)].append(1) 
                    x = float(a[3])
                    y = 0
                elif a[2]=='L':                    
                    locals()['C%s_1' % (j)].append(0)
                    locals()['C%s_2' % (j)].append(1)
                    locals()['C%s_3' % (j)].append(1)
                    locals()['C%s_4' % (j)].append(0) 
                    x = -float(a[3])
                    y = 0  
                elif a[2]=='N':
                    locals()['C%s_1' % (j)].append(0)
                    locals()['C%s_2' % (j)].append(0)
                    locals()['C%s_3' % (j)].append(0)
                    locals()['C%s_4' % (j)].append(0)  
                    x = 0
                    y = 0
            elif a[1]!='0' and (a[3]=='0'or a[2]=='N'):
                if a[0]=='U':
                    locals()['C%s_1' % (j)].append(1)
                    locals()['C%s_2' % (j)].append(1)
                    locals()['C%s_3' % (j)].append(0)
                    locals()['C%s_4' % (j)].append(0) 
                    x = 0
                    y = float(a[1])                
                elif a[0]=='D':
                    locals()['C%s_1' % (j)].append(0)
                    locals()['C%s_2' % (j)].append(0)
                    locals()['C%s_3' % (j)].append(1)
                    locals()['C%s_4' % (j)].append(1)
                    x = 0
                    y = -float(a[1])  
                elif a[2]=='N':
                    locals()['C%s_1' % (j)].append(0)
                    locals()['C%s_2' % (j)].append(0)
                    locals()['C%s_3' % (j)].append(0)
                    locals()['C%s_4' % (j)].append(0)  
                    x = 0
                    y = 0
            else:
                if a[0]=='U' and a[2]=='R':
                    locals()['C%s_1' % (j)].append(1)
                    locals()['C%s_2' % (j)].append(0)
                    locals()['C%s_3' % (j)].append(0)
                    locals()['C%s_4' % (j)].append(0) 
                    x = float(a[3])
                    y = float(a[1])
                elif a[0]=='U' and a[2]=='L':
                    locals()['C%s_1' % (j)].append(0)
                    locals()['C%s_2' % (j)].append(1)
                    locals()['C%s_3' % (j)].append(0)
                    locals()['C%s_4' % (j)].append(0) 
                    x = -float(a[3])
                    y = float(a[1])
                elif a[0]=='D' and a[2]=='L':
                    locals()['C%s_1' % (j)].append(0)
                    locals()['C%s_2' % (j)].append(0)
                    locals()['C%s_3' % (j)].append(1)
                    locals()['C%s_4' % (j)].append(0)
                    x = -float(a[3])
                    y = -float(a[1])
                elif a[0]=='D' and a[2]=='R':
                    locals()['C%s_1' % (j)].append(0)
                    locals()['C%s_2' % (j)].append(0)
                    locals()['C%s_3' % (j)].append(0)
                    locals()['C%s_4' % (j)].append(1)
                    x = -float(a[3])
                    y = -float(a[1])
                elif a[0]=='F' and a[2]=='F':
                    locals()['C%s_1' % (j)].append(-1)
                    locals()['C%s_2' % (j)].append(-1)
                    locals()['C%s_3' % (j)].append(-1)
                    locals()['C%s_4' % (j)].append(-1)
                    locals()['C%s_r' % (j)].append(-1)
            if a[0]!='F' and a[2]!='F':
                z = cmath.polar(complex(x , y))
                r = z[0]
                locals()['C%s_r' % (j)].append(r)
        j = j+1
    for j in range(len(object_cols)):
        df['C%s_r' % (j)] = locals()['C%s_r' % (j)]  
        df['C%s_1' % (j)] = locals()['C%s_1' % (j)]
        df['C%s_2' % (j)] = locals()['C%s_2' % (j)]
        df['C%s_3' % (j)] = locals()['C%s_3' % (j)]
        df['C%s_4' % (j)] = locals()['C%s_4' % (j)]
    df = df.drop(object_cols,axis=1)
    return df

df = polor(df)
test = polor(test)

#將剩餘少量缺失值補眾數
df = df.fillna(df.mode().iloc[0])
test = test.fillna(test.mode().iloc[0])

#label
y = df[['Input_A1_020','Input_A2_016','Input_A2_017','Input_A2_024','Input_A3_013',
        'Input_A3_015','Input_A3_016','Input_A3_017','Input_A3_018','Input_A6_001',
        'Input_A6_011','Input_A6_019','Input_A6_024','Input_C_013' ,'Input_C_046' ,
        'Input_C_049' ,'Input_C_050' ,'Input_C_057' ,'Input_C_058' ,'Input_C_096'
        ]]

# 去除20項預測目標
df = df.drop(['Input_A1_020','Input_A2_016','Input_A2_017','Input_A2_024','Input_A3_013',
              'Input_A3_015','Input_A3_016','Input_A3_017','Input_A3_018','Input_A6_001',
              'Input_A6_011','Input_A6_019','Input_A6_024','Input_C_013' ,'Input_C_046' ,
              'Input_C_049','Input_C_050' ,'Input_C_057' ,'Input_C_058' ,'Input_C_096'], axis=1)

from sklearn.ensemble import RandomForestRegressor ,ExtraTreesRegressor

final = pd.DataFrame(index=list(range(1,96)))
final.index.name = '預測筆數'

# 調整參數 A6_024  
  
rnd_clf = ExtraTreesRegressor(n_estimators=10,random_state=42)
rnd_clf.fit(df , y[y.columns[12]])
feature_imp = pd.Series(rnd_clf.feature_importances_, index = df.columns).sort_values(ascending = False)
X_train_fe = df[feature_imp.index[:21]]
X_test_fe = test[feature_imp.index[:21]]
rnd_clf = ExtraTreesRegressor(max_features = 21,n_estimators=13,min_samples_leaf=1,max_depth=12)
rnd_clf.fit( X_train_fe,y[y.columns[12]] )
predictions = rnd_clf.predict(X_test_fe)
final["Input_A6_024"] = predictions


# 調整參數 A3_016  

rnd_clf = ExtraTreesRegressor(n_estimators=152,random_state=42)
rnd_clf.fit(df , y[y.columns[6]])
feature_imp = pd.Series(rnd_clf.feature_importances_, index = df.columns).sort_values(ascending = False)
X_train_fe = df[feature_imp.index[:44]]
X_test_fe = test[feature_imp.index[:44]]
rnd_clf = ExtraTreesRegressor(max_features = 44,n_estimators=182,min_samples_leaf=1,max_depth=10)
rnd_clf.fit( X_train_fe,y[y.columns[6]] )
predictions = rnd_clf.predict(X_test_fe)
final["Input_A3_016"] = predictions

# 調整參數 C_013 

y_max = y[y.columns[13]].max()
y_min = y[y.columns[13]].min()
y[y.columns[13]] = (y[y.columns[13]]-y_min)/(y_max-y_min)
rnd_clf = RandomForestRegressor(n_estimators=70,random_state=42)
rnd_clf.fit(df , y[y.columns[13]])
feature_imp = pd.Series(rnd_clf.feature_importances_, index = df.columns).sort_values(ascending = False)
X_train_fe = df[feature_imp.index[:78]]
X_test_fe = test[feature_imp.index[:78]]
rnd_clf = RandomForestRegressor(max_features = 46,n_estimators=51,min_samples_leaf=12,max_depth=12)
rnd_clf.fit( X_train_fe,y[y.columns[13]] )
predictions = rnd_clf.predict(X_test_fe)
predictions = predictions*(y_max-y_min)+y_min
final["Input_C_013"] = predictions

# 調整參數 A2_016  

rnd_clf = ExtraTreesRegressor(n_estimators=82,random_state=42)
rnd_clf.fit(df , y[y.columns[1]])
feature_imp = pd.Series(rnd_clf.feature_importances_, index = df.columns).sort_values(ascending = False)
X_train_fe = df[feature_imp.index[:40]]
X_test_fe = test[feature_imp.index[:40]]
rnd_clf = ExtraTreesRegressor(max_features = 28,n_estimators=122,min_samples_leaf=1,max_depth=14)
rnd_clf.fit( X_train_fe,y[y.columns[1]] )
predictions = rnd_clf.predict(X_test_fe)
final["Input_A2_016"] = predictions

# 調整參數 A3_017 
   
rnd_clf = RandomForestRegressor(n_estimators=82,random_state=42)
rnd_clf.fit(df , y[y.columns[7]])
feature_imp = pd.Series(rnd_clf.feature_importances_, index = df.columns).sort_values(ascending = False)
X_train_fe = df[feature_imp.index[:15]]
X_test_fe = test[feature_imp.index[:15]]
rnd_clf = RandomForestRegressor(max_features = 8,n_estimators=194,min_samples_leaf=3,max_depth=11)
rnd_clf.fit( X_train_fe,y[y.columns[7]] )
predictions = rnd_clf.predict(X_test_fe)
final["Input_A3_017"] = predictions

# 調整參數 C_050 

rnd_clf = ExtraTreesRegressor(n_estimators=120,random_state=42)
rnd_clf.fit(df , y[y.columns[16]])
feature_imp = pd.Series(rnd_clf.feature_importances_, index = df.columns).sort_values(ascending = False)
X_train_fe = df[feature_imp.index[:45]]
X_test_fe = test[feature_imp.index[:45]]
rnd_clf = RandomForestRegressor(max_features = 42,n_estimators=27,min_samples_leaf=4,max_depth=8)
rnd_clf.fit( X_train_fe,y[y.columns[16]] )
predictions = rnd_clf.predict(X_test_fe)
final["Input_C_050"] = predictions

# 調整參數 A6_001 

rnd_clf = ExtraTreesRegressor(n_estimators=2,random_state=42)
rnd_clf.fit(df , y[y.columns[9]])
feature_imp = pd.Series(rnd_clf.feature_importances_, index = df.columns).sort_values(ascending = False)
X_train_fe = df[feature_imp.index[:23]]
X_test_fe = test[feature_imp.index[:23]]
rnd_clf = ExtraTreesRegressor(max_features = 23,n_estimators=21,min_samples_leaf=1,max_depth=10)
rnd_clf.fit( X_train_fe,y[y.columns[9]] )
predictions = rnd_clf.predict(X_test_fe)
final["Input_A6_001"] = predictions

# 調整參數 C_096 

rnd_clf = ExtraTreesRegressor(n_estimators=208,random_state=42)
rnd_clf.fit(df , y[y.columns[19]])
feature_imp = pd.Series(rnd_clf.feature_importances_, index = df.columns).sort_values(ascending = False)
X_train_fe = df[feature_imp.index[:18]]
X_test_fe = test[feature_imp.index[:18]]
rnd_clf = ExtraTreesRegressor(max_features = 11,n_estimators=27,max_depth=12,min_samples_leaf=1)
rnd_clf.fit( X_train_fe,y[y.columns[19]] )
predictions = rnd_clf.predict(X_test_fe)
final["Input_C_096"] = predictions

# 調整參數 A3_018  

rnd_clf = ExtraTreesRegressor(n_estimators=297,random_state=42)
rnd_clf.fit(df , y[y.columns[8]])
feature_imp = pd.Series(rnd_clf.feature_importances_, index = df.columns).sort_values(ascending = False)
X_train_fe = df[feature_imp.index[:35]]
X_test_fe = test[feature_imp.index[:35]]
rnd_clf = RandomForestRegressor(max_features = 10,n_estimators=42,min_samples_leaf=2,max_depth=10)
rnd_clf.fit( X_train_fe,y[y.columns[8]] )
predictions = rnd_clf.predict(X_test_fe)
final["Input_A3_018"] = predictions

# 調整參數 A6_019  

rnd_clf = ExtraTreesRegressor(n_estimators=178,random_state=42)
rnd_clf.fit(df , y[y.columns[11]])
feature_imp = pd.Series(rnd_clf.feature_importances_, index = df.columns).sort_values(ascending = False)
X_train_fe = df[feature_imp.index[:45]]
X_test_fe = test[feature_imp.index[:45]]
rnd_clf = ExtraTreesRegressor(max_features = 45,n_estimators=91,min_samples_leaf=1,max_depth=13)
rnd_clf.fit( X_train_fe,y[y.columns[11]] )
predictions = rnd_clf.predict(X_test_fe)
final["Input_A6_019"] = predictions

# 調整參數 A1_020  

rnd_clf = ExtraTreesRegressor(n_estimators=178,random_state=42)
rnd_clf.fit(df , y[y.columns[0]])
feature_imp = pd.Series(rnd_clf.feature_importances_, index = df.columns).sort_values(ascending = False)
X_train_fe = df[feature_imp.index[:32]]
X_test_fe = test[feature_imp.index[:32]]
rnd_clf = ExtraTreesRegressor(max_features = 32,n_estimators=150,min_samples_leaf=1,max_depth=23)
rnd_clf.fit( X_train_fe,y[y.columns[0]] )
predictions = rnd_clf.predict(X_test_fe)
final["Input_A1_020"] = predictions

# 調整參數 A6_011  

rnd_clf = RandomForestRegressor(n_estimators=53,random_state=42)
rnd_clf.fit(df , y[y.columns[10]])
feature_imp = pd.Series(rnd_clf.feature_importances_, index = df.columns).sort_values(ascending = False)
X_train_fe = df[feature_imp.index[:50]]
X_test_fe = test[feature_imp.index[:50]]
rnd_clf = RandomForestRegressor(max_features = 8,n_estimators=36,max_depth=16,min_samples_leaf=1)
rnd_clf.fit( X_train_fe,y[y.columns[10]] )
predictions = rnd_clf.predict(X_test_fe)
final["Input_A6_011"] = predictions

# 調整參數 A3_015  
 
rnd_clf = ExtraTreesRegressor(n_estimators=172,random_state=42)
rnd_clf.fit(df , y[y.columns[5]])
feature_imp = pd.Series(rnd_clf.feature_importances_, index = df.columns).sort_values(ascending = False)
X_train_fe = df[feature_imp.index[:31]]
X_test_fe = test[feature_imp.index[:31]]
rnd_clf = ExtraTreesRegressor(max_features = 31,n_estimators=60,min_samples_leaf=1,max_depth=20)
rnd_clf.fit( X_train_fe,y[y.columns[5]] )
predictions = rnd_clf.predict(X_test_fe)
final["Input_A3_015"] = predictions

# 調整參數 C_046

rnd_clf = RandomForestRegressor(n_estimators=164,random_state=42)
rnd_clf.fit(df , y[y.columns[14]])
feature_imp = pd.Series(rnd_clf.feature_importances_, index = df.columns).sort_values(ascending = False)
X_train_fe = df[feature_imp.index[:2]]
X_test_fe = test[feature_imp.index[:2]]
rnd_clf = RandomForestRegressor(max_features = 2,n_estimators=58,min_samples_leaf=14,max_depth=4)
rnd_clf.fit( X_train_fe,y[y.columns[14]] )
predictions = rnd_clf.predict(X_test_fe)
final["Input_C_046"] = predictions

# 調整參數 C_049  

y_max = y[y.columns[15]].max()
y_min = y[y.columns[15]].min()
y[y.columns[15]] = (y[y.columns[15]]-y_min)/(y_max-y_min)
rnd_clf = RandomForestRegressor(n_estimators=150,random_state=42)
rnd_clf.fit(df , y[y.columns[15]])
feature_imp = pd.Series(rnd_clf.feature_importances_, index = df.columns).sort_values(ascending = False)
X_train_fe = df[feature_imp.index[:45]]
X_test_fe = test[feature_imp.index[:45]]
rnd_clf = RandomForestRegressor(max_features = 44,n_estimators=149,min_samples_leaf=3,max_depth=8)
rnd_clf.fit( X_train_fe,y[y.columns[15]] )
predictions = rnd_clf.predict(X_test_fe)
predictions = predictions*(y_max-y_min)+y_min
final["Input_C_049"] = predictions

# 調整參數 A2_024  

rnd_clf = ExtraTreesRegressor(n_estimators=18,random_state=42)
rnd_clf.fit(df , y[y.columns[3]])
feature_imp = pd.Series(rnd_clf.feature_importances_, index = df.columns).sort_values(ascending = False)
X_train_fe = df[feature_imp.index[:34]]
X_test_fe = test[feature_imp.index[:34]]
rnd_clf = ExtraTreesRegressor(max_features = 34,n_estimators=27,min_samples_leaf=1,max_depth=10)
rnd_clf.fit( X_train_fe,y[y.columns[3]] )
predictions = rnd_clf.predict(X_test_fe)
final["Input_A2_024"] = predictions

# 調整參數 C_058  

rnd_clf = RandomForestRegressor(n_estimators=99,random_state=42)
rnd_clf.fit(df , y[y.columns[18]])
feature_imp = pd.Series(rnd_clf.feature_importances_, index = df.columns).sort_values(ascending = False)
X_train_fe = df[feature_imp.index[:18]]
X_test_fe = test[feature_imp.index[:18]]
rnd_clf = ExtraTreesRegressor(max_features = 18,n_estimators=67,max_depth=18,min_samples_leaf=1)
rnd_clf.fit( X_train_fe,y[y.columns[18]] )
predictions = rnd_clf.predict(X_test_fe)
final["Input_C_058"] = predictions

# 調整參數 C_057 

rnd_clf = ExtraTreesRegressor(n_estimators=98,random_state=42)
rnd_clf.fit(df , y[y.columns[17]])
feature_imp = pd.Series(rnd_clf.feature_importances_, index = df.columns).sort_values(ascending = False)
X_train_fe = df[feature_imp.index[:30]]
X_test_fe = test[feature_imp.index[:30]]
rnd_clf = ExtraTreesRegressor(max_features = 30,n_estimators=101,min_samples_leaf=1,max_depth=15)
rnd_clf.fit( X_train_fe,y[y.columns[17]] )
predictions = rnd_clf.predict(X_test_fe)
final["Input_C_057"] = predictions

# 調整參數 A3_013  
    
rnd_clf = RandomForestRegressor(n_estimators=70,random_state=42)
rnd_clf.fit(df , y[y.columns[4]])
feature_imp = pd.Series(rnd_clf.feature_importances_, index = df.columns).sort_values(ascending = False)
X_train_fe = df[feature_imp.index[:26]]
X_test_fe = test[feature_imp.index[:26]]
rnd_clf = RandomForestRegressor(max_features = 7,n_estimators=70,min_samples_leaf=1,max_depth=16)
rnd_clf.fit( X_train_fe,y[y.columns[4]] )
predictions = rnd_clf.predict(X_test_fe)
final["Input_A3_013"] = predictions

# 調整參數 A2_017 

rnd_clf = RandomForestRegressor(n_estimators=84,random_state=42)
rnd_clf.fit(df , y[y.columns[2]])
feature_imp = pd.Series(rnd_clf.feature_importances_, index = df.columns).sort_values(ascending = False)
X_train_fe = df[feature_imp.index[:20]]
X_test_fe = test[feature_imp.index[:20]]
rnd_clf = RandomForestRegressor(max_features = 20 ,n_estimators=61,min_samples_leaf=1,max_depth=11)
rnd_clf.fit( X_train_fe,y[y.columns[2]] )
predictions = rnd_clf.predict(X_test_fe)
final["Input_A2_017"] = predictions

final.to_excel("109061_TestResult.xlsx")
print("預測完成")
