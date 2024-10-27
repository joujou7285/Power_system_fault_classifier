import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve, validation_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import pickle
# 載入資料集到 DataFrame 中
file_path = r'C:\Users\jou\OneDrive\桌面\電力系統故障監控\classData.csv'
df = pd.read_csv(file_path)
# 將 G, C, B, A 合併為一個新的列 "CLASS"
df['CLASS'] = df[['G', 'C', 'B', 'A']].values.tolist()
df = df.drop(columns=['G', 'C', 'B', 'A'])
# 定義 CLASS 轉換
mapping = {
    '[0, 0, 0, 0]': 0,  # No Fault
    '[1, 0, 0, 1]': 1,  # LG Fault
    '[0, 1, 1, 1]': 2,  # LL Fault
    '[1, 0, 1, 1]': 3,  # LLG Fault
    '[0, 1, 1, 0]': 4,  # LLL Fault
    '[1, 1, 1, 1]': 5   # LLLG Fault
}

# 將 CLASS 列轉換為字符串格式以進行轉換
df['CLASS'] = df['CLASS'].astype(str)
# # 將 CLASS 列轉換成數值
df['CLASS_NUM'] = df['CLASS'].map(mapping)

# 檢查 CLASS 列中的唯一值和它們的出現次數
print(df['CLASS'].value_counts())

# 使用 pd.crosstab() 來建立 CLASS 和 CLASS_NUM 的對照表
classification_table = pd.crosstab(df['CLASS'], df['CLASS_NUM'], rownames=['CLASS (List Format)'], colnames=['CLASS_NUM (Numeric)'])
# 打印結果
print(classification_table)

df = df.drop(columns=['CLASS'])
print(df)
# 查看更新後的 DataFrame
print(df.describe())
df.info()
# 數據標準化
variables = ['Ia', 'Ib', 'Ic', 'Va', 'Vb', 'Vc']
scaler = StandardScaler()
df[variables] = scaler.fit_transform(df[variables])
print(df)
# 數據分成測試集，訓練集
X = df[variables].values  # 特徵資料（欄位 1 到 6）
y = df["CLASS_NUM"].values  # 目標變數
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
# 隨機森林分類器
params = {
    "n_estimators": 200,  # 未變
    "max_depth": 10,      # 更新為 10
    "random_state": 7,    # 未變
    "class_weight": "balanced",    # 新增
    "max_features": "sqrt",        # 新增
    "min_samples_leaf": 1,         # 新增
    "min_samples_split": 2         # 新增
}


classifier = RandomForestClassifier(**params)

# 訓練模型
classifier.fit(X_train, y_train)
# 預測測試集
y_pred = classifier.predict(X_test)

# 步驟 3: 交叉驗證評估
accuracy = cross_val_score(classifier, X_test, y_test, scoring="accuracy", cv=3)
print(f"交叉驗證後的準確度: {round(100 * accuracy.mean(), 2)} %")

# 步驟 4: 混淆矩陣
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

# 步驟 5: 分類報告
print("\n分類報告：\n")
print(classification_report(y_test, y_pred))
output_file= "saved_random_forest_model.pkl"
with open(output_file,"wb")as f:
    pickle.dump(classifier,f)
# 超參數調整
from sklearn.model_selection import GridSearchCV

# 定義超參數網格
# param_grid = {
#     'n_estimators': [200],
#     'max_depth': [5, 7, 10],
#     'min_samples_split': [2, 5, 10],
#     'min_samples_leaf': [1, 2, 4],
#     'max_features': ['auto', 'sqrt'],
#     'class_weight': [None, 'balanced']
# }
#
# # 使用 GridSearchCV 進行超參數調整
# grid_search = GridSearchCV(estimator=classifier, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
# grid_search.fit(X_train, y_train)
#
# # 打印最佳參數
# print("Best Parameters:", grid_search.best_params_)
#
# # 使用最佳參數進行預測
# best_clf = grid_search.best_estimator_
# y_pred = best_clf.predict(X_test)
#
# # 評估模型性能
# print("Confusion Matrix:")
# print(confusion_matrix(y_test, y_pred))
# print("\nClassification Report:")
# print(classification_report(y_test, y_pred))
