import pandas as pd  # 資料處理
from sklearn.preprocessing import StandardScaler  # 標準化
from sklearn.svm import LinearSVC  # 線性支援向量機
from sklearn.metrics import accuracy_score, f1_score  # 計算準確率, F1分數
from sklearn.preprocessing import LabelEncoder  # 類別編碼器

# 題目一
df = pd.read_csv('pokemon.csv')

# 題目二 
df = df.dropna(subset=['Attack', 'Defense'])

# 題目三
scaler = StandardScaler()
df = df[df['Type1'].isin(['Normal', 'Fighting', 'Ghost'])]
X = df[['Attack', 'Defense']]
y = df['Type1']
le = LabelEncoder()
y = le.fit_transform(y)
X = scaler.fit_transform(X)

# 題目四
clf = LinearSVC(C=0.1, dual=False, class_weight='balanced')
clf.fit(X, y)
y_pred = clf.predict(X)

# 題目五
wrong = (y_pred != y).sum()
accuracy = accuracy_score(y, y_pred)
f1 = f1_score(y, y_pred, average='weighted')

# 題目六
new = [[100, 75]]
new_scaled = scaler.transform(new)
prediction = clf.predict(new_scaled)
label = le.inverse_transform(prediction)

# 答案一
print(f"錯誤分類個數：{wrong}")
# 答案二
print(f"準確度：{accuracy:.4f}")
# 答案三
print(f"F1：{f1:.4f}")
# 答案四
print(f"寶可夢預測為：{label[0]}")
