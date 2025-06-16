import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score

# 題目一
df = pd.read_csv('pokemon.csv')

# 題目二 
df = df.dropna(subset=['Attack', 'Defense'])

# 題目三
scaler = StandardScaler()

# 題目四
df = df[df['Type1'].isin(['Normal', 'Fighting', 'Ghost'])]
X = scaler.fit_transform(df[['Attack', 'Defense']]) 
y = df['Type1']

clf = SVC()
clf.fit(X, y)

y_pred = clf.predict(X)

# 題目五
accuracy = accuracy_score(y, y_pred)
f1 = f1_score(y, y_pred, average='weighted')
# 題目六
new= [[100, 75]]
new_scaled = scaler.transform(new)
prediction = clf.predict(new_scaled)

# 答案一
wrong = (y_pred != y).sum()
print(f"錯誤分類個數：{wrong}")

# 答案二
print(f"準確度：{accuracy:.4f}")

# 答案三
print(f"F1：{f1:.4f}")

# 答案四
print(f"寶可夢預測為：{prediction[0]}")
