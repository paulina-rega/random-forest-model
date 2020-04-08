import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.feature_selection import SelectFromModel


df = pd.read_csv('IBM-HR-Employee-Attrition.csv')

X = df.drop(['Attrition', 'EmployeeNumber'], axis = 1)
Y = df['Attrition']


# OneHotEncoding for non-numeric values
columns_to_encode = list(X.select_dtypes(include='object').columns)

for col in columns_to_encode:
    one_hot = pd.get_dummies(X[col], prefix=col)
    X = X.drop(col, axis=1)
    X = X.join(one_hot)


y_encoding = {'Yes': 1, 'No': 0}
Y = Y.map(y_encoding)

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3)
clf = RandomForestClassifier(n_estimators=100, random_state = 1, n_jobs =-1)
clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)

print('Accuracy (without choosing parameters): {}'.format(
    metrics.accuracy_score(y_test, y_pred)))


# choosing parameters:
    
labels = list(x_train.columns)
sfm = SelectFromModel(clf, threshold = 0.02)
sfm.fit(x_train, y_train)


x_important_train = sfm.transform(x_train)
x_important_test = sfm.transform(x_test)

clf_important = RandomForestClassifier(n_estimators=100, random_state = 1, 
                                       n_jobs =-1)
clf_important.fit(x_important_train, y_train)

y_important_pred = clf_important.predict(x_important_test)

print('Accuracy (chosen parameters with highest importance): {}'.format(
    metrics.accuracy_score(y_important_pred, y_pred)))
