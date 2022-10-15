from time import process_time
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.impute import KNNImputer
from sklearn import metrics
from sklearn import preprocessing
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from scipy.stats import ttest_ind
from sklearn.metrics import roc_curve

# получаем данные для обучения модели
df = pd.read_csv('./data/credit_train.csv')

print('\nКоличество строк и количество столбцов:')
print(df.shape[0], df.shape[1])

print('\nОсновная статистика по датасету:')
print(df.describe())

print('\nПросматриваем признаки:')
print(df.dtypes)

print('\nПроверяем количество пропусков:')
print(df.isnull().sum())

# целевым полем выбираем Loan Status
# меняем значения на 0 - не погашен и 1 - погашен
df.loc[(df['Loan Status'] == 'Charged Off', 'Loan Status')] = 0
df.loc[(df['Loan Status'] == 'Fully Paid', 'Loan Status')] = 1

# удаляем столбцы, в которых пропусков больше 20% от общего количества строк
for i in df.iloc[:, 2:].columns:
    if df[i].notna().sum() < 0.9 * len(df):
        del df[i]

print('\nОсталось столбцов:')
print(df.shape[1])

# раскладываем текстовые значения Home Ownership по признакам
df_g = pd.get_dummies(df, columns=['Home Ownership', 'Term', 'Years in current job', 'Purpose'], drop_first=True)

print('\nСтало столбцов:')
print(df_g.shape[1])

print('\nПросматриваем признаки:')
print(df_g.dtypes)

# убираем пропуски
knn = KNNImputer(n_neighbors=8)
knn.fit(df_g.iloc[:,2:].values)
df_g_knn = knn.transform(df_g.iloc[:,2:].values)

df2 = df_g_knn

df3 = pd.DataFrame(df2, columns=df_g.iloc[:,2:].columns)

# поиск выбросов (комп падает)
# df_for_clust = preprocessing.normalize(df3.iloc[:,:-2].values)

# outlier_detection = DBSCAN(eps=7)
# clusters = outlier_detection.fit_predict(df_for_clust)

# print('\nКоличество выбросов:')
# rint(list(clusters).count((-1)))

# корреляция
cor = df3.corr()
plt.figure(figsize=(15, 12))
sns.heatmap(cor, xticklabels=False, yticklabels=False, cmap='coolwarm')
# plt.show()

# удаляем столбцы с высокой корреляцией
all_features = list(cor.columns)
f_to_del = []

for x in cor.columns:
    if x == 'Loan Status':
        continue
    for i in cor.index:
        if all_features.index(i) >= all_features.index(x) or i == 'Loan Status':
            continue
        else:
            if abs(cor.loc[x, i]) > 0.8:
                if abs(cor.loc['Loan Status', x]) > abs(cor.loc['Loan Status', i]):
                    f_to_del.append(i)
                else:
                    f_to_del.append(x)

f_to_del = set(f_to_del)

for x in f_to_del:
    del df3[x]

print('\nОсталось столбцов:')
print(df3.shape[1])

# обучение модели
not_nan = (df['Loan Status'] == 0) | (df['Loan Status'] == 1)

scaler = MinMaxScaler()
scaler.fit(df3)
df3_norm = pd.DataFrame(data=scaler.transform(df3), columns=df3.columns)

x_all = df3_norm.iloc[:,:-2].values
x = df3_norm[not_nan].iloc[:,:-2].values
y = df3_norm[not_nan].iloc[:,-1].values

log_r = LogisticRegression(penalty='l2').fit(x, y)
model = SelectFromModel(log_r, prefit=True, max_features=100)
x_new = model.transform(x_all)

print(x_new)

# разбиение на обучающую и тестовую выборки
x = x_new
y = df[not_nan]['Loan Status'].values
x_to_pred = x_new[-10000:]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=22)

print('Размер обучающей выборки {},\nРазмер тестовой выборки {} \n'.format(len(x_train), len(x_test)))
print('Количество объектов 1 класса в обучающей выборке {},\nВ тестовой выборке {}'.format(y_train.sum(), y_test.sum()))

# балансировка классов
train = pd.DataFrame(data=x_train)
train['Loan Status'] = y_train

train_1 = train[train['Loan Status'] == 1]
train_0 = train[train['Loan Status'] == 0]
train_2 = df[~not_nan].values

train_0 = train_0.sample(train_1.shape[0] * 2, random_state=0, replace=True)

train_1 = pd.concat([train_1, train_1.copy()])
train_bal = pd.concat([train_1, train_0])

x_train = train_bal.iloc[:,:-1].values
y_train = train_bal.loc[:,['Loan Status']].values

print('Балансировка обучающей выборки:')
print(np.unique(y_train, return_counts=True)[1])

print('')

print('Балансировка тестовой выборки:')
print(np.unique(y_test, return_counts=True)[1])

# сравнение выборок
test_mean = np.mean(x_test, axis=0)
pred_mean = np.mean(x_new[-10000:], axis=0)

print(len(test_mean), len(pred_mean))

print(ttest_ind(test_mean, pred_mean))

def results(y_test, y_pred, sec=0):
    report = metrics.classification_report(y_test, y_pred, target_names=['recovered', 'uncovered'])
    print(report)

    print('\nПлощадь под ROC-кривой - ' + str(round(metrics.roc_auc_score(y_test, y_pred), 4)))
    if sec != 0:
        print('\nВремя работы кода: ' + str(round(sec, 4)) + ' сек.')

def ROC_curve(y_test, model):
    sns.set(font_scale=1.5)
    sns.set_color_codes('muted')

    plt.figure(figsize=(10, 8))
    fpr, tpr, treshholds = metrics.roc_curve(y_test, model.predict_proba(x_test)[:,1], pos_label=1)
    lw = 2
    plt.plot(fpr, tpr, lw=lw, label='ROC curve')
    plt.plot([0, 1], [0, 1])
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve')
    plt.show()


# логистическая регрессия
start_time = process_time()

clf = LogisticRegression(random_state=0, C=1, penalty='l2', solver='liblinear')
y = y_train.ravel()
y_train = np.array(y).astype(int)
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)

finish_time = process_time()

sec = finish_time - start_time

y_test = y_test.astype(int)

results(y_test, y_pred, sec)

# по результатам:
# precision - точность (сколько объектов выбранного класса действительно относятся к данному классву)
# recall - скольким людям, которые не вернут кредит, мы дали кредит
# f1-score - среднегармонический по двум предыдущим параметрам

print('\nПлощадь под ROC-кривой по вероятностям - ' + str(round(metrics.roc_auc_score(y_test, clf.predict_proba(x_test)[:,1]), 4)))

ROC_curve(y_test, clf)