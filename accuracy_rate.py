import sklearn
from sklearn.metrics import accuracy_score
import pandas 

x_path = 'udt211_test_full.txt'
y_path = ''

columns = ['token','lemma','pos','morph','morph_full']
x_df = pandas.read_csv(x_path, encoding='utf-8',sep='\t', header=None,names=columns)
y_df = pandas.read_csv(y_path, encoding='utf-8', sep='\t', header=None)
print(x_df['pos'])
print(y_df[2])
x_labels = x_df['pos']
y_labels = y_df[5]

accuracy = accuracy_score(x_labels,y_labels)
print(f"{y_path} trenētā modeļa morfoloģiskā taga precizitāte: {accuracy}")
