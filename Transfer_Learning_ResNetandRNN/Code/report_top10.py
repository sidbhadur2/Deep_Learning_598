import pandas as pd

df = pd.read_csv('performance_difference.csv')
df_improved = df.sort_values('Accuracy', ascending = False)
df_worse = df.sort_values('Accuracy')

df_improved.to_csv('Improved_Accuracy.csv')
df_worse.to_csv('Worse_Accuracy.csv')


