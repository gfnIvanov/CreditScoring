import pandas as pd
import io

df_test = pd.read_csv('./data/credit_test.csv')

df_results = pd.read_csv('./data/results.csv')

results = []

i = 0

for x in df_test.values:
    if df_results['Scored Probabilities'][i] > 0.5:
        status = 'Fully Paid'
    else:
        status = 'Charged Off'
    results.append(str(x[0]) + ',' + status)
    i += 1

df = pd.DataFrame(results, columns=["colummn"])

df.to_csv('./data/resultData.csv', encoding='utf8', index=False)