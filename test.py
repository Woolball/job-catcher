import pandas as pd

with open('/home/woolball/Downloads/mycareersfuture.json', encoding='utf-8') as inputfile:
    df = pd.read_json(inputfile)

df.to_csv('/home/woolball/Downloads/csvfile.csv', encoding='utf-8', index=False)
