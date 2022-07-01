import pandas as pd

df = pd.read_csv('./ogv_cover_info.tsv', sep='\t')
avid_set = set()
for i in range(len(df)):
    avid = df['avid'][i]
    avid_set.add(avid)
print(len(avid_set))
