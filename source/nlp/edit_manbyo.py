import pandas as pd

df = pd.read_excel('../MANBYO_201907/MANBYO_201907/MANBYO_20190704.xlsx')

# df = df[df['信頼度LEVEL'] != 'D']
# df = df[df['信頼度LEVEL'] != 'E']
# df = df[df['信頼度LEVEL'] != 'F']

df[['出現形', 'しゅつげんけい;icd=ICDコード;lv=信頼度LEVEL/freq=頻度LEVEL;標準病名']].to_csv('../dic/manbyo_seed.csv', index=False)
