import pandas as pd
import os
from glob import glob


path = './ratingData/output/'
files = glob(os.path.join(path,"*.csv"),recursive=True)
print(files)

train_output = pd.DataFrame(columns = ['text', 'score'])
test_output = pd.DataFrame(columns = ['text', 'score'])

count = 0
for filename in files:
    df_output = pd.read_csv(f"{filename}",names=['text', 'score'],skiprows=[0])
    if count % 10 < 8:
        train_output = pd.concat([train_output, df_output], ignore_index=True)
    else:
        test_output = pd.concat([test_output, df_output], ignore_index=True)
    count += 1

train_output.to_csv("train_data.csv")
test_output.to_csv("test_data.csv")
