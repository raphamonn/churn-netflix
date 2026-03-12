# %%
import pandas as pd
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import seaborn as sns
# %%
df = pd.read_csv('./data/netflix_customer_churn.csv')

# %%
sns.pairplot(df, hue='churned')
# %%
df.info()
# %%
# %%
