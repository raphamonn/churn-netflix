# %%
import pandas as pd
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
# %%
df = pd.read_csv('./data/netflix_customer_churn.csv')
# %%
df.head()
# %%
# Reposicionando Churn para a ultima posição
col = df.pop('churned')
df['churned'] = col

# %%
# Essas são as variáveis usadas
features = df.columns[1:-1]
# Essa é a target
target = 'churned'
X, y = df[features], df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    random_state=42,
                                                    test_size=0.2,
                                                    stratify=y
                                                    )
# %%
print(X_train.shape)
print(X_test.shape)
# %%

# Isso significa que as variáveis resposta podem estar semelhantes a partir da divisão aleatória
print('Taxa variável resposta geral', y.mean())
print('Taxa variável resposta Treino', y_train.mean())
print('Taxa variável resposta Teste', y_test.mean())
# %%
sns.pairplot(df, hue='churned')
plt.show()
# %%
df.info()

# %%
df.isnull().sum()
# %%
sns.boxplot(df)
# %%
df_analise = X_train
df_analise[target] = y_train
# %%
sumario = df_analise.select_dtypes(exclude='object').groupby(
    'churned').agg(['mean', 'median']).T
sumario.rename({0: 'não deu churn', 1: 'deu churn'}, axis=1)
sumario['diff_rel'] = sumario[0]/sumario[1]
sumario['diff_abs'] = sumario[0] - sumario[1]
sumario.sort_values(by='diff_rel', ascending=False)
# %%
df_analise.corr(numeric_only=True).sort_values(by='churned')
# %%

# %%
