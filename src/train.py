# %%
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn import tree
# %%
df = pd.read_csv('../data/netflix_customer_churn.csv')
# %%
df.head()
# %%
# Reposicionando Churn para a ultima posição
col = df.pop('churned')
df['churned'] = col
# %%
sns.pairplot(df, hue='churned')
plt.show()

sns.boxplot(df)
plt.show()

print(df.isnull().sum())
df.info()
# %%

# Essas são as variáveis usadas
features = df.columns[1:-1]
# %%
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
# %%

# Isso significa que as variáveis resposta podem estar semelhantes a partir da divisão aleatória
print('Taxa variável resposta geral', y.mean())
print('Taxa variável resposta Treino', y_train.mean())
print('Taxa variável resposta Teste', y_test.mean())

# %%
df_analise = X_train.copy()
df_analise[target] = y_train.copy()

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
# Pre-processing Pipeline
cat_cols = X_train.select_dtypes(include='object').columns
num_cols = X_train.select_dtypes(include='number').columns

# %%

preprocess = ColumnTransformer([
    ('cat', OneHotEncoder(
        handle_unknown='ignore',
        sparse_output=False), cat_cols)
], remainder='passthrough', verbose_feature_names_out=False
)

pipeline_pre = Pipeline(steps=[
    ('preprocess', preprocess)
])
pipeline_pre.set_output(transform="pandas")

X_train_transformed = pipeline_pre.fit_transform(X_train)

# %%
arvore = tree.DecisionTreeClassifier(random_state=42, max_depth=5)
arvore.fit(X_train_transformed, y_train)
# %%
plt.figure(dpi=800)
tree.plot_tree(arvore, feature_names=X_train_transformed.columns,
               filled=True, rounded=True, class_names=['Dont Churned', 'churned'])
plt.show()
# %%
