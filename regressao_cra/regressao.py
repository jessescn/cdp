import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Carregando o dataset da regressao
df = pd.read_csv('https://canvas.instructure.com/files/79840847/download?download_frd=1')

# Declarando as variáveis
x = df[['Cálculo1', 'LPT', 'P1', 'IC', 'Cálculo2']]
y = df.cra.values

one_predictor = ['IC']
three_predictors = ['IC', 'Cálculo1', 'P1']
five_predictors = ['Cálculo1', 'LPT', 'P1', 'IC', 'Cálculo2']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)

def plot_corr(corr):
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask, 1)] = True

    return sns.heatmap(corr, mask=mask, cmap='RdBu', square=True, linewidths=.5)

# Calculando a correlação
plot_corr(df.corr()).plot(title="Disciplinas")

def calculate_metrics(predict_test, predict_train, display_results):
  
  rmse_test = round(np.sqrt(mean_squared_error(y_test, predict_test)), 5)
  rmse_train = round(np.sqrt(mean_squared_error(y_train, predict_train)), 5)
  r2_train = round(r2_score(y_train, predict_train), 5)
  r2_test = round(r2_score(y_test, predict_test), 5)
  
  if display_results:

    print("\nValor RMSE do teste : {}".format(rmse_test))
    print("Valor RMSE do treino : {}\n".format(rmse_train))
    print("Valor R² do treino: {}".format(r2_train))
    print("Valor R² do teste: {}".format(r2_test))
  
  return [rmse_test, rmse_train]
  

def linear_regression(train_predictors, test_predictors, display_results=True):

  linear_regression = LinearRegression()
  linear_regression.fit(train_predictors, y_train)
  cra_predict_test = linear_regression.predict(test_predictors)
  cra_predict_train = linear_regression.predict(train_predictors)

  return calculate_metrics(cra_predict_test, cra_predict_train, display_results)
  

# Regressao apenas com a variavel IC
  
train = x_train[one_predictor]
test = x_test[one_predictor]

[rmse_test, rmse_train] = linear_regression(train, test)

# Regressao com as variaveis 'IC', 'Cálculo1'e 'P1'

train = x_train[three_predictors]
test = x_test[three_predictors]

[rmse_test, rmse_train] = linear_regression(train, test)

train = x_train[five_predictors]
test = x_test[five_predictors]

[rmse_test, rmse_train] = linear_regression(train, test)

# Criando 10 partições de teste e treino e calculando a média dos MRSE dos três modelos (1, 3 e 5 variáveis)
def calculate_errors(variables, x_train, x_test):
  train = x_train[variables]
  test = x_test[variables]
  return linear_regression(train, test, False)

def calculate_rmse(x_test, x_train, y_test, y_train):
  train_values, test_values = [], []
  predictors = [one_predictor, three_predictors, five_predictors]
  
  for predictors in predictors:
  
    [rmse_test, rmse_train] = calculate_errors(predictors, x_train, x_test)
    train_values.append(rmse_train)
    test_values.append(rmse_test)
    
  return [train_values, test_values]
  

def generate_partitions():
  train_models, test_models = [], []
  trai = []
  random_values = [0,4,15,24,29,35,42,48,56,70]
  
  for random in random_values:
    
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.1, random_state=random)
  
    [train_values, test_values] = calculate_rmse(test_x, train_x, test_y, train_y)
    train_models.append(train_values)
    test_models.append(test_values)
    
  train_means = [0,0,0]
  test_means = [0,0,0]
  
  
  for i in range(3):
    for j in range(10):
      train_means[i] += train_models[j][i]
      test_means[i] += test_models[j][i]
      
  for i in range(3):
    train_means[i] = train_means[i]/10
    test_means[i] = test_means[i]/10
  
  return [train_means, test_means]
  
[train_means, test_means] = generate_partitions()

print('RMSE dos seguintes modelos:')
print('\nModelo com uma variável (M1): \nRMSE de treino = {}\nRMSE de teste = {}'.format(train_means[0], test_means[0]))
print('\nModelo com três variáveis (M1): \nRMSE de treino = {}\nRMSE de teste = {}'.format(train_means[1], test_means[1]))
print('\nModelo com cinco variáveis (M1): \nRMSE de treino = {}\nRMSE de teste = {}'.format(train_means[2], test_means[2]))

x_pos = np.arange(len(train_means))

w = 0.3

plt.bar(x_pos - 0.17, train_means, width=w, color='b', align='center')
plt.bar(x_pos + 0.17, test_means, width=w, color='r', align='center')

plt.xticks(x_pos, ['M1', 'M2', 'M3'])
plt.legend(['treino', 'teste'])

plt.show() 