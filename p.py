import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 
   
def choose_test(df, sample):
   diabetes = df[df['Outcome'] == 1]
   no_diabetes = df[df['Outcome'] == 0]
   train_diabetes = diabetes.sample(n=sample, replace=False)
   train_no_diabetes = no_diabetes.sample(n=sample, replace=False)
   test_diabetes = diabetes.drop(train_diabetes.index)
   test_no_diabetes = no_diabetes.drop(train_no_diabetes.index)
   training = pd.concat([train_diabetes, train_no_diabetes])
   test = pd.concat([test_diabetes, test_no_diabetes])
   training = training.values
   test = test.values
   np.random.shuffle(training)
   np.random.shuffle(test)
   return training, test

def lin_reg(training_set):
   X = [[1] + list(row[:-1]) for row in training_set]
   y = [[row[-1]] for row in training_set]

   X_T = np.transpose(X)
   X_T_X = np.matmul(X_T, X)
   i_X_T_X = np.linalg.inv(X_T_X)
   X_T_y = np.matmul(X_T, y)
   res = np.matmul(i_X_T_X, X_T_y)
   return res

def predict(test, weight):
   res = weight[0][0]
   for i in range(len(test)):
      res += weight[i+1][0] * test[i]
   return res
  
def accuracy(test_set, weights):
   acc = 0
   for sample in test_set:
    if predict(sample[:-1], weights) >= .5 and sample[-1] == 1:
       acc += 1
    elif predict(sample[:-1], weights) < .5 and sample[-1] == 0:
       acc += 1
   return acc/len(test_set)
       
def run_trials(n_vals, num_trial, data):
   res = {}
   for n in n_vals:
      total_acc = 0
      for _ in range(num_trial):
         training_set, test_set = choose_test(data, n)
         
         
         weights = lin_reg(training_set)
        
         acc = accuracy(test_set, weights)
         total_acc += acc
         
      res[n] = total_acc/num_trial * 100
  
   return res
      
def plot_acc(res):
   x_axis = list(res.keys())
   y_axis = list(res.values())
   plt.plot(x_axis, y_axis, marker='o')
   plt.title('Accuracy rate vs. Samples')
   plt.xlabel('Number of Samples')
   plt.ylabel('Accuracy')
   plt.show()
   
  
def main():
  file = 'pima-indians-diabetes.xlsx'
  df = pd.read_excel(file)
  n_vals = [40, 80, 120, 160, 200]
  num_trial = 1000
  res = run_trials(n_vals, num_trial, df)
  plot_acc(res)
  print(res)

if __name__ == "__main__":
    main()
