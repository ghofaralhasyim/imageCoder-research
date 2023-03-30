import pandas as pd
from matplotlib import pyplot as plt

plt.rcParams["figure.figsize"] = [10, 3.50]
plt.rcParams["figure.autolayout"] = True

columns = ['loss', 'val_loss']

df = pd.read_csv("./bin/conv2d-gru/training_val_losses_1.csv", usecols=columns)
df.plot()
plt.xlabel('epochs')

plt.show()