import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df1 = pd.read_csv('./report/conv2d-gru-2/training_val_losses.csv')
df2 = pd.read_csv('./report/conv2d-lstm/conv2d-lstm-training_val_losses.csv')
df3 = pd.read_csv('./report/efficientNetV2B0-gru/training_val_losses.csv')

plt.figure(figsize=(10, 5))

plt.plot(df1['loss'], label='conv2d-gru train loss')
plt.plot(df1['val_loss'], label='conv2d-gru val loss')

# plt.plot(df2['loss'], label='conv2d-lstm train loss')
# plt.plot(df2['val_loss'], label='conv2d-lstm val loss')

plt.plot(df3['loss'], label='effnet-gru train loss')
plt.plot(df3['val_loss'], label='effnet-gru val loss')

plt.title('Training loss & Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')

plt.legend()
plt.show()
