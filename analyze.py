import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Import run log from CSV file
df = pd.DataFrame.from_csv('results/run.py.csv')

# Remove two first rows to remove initialization effects
df = df.loc[df.index > 0]

# Select time difference column
delta_t = df['Time_Diff']

# Plot time difference as a histogram
delta_t.hist(bins=100)
plt.show()
