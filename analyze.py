import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Import run log from CSV file
df_py = pd.DataFrame.from_csv('results/run.py.csv')
df_cc = pd.DataFrame.from_csv('results/run.cc.csv')

# Remove two first rows to remove initialization effects
df_py = df_py.loc[df_py.index > 0]
df_cc = df_cc.loc[df_cc.index > 0]


# Select time difference column
dt_py = df_py['Time_Diff']
dt_cc = df_cc['Time_Diff']

deg_py = df_py['Output']
deg_cc = df_cc['Output']

# Figure 1 - Time histograms
bins = np.linspace(0.0,0.008,100)
plt.figure(1)
plt.title('Inference timing distribution')
plt.subplot(211)
dt_py.hist(bins=bins)
plt.title('Python')
plt.subplot(212)
dt_cc.hist(bins=bins)
plt.title('C++')
plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.35,
                    wspace=0.35)


plt.figure(2)
plt.title('Inference output')
plt.subplot(211)
dt_py.plot()
plt.subplot(212)
dt_cc.plot()

# Plot time difference as a histogram
plt.show()
