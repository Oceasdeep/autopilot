"""Analyze the execution timing statistics of Python and C++ inferences

This script loads the run log files created by the Python and C++ inference
implementations and displays statistical plots of the data.
"""
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

# Plot histograms of inference execution durations
bins = np.linspace(0.0,0.008,100)
plt.figure(1)
plt.title('Inference duration distribution')
plt.subplot(211)
dt_py.hist(bins=bins)
plt.title('Python')
plt.subplot(212)
dt_cc.hist(bins=bins)
plt.title('C++')
plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.55,
                    wspace=0.35)


# Plot inference execution durations over time
plt.figure(2)
plt.subplot(211)
dt_py.plot()
plt.title('Python')
plt.subplot(212)
dt_cc.plot()
plt.title('C++')
plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.55,
                    wspace=0.35)


# Plot steering angle over time
plt.figure(3)
plt.subplot(211)
deg_py.plot()
plt.title('Python')
plt.subplot(212)
deg_cc.plot()
plt.title('C++')
plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.55,
                    wspace=0.35)

# Plot time difference as a histogram
plt.show()
