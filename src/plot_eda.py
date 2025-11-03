import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Set Seaborn style
sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)

# load the Borg traces dataset
df_borg = pd.read_csv("../data/raw/borg_traces.zip")
print(df_borg.shape)
# df_borg.head().T


# plot by average_cpu_usage over time
plt.figure(figsize=(14, 7))
sns.lineplot(data=df_borg, x="time_sec", y="average_cpu_usage", ci=None)
plt.title("Average CPU Usage Over Time")
plt.xlabel("Time (seconds)")
plt.ylabel("Average CPU Usage")
plt.show()
