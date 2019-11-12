import pandas as pd
import matplotlib.pyplot as plt

input_df = pd.read_csv('../output/20newsGroup18828.csv')

print(input_df.shape)

fig, ax = plt.subplots(figsize=(17, 7))
agg_df = input_df.groupby('category').count().sort_values('category')
fig_plot = agg_df.plot(kind='bar', ax=ax, title='The count of each category of the dataset',
            legend=False)
fig_plot.set_xlabel('Categories')
fig_plot.set_ylabel('Count')

plt.tight_layout()
# plt.autoscale()
# plt.show()
plt.savefig('../output/count.png', format='png')