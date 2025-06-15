import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read the CSV file into a DataFrame
df = pd.read_csv('log/regnet_loss.csv')

# Convert the `loss` column to numeric, coercing errors to NaN
df['loss'] = pd.to_numeric(df['loss'], errors='coerce')

# Drop rows where `loss` is NaN
df.dropna(subset=['loss'], inplace=True)

# Calculate Exponential Moving Average (EMA)
# Using span for EWM, a common way to specify the smoothing.
# span = 1 / alpha where alpha is the smoothing factor.
# A smaller span (larger alpha) means more weight on recent observations, less smoothing.
# A larger span (smaller alpha) means less weight on recent observations, more smoothing.
# Let's choose a span that gives a reasonable smoothing, e.g., span=50 for similar effect as rolling mean window of 50
span = 50
df['ema_loss'] = df['loss'].ewm(span=span, adjust=False).mean()

# Display the first 5 rows of the DataFrame with the new column
print(df.head().to_markdown(index=False, numalign="left", stralign="left"))

# Print the column names and their data types
print(df.info())

# Create the line plot for the Exponential Moving Average
plt.figure(figsize=(12, 7))
sns.lineplot(
    data=df,
    x='iteration',
    y='ema_loss',
    color='purple',
    linewidth=2
)

# Add title and labels
plt.title(f'Wykres wykładniczej średniej kroczącej straty (span = {span})')
plt.xlabel('Iteracja')
plt.ylabel('Wykładnicza średnia krocząca straty')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()

# Save the plot
plt.savefig('exponential_smoothing_plot.png')