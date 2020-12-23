
#svm computing recall accuraccy etc
#print("accuracy on test dataset: {}".format(accuracy_score(y_test.to_numpy(), y_test_predicted)))
#print("recall on test dataset: {}".format(recall_score(y_test.to_numpy(), y_test_predicted)))
#print("precision on test dataset: {}".format(precision_score(y_test.to_numpy(), y_test_predicted)))
#print("cost reduction of node %s: " %j, "{}" .format(node_cost_reduction), " euros per QALY\n")
import pandas as pd

# Define a dictionary containing Students data
data = {'Name': ['Jai', 'Princi', 'Gaurav', 'Anuj'],
        'Height': [5.1, 6.2, 5.1, 5.2],
        'Qualification': ['Msc', 'MA', 'Msc', 'Msc']}

# Convert the dictionary into DataFrame
df = pd.DataFrame(data)

address = ['Delhi', 'Bangalore', 'Chennai', 'Patna']

plt.style.use('seaborn-darkgrid')

# create a color palette
palette = plt.get_cmap('Set1')

# multiple line plot
num = 0
for column in df.drop('x', axis=1):
        num += 1
plt.plot(df['x'], df[column], marker='', color=palette(num), linewidth=1, alpha=0.9, label=column)

# Add legend
plt.legend(loc=2, ncol=2)

# Add titles
plt.title("A (bad) Spaghetti plot", loc='left', fontsize=12, fontweight=0, color='orange')
plt.xlabel("Time")
plt.ylabel("Score")

# Using 'Address' as the column name
# and equating it to the list
df['Address'] = address