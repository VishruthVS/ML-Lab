import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
dataset = pd.read_csv('./ToyotaCorolla.csv')

# Extracting variables
x = dataset['KM']
y = dataset['Doors']
z = dataset['Price']

# Scatter plot
plt.scatter(x, z)
plt.xlabel('KM')
plt.ylabel('Price')
plt.title('Scatter Plot of KM vs Price')
plt.show()

# Box plot
plt.boxplot(z)
plt.ylabel('Price')
plt.title('Box Plot of Price')
plt.show()

# Heat map
correlation_matrix = dataset.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# Contour plot
plt.tricontourf(x, y, z, cmap='coolwarm')
plt.xlabel('KM')
plt.ylabel('Doors')
plt.title('Contour Plot of KM vs Doors')
plt.colorbar(label='Price')
plt.show()

# 3D Surface plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_trisurf(x, y, z, cmap='jet')
ax.set_title('3D Surface Plot')
ax.set_xlabel('KM')
ax.set_ylabel('Doors')
ax.set_zlabel('Price')
plt.show()
