import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

path = '.....'
df = pd.read_csv(path, index_col=0)

main_variables = df.groupby('player_name', as_index=False).agg({'player_height': 'mean', 'player_weight':'mean'})

# Visualise distribution of height and weight data

fig, ax = plt.subplots(1,2,figsize=(16, 8),sharey=True)
plt.subplots_adjust(wspace=0.05)

sns.distplot(main_variables ['player_height'], color='g', ax=ax[0], label='_nolegend_', kde=False)
sns.distplot(main_variables ['player_weight'], color='g', ax=ax[1], label='_nolegend_', kde=False)
ax[0].axvline(main_variables ['player_height'].mean(), color='b', label='NBA Mean')
ax[1].axvline(main_variables ['player_weight'].mean(), color='b', label='NBA Mean')

# Add lines for average adults to compare
ax[0].axvline(175.3, color='r', label='Average US Male Adult')
ax[1].axvline(88.8, color='r', label='Average US Male Adult')

ax[0].yaxis.set_label_text('Count')
ax[0].xaxis.set_label_text('Height (cm)')
ax[1].xaxis.set_label_text('Weight (kg)')
plt.suptitle("Height and Weight Distribution", fontsize=22)
plt.legend(loc='upper right', bbox_to_anchor=(0.98, 1.06), frameon=False)
sns.despine(ax=ax[1], left=True)
sns.despine(ax=ax[0])

plt.show()

# Linear Reggresion
plt.figure(figsize=(16, 8))
sns.regplot(x='player_weight', y='player_height', data=main_variables, color='m')
plt.title('Correlation between Height and Weight of Players' , fontsize=22)
plt.ylabel('Height (cm)')
plt.xlabel('Weight (kg)')
sns.despine()

plt.show()

# Removing Non-integer features.
df2 = df.drop(['player_name','team_abbreviation', 'college', 'country', 'draft_year', 'draft_round', 'draft_number', 'season'], axis = 1) 
X = df2.values

# Using the standard scaler method to standardize all of the features by converting them into values between -3 and +3.
X = StandardScaler().fit_transform(X)

# Using Principal Component Analysis or PCA in short to reduce the dimensionality of the data in order to optimize the result of the clustering.
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(X)

PCA_dataset = pd.DataFrame(data = principalComponents, columns = ['component1', 'component2'] )

principal_component1 = PCA_dataset['component1']
principal_component2 = PCA_dataset['component2']

plt.figure()
plt.figure(figsize=(16, 10))
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.title('2  PCA Components')
plt.scatter(PCA_dataset['component1'], PCA_dataset['component2'], color="r")
plt.show()

kmeans = KMeans(n_clusters = 4 , init = 'k-means++', random_state = 1)
y_kmeans = kmeans.fit_predict(principalComponents)


# Plotting the clusters.
plt.figure()
plt.scatter(principalComponents[y_kmeans == 0, 0], principalComponents[y_kmeans == 0, 1], s = 100, c = 'chartreuse', label = 'Cluster 1')
plt.scatter(principalComponents[y_kmeans == 1, 0], principalComponents[y_kmeans == 1, 1], s = 100, c = 'gold', label = 'Cluster 2')
plt.scatter(principalComponents[y_kmeans == 2, 0], principalComponents[y_kmeans == 2, 1], s = 100, c = 'silver', label = 'Cluster 3')
plt.scatter(principalComponents[y_kmeans == 3, 0], principalComponents[y_kmeans == 3, 1], s = 100, c = 'olive', label = 'Cluster 4')

plt.title('Clusters of NBA Players')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()
