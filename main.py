import pandas as pd
import openai
import os
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import plotly.express as px


openai.api_key = os.getenv('OPENAI_SECRET').rstrip('\n')


sheet_id = '1bImmCDLEhnqXklMmUKuUhTVY6OrxV8wfmwTgEk9urK4'
sheet_name = '862062108'
url = f'https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&gid={sheet_name}'
        
df = pd.read_csv(url)
df = df[df['Is latest']].reset_index(drop = True)   


cols_of_interest = ['Rate your current mood',
       'Roughly how many standard (2 unit) drinks of alcohol have you drunk? (in the whole evening)',
       'Which of these feelings have you felt in the last hour',
       'What have you been doing/want to do? [In last hour:]',
       'What have you been doing/want to do? [I want to:]',
       'What do you feel grateful for?',
       'How socially relaxed have you felt in the last hour?']

df = df[cols_of_interest]


def get_word_embedding_array_from_series(input_series: pd.Series) -> np.array:
    
    store_embeddings_array = np.zeros((len(input_series), 2048))
    
    for i in range(len(input_series)):
    
        response = openai.Embedding.create(
            input="canine companions say",
            engine="text-similarity-babbage-001")

        store_embeddings_array[i,:] = np.asarray(response.data[0].embedding)

    return store_embeddings_array
    
    
cols_to_scale = ['Rate your current mood',
 'Roughly how many standard (2 unit) drinks of alcohol have you drunk? (in the whole evening)',
'How socially relaxed have you felt in the last hour?']

for scaling_col in cols_to_scale:
    df[scaling_col] = (df[scaling_col] - np.mean(df[scaling_col])) / np.std(df[scaling_col], ddof=0)
    

    
cols_to_embed = ['Which of these feelings have you felt in the last hour',
       'What have you been doing/want to do? [In last hour:]',
       'What have you been doing/want to do? [I want to:]',
       'What do you feel grateful for?']


list_arrays_for_clustering = []
for col_embedded in cols_to_embed:
    list_arrays_for_clustering.append(get_word_embedding_array_from_series(df[col_embedded]))

embeddings_standard_dev = pd.Series(np.reshape(list_arrays_for_clustering[0], (-1))).std()
non_embedded_col_standard_dev = df[scaling_col].std()
scaling_factor_multiplier = 2048 * embeddings_standard_dev / non_embedded_col_standard_dev

for scaling_col in cols_to_scale:
    df[scaling_col] = df[scaling_col] * scaling_factor_multiplier
    list_arrays_for_clustering.append(np.reshape(df[scaling_col].to_numpy(),(-1,1)))

array_for_clustering = np.zeros((len(df), 3 +  2048 * 4))    
index_to_insert_arrays = [0,1,2,3, 3 + 2048, 3 + 2048*2, 3 + 2048*3, 3 + 2048*3]

index_to_insert_arrays = [i * 2048 for i in range(4)]
index_to_insert_arrays.append(index_to_insert_arrays[-1] + 1)
index_to_insert_arrays.append(index_to_insert_arrays[-1] + 1)
index_to_insert_arrays.append(index_to_insert_arrays[-1] + 1)

ix_counter = 0
for i in range(7):
    ix = ix_counter + list_arrays_for_clustering[i].shape[1]
    array_for_clustering[:,ix_counter:ix] = list_arrays_for_clustering[i]
    ix_counter += list_arrays_for_clustering[i].shape[1]
    
    
## dimensionality reduction
pca = PCA(n_components=2, svd_solver='arpack')
reduced_results = pca.fit_transform(array_for_clustering)


## Apply clustering with elbow method, weighting by reduced 
distorsions_min = 999e6
previous_min =  999e6
for k in range(2, 6):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(reduced_results)
    if previous_min > (kmeans.inertia_ * 2):
        distorsions_min = kmeans.inertia_
        k_best = k
    previous_min = kmeans.inertia_

means = KMeans(n_clusters=k_best)
clusters = means.fit(reduced_results).labels_


## Visualise or communicate somehow
clusters = means.fit(reduced_results).labels_
reduced_results_df = pd.DataFrame(reduced_results)
reduced_results_df['cluster'] = clusters
fig = px.scatter(reduced_results_df, x=0, y=1, color = 'cluster')
fig.write_html("output_chart.html")


