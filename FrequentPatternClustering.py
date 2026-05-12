import pandas as pd
import numpy as np
import time
import random
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score
from pyECLAT import ECLAT

# Load the dataset
# (Make sure to uncomment the line below and ensure the filename is correct)
df = pd.read_csv('steam-200k.csv', header=None, names=['user_id', 'game', 'behavior', 'hours', '0'])

# =========================================================
# 3a) Data Preprocessing & Inspection
# =========================================================
print("="*60)
print("--- STEP 1: DATA PREPROCESSING & INSPECTION ---")
print("="*60)

# Drop arbitrary '0' column
df = df.drop('0', axis=1)
print(f"1. Original Dataset Loaded: {df.shape[0]} rows.")

# --- FP Mining Prep ---
print("\n2. Preparing Data for Frequent Pattern (FP) Mining...")
df_purchase = df[df['behavior'] == 'purchase']
transactions = df_purchase.groupby('user_id')['game'].apply(list).tolist()
print(f"   -> Grouped into {len(transactions)} unique user libraries (transactions).")

# One-hot encoding for Apriori/FP-Growth
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df_fp = pd.DataFrame(te_ary, columns=te.columns_)
print(f"   -> One-Hot Encoded Matrix Shape: {df_fp.shape[0]} users x {df_fp.shape[1]} unique games.")
print("   *Notice the high dimensionality (thousands of games)! This causes extreme sparsity.*")

# --- Clustering Prep ---
print("\n3. Preparing Data for Clustering Analysis...")
df_play = df[df['behavior'] == 'play']
games_owned = df_purchase.groupby('user_id')['game'].count().rename('Total_Games_Owned')
hours_played = df_play.groupby('user_id')['hours'].sum().rename('Total_Hours_Played')
max_hours = df_play.groupby('user_id')['hours'].max().rename('Max_Hours_Single_Game')

df_cluster = pd.concat([games_owned, hours_played, max_hours], axis=1).fillna(0)
print("   -> Extracted User Features (Sample Unscaled Row):")
print(f"      {df_cluster.iloc[0].to_dict()}")

# Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_cluster)
df_cluster_scaled = pd.DataFrame(X_scaled, columns=df_cluster.columns, index=df_cluster.index)


# =========================================================
# 3b & 4b) Frequent Pattern Mining & Evaluation
# =========================================================
print("\n" + "="*60)
print("--- STEP 2: FREQUENT PATTERN MINING ---")
print("="*60)

min_sup = 0.02 # 2% support threshold
min_conf = 0.5 # 50% confidence threshold

# 1. Apriori
print("\n[Running Apriori...]")
start_time = time.time()
frequent_itemsets_apriori = apriori(df_fp, min_support=min_sup, use_colnames=True)
rules_apriori = association_rules(frequent_itemsets_apriori, metric="confidence", min_threshold=min_conf)
print(f"-> Finished in {time.time() - start_time:.4f} seconds.")
print(f"-> Found {len(rules_apriori)} rules.")

# 2. FP-Growth (Should be faster!)
print("\n[Running FP-Growth...]")
start_time = time.time()
frequent_itemsets_fpgrowth = fpgrowth(df_fp, min_support=min_sup, use_colnames=True)
rules_fpgrowth = association_rules(frequent_itemsets_fpgrowth, metric="confidence", min_threshold=min_conf)
print(f"-> Finished in {time.time() - start_time:.4f} seconds.")
print(f"-> Found {len(rules_fpgrowth)} rules.")

# Show an example of the rules found
if len(rules_fpgrowth) > 0:
    print("\n   *** Sample Rule Found ***")
    sample_rule = rules_fpgrowth.iloc[0]
    antecedents = list(sample_rule['antecedents'])
    consequents = list(sample_rule['consequents'])
    print(f"   If user buys {antecedents} -> They are {sample_rule['confidence']*100:.1f}% likely to buy {consequents}")

# 3. ECLAT (With the safety fix)
print("\n[Running ECLAT...]")
print("-> Applying heuristic fix: Downsampling to 1500 users & limiting combinations to max 2.")
random.seed(42)
sample_size = min(1500, len(transactions))
sampled_transactions = random.sample(transactions, sample_size)
df_transactions = pd.DataFrame(sampled_transactions)

start_time = time.time()
eclat_instance = ECLAT(data=df_transactions, verbose=False)
rule_indices, rule_supports = eclat_instance.fit(min_support=min_sup, min_combination=1, max_combination=2)
print(f"-> Finished ECLAT in {time.time() - start_time:.4f} seconds.")


# =========================================================
# 3c & 4c) Clustering Analysis & Evaluation
# =========================================================
print("\n" + "="*60)
print("--- STEP 3: CLUSTERING ANALYSIS ---")
print("="*60)

evaluation_results = {}

# 1. K-Means
print("\n[Running K-Means (k=4)...]")
kmeans_model = KMeans(n_clusters=4, random_state=42, n_init=10)
kmeans_clusters = kmeans_model.fit_predict(X_scaled)
df_cluster['KMeans_Cluster'] = kmeans_clusters
evaluation_results['K-Means'] = {
    'Silhouette': silhouette_score(X_scaled, kmeans_clusters),
    'Davies-Bouldin': davies_bouldin_score(X_scaled, kmeans_clusters)
}
print("-> Profiling K-Means Clusters (Averages per group):")
cluster_profiles = df_cluster.groupby('KMeans_Cluster')[['Total_Games_Owned', 'Total_Hours_Played']].mean().round(1)
print(cluster_profiles)

# 2. AGNES (Hierarchical)
print("\n[Running AGNES (k=4)...]")
agnes_model = AgglomerativeClustering(n_clusters=4, metric='euclidean', linkage='ward')
agnes_clusters = agnes_model.fit_predict(X_scaled)
evaluation_results['AGNES'] = {
    'Silhouette': silhouette_score(X_scaled, agnes_clusters),
    'Davies-Bouldin': davies_bouldin_score(X_scaled, agnes_clusters)
}
print("-> AGNES completed successfully.")

# 3. DBSCAN
print("\n[Running DBSCAN...]")
dbscan_model = DBSCAN(eps=0.5, min_samples=10)
dbscan_clusters = dbscan_model.fit_predict(X_scaled)
unique_clusters = len(set(dbscan_clusters)) - (1 if -1 in dbscan_clusters else 0)
outliers = list(dbscan_clusters).count(-1)

print(f"-> DBSCAN found {unique_clusters} core cluster(s) and {outliers} outlier(noise) users.")

if unique_clusters > 1:
    evaluation_results['DBSCAN'] = {
        'Silhouette': silhouette_score(X_scaled, dbscan_clusters),
        'Davies-Bouldin': davies_bouldin_score(X_scaled, dbscan_clusters)
    }
else:
    evaluation_results['DBSCAN'] = {'Silhouette': 'N/A', 'Davies-Bouldin': 'N/A'}

print("\n" + "="*60)
print("--- STEP 4: FINAL CLUSTERING COMPARISON ---")
print("="*60)

cluster_eval_df = pd.DataFrame(evaluation_results).T
print(cluster_eval_df.to_string())