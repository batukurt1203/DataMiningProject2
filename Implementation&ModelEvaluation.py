import time
import pandas as pd
from mlxtend.frequent_patterns import association_rules
from sklearn.metrics import silhouette_score, davies_bouldin_score

from FrequentPatternClustering import X_scaled, rule_indices, frequent_itemsets_fpgrowth, frequent_itemsets_apriori, \
    agnes_clusters, dbscan_clusters, kmeans_clusters

print("--- 4a, 4b & 4c) Implementation, Evaluation, and Comparison ---")

# ==========================================
# 4b) Evaluating Frequent Pattern Mining
# ==========================================
print("\n--- Evaluating Frequent Pattern Mining Algorithms ---")
# To truly compare them, we often look at execution time and rule generation.
# We will generate association rules from the frequent itemsets found in Step 3.
min_confidence = 0.5

# 1. Apriori Rules
start_time = time.time()
rules_apriori = association_rules(frequent_itemsets_apriori, metric="confidence", min_threshold=min_confidence)
apriori_time = time.time() - start_time
print(f"Apriori: Generated {len(rules_apriori)} rules in {apriori_time:.4f} seconds.")

# 2. FP-Growth Rules
start_time = time.time()
rules_fpgrowth = association_rules(frequent_itemsets_fpgrowth, metric="confidence", min_threshold=min_confidence)
fpgrowth_time = time.time() - start_time
print(f"FP-Growth: Generated {len(rules_fpgrowth)} rules in {fpgrowth_time:.4f} seconds.")

# 3. ECLAT Rules (Note: pyECLAT outputs support directly, so we just check the output size)
print(f"ECLAT: Generated {len(rule_indices)} frequent itemsets combinations.")

# ==========================================
# 4c) Evaluating Clustering Analysis
# ==========================================
print("\n--- Evaluating Clustering Analysis Algorithms ---")
# We use the Silhouette Score (1 is best, -1 is worst) and
# Davies-Bouldin Index (lower is better) to evaluate cluster quality.

evaluation_results = {}

# 1. K-Means Evaluation
sil_kmeans = silhouette_score(X_scaled, kmeans_clusters)
db_kmeans = davies_bouldin_score(X_scaled, kmeans_clusters)
evaluation_results['K-Means'] = {'Silhouette': sil_kmeans, 'Davies-Bouldin': db_kmeans}

# 2. AGNES Evaluation
sil_agnes = silhouette_score(X_scaled, agnes_clusters)
db_agnes = davies_bouldin_score(X_scaled, agnes_clusters)
evaluation_results['AGNES'] = {'Silhouette': sil_agnes, 'Davies-Bouldin': db_agnes}

# 3. DBSCAN Evaluation
# DBSCAN can sometimes classify everything as noise (-1) or a single cluster
# if eps/min_samples are poorly tuned, which breaks the silhouette score.
unique_clusters = len(set(dbscan_clusters)) - (1 if -1 in dbscan_clusters else 0)
if unique_clusters > 1:
    sil_dbscan = silhouette_score(X_scaled, dbscan_clusters)
    db_dbscan = davies_bouldin_score(X_scaled, dbscan_clusters)
    evaluation_results['DBSCAN'] = {'Silhouette': sil_dbscan, 'Davies-Bouldin': db_dbscan}
else:
    evaluation_results['DBSCAN'] = {'Silhouette': 'N/A (1 or 0 valid clusters)', 'Davies-Bouldin': 'N/A'}

# Display Clustering Results
cluster_eval_df = pd.DataFrame(evaluation_results).T
print("\nClustering Performance Metrics:")
print(cluster_eval_df.to_string())