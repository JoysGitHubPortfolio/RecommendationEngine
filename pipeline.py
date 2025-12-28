#Import Dependencies
!pip install implicit
!pip install pyngrok

# Data processing and visuals
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning
from scipy.sparse import csr_matrix
from implicit.als import AlternatingLeastSquares
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# Model Serving
from flask import Flask, request, jsonify
from pyngrok import ngrok
import threading

# Environment Handling
from google.colab import userdata
import warnings
warnings.filterwarnings('ignore')

"""# Ingestion"""
!ls drive/MyDrive/Muzz/
!unzip drive/MyDrive/Muzz/swipes.zip -d swipes_data
df = pd.read_csv("swipes_data/swipes.csv")
df.head()

"""# EDA
## Schema Statistics
"""

print(len(df))
print(df['timestamp'].min())
print(df['timestamp'].max())
df["like"].mean()

unique_users = df[["decidermemberid", "decidergender"]].drop_duplicates().groupby("decidergender").size()
unique_users

"""## Data Cleansing"""

# Observe how many same-gender interactions occur (not valid by front-end constraints)
ff_rows = len(df.loc[(df['decidergender'] == 'F') & (df['othergender'] == 'F')])
mm_rows = len(df.loc[(df['decidergender'] == 'M') & (df['othergender'] == 'M')])
print(ff_rows, mm_rows)

mask_valid = (
    (df["decidergender"] == "M") & (df["othergender"] == "F")
) | (
    (df["decidergender"] == "F") & (df["othergender"] == "M")
)

df = df[mask_valid].copy(deep=True)

like_rates = df.groupby(['decidergender', 'decidermemberid']).agg(num_swipes = ('like', 'size'),
                                                                  num_likes = ('like', 'sum'),
                                                                  like_rate = ('like', 'mean')).reset_index()

like_rates = like_rates[like_rates['like_rate'] > 0].copy(deep=True) # effectively filter out inactive users
like_rates = like_rates[like_rates['like_rate'] < 1].copy(deep=True) # effectively filter out spamming users

print('Number of Valid Users:', len(like_rates), '\n')
like_rates[0:5]

"""## Visualisation"""

def plot_my_member_dists(member_counts: pd.DataFrame = like_rates,
                         variable: str = 'num_swipes',
                         log: bool = False
                         ):

  fig, axes = plt.subplots(1, 2, figsize=(6, 2.5),
                           sharey=True,
                           constrained_layout=True)
  for ax, gender in zip(axes, ['M', 'F']):
      x = member_counts.loc[member_counts['decidergender'] == gender, variable]

      if log:
        x = np.log1p(x) # use log scale and 1+probability to normalise range of values
        log_title = 'log scale'
      else:
        log_title = ''

      if variable == 'num_swipes':
        ax.set_xlim(0,1000)
        bins=500
      if variable == 'num_likes':
        ax.set_xlim(0,250)
        bins=250
      if variable == 'like_rate':
        ax.set_xlim(0,1)
        bins=250

      ax.hist(x, bins=bins, density=True)
      x.plot(kind='kde',
            ax=ax,
            bw_method=0.25)

      ax.set_title(f'{gender} - Distribution:\n{log_title}')
      ax.set_xlabel(f'{variable}')
      ax.set_ylabel('Probability Density')

  plt.show()

plot_my_member_dists(variable='num_swipes')
plot_my_member_dists(variable='num_likes')
plot_my_member_dists(variable='like_rate')

"""# Feature Engineering"""

# keep  deciders whose (gender, id) with non-0, non-spam likes
valid_deciders = like_rates[['decidergender', 'decidermemberid']]

df_filtered = df.merge(
    valid_deciders,
    on=['decidergender', 'decidermemberid'],
    how='inner'
).copy(deep=True)

df_filtered.shape

# check filtered interactions df has only users that had valid like_rate
len(df_filtered['decidermemberid'].unique())

# check retained users have like_rate between (0, 1)
check = (
    df_filtered
    .groupby(['decidergender', 'decidermemberid'])['like']
    .mean()
)

check.min(), check.max()

df_filtered[0:5]

# This means what's the like rate when users saw each other in their queues compared to when they didn't.
# We take all pairs of user IDs and see if they map in reverse.
# This result shows that users are more prone to like each other if they saw each other.

pairs = set(zip(df_filtered.decidermemberid, df_filtered.othermemberid))
df_filtered["reciprocal_possible"] = [
    (o, d) in pairs for d, o in zip(df_filtered.decidermemberid, df_filtered.othermemberid)
]

df_filtered.groupby("reciprocal_possible")["like"].mean()

len(pairs)

"""# Model Build"""

df_filtered.info()

"""## Data Preparation"""

# Filter for likes only (like=1 means positive interaction)
interactions = df_filtered[df_filtered['like'] == 1].copy(deep=True)

# Count interactions per user-item pair (confidence measure)
interaction_counts = interactions.groupby(['decidermemberid', 'othermemberid']).size().reset_index(name='interaction_count')

# Create user and item mappings
unique_users = interaction_counts['decidermemberid'].unique()
unique_items = interaction_counts['othermemberid'].unique()

user_mapping = {user: idx for idx, user in enumerate(unique_users)}
item_mapping = {item: idx for idx, item in enumerate(unique_items)}
reverse_user_mapping = {idx: user for user, idx in user_mapping.items()}
reverse_item_mapping = {idx: item for item, idx in item_mapping.items()}

# Map to matrix indices
interaction_counts['user_idx'] = interaction_counts['decidermemberid'].map(user_mapping)
interaction_counts['item_idx'] = interaction_counts['othermemberid'].map(item_mapping)

print(f"Users: {len(unique_users)}, Items: {len(unique_items)}")
print(f"Total interactions: {len(interaction_counts)}")

"""Note there are about 3M interactions in an M*N space which could have 3B pairwise possibilities. This means when we represent the system the matrix is mostly sparse (zero). ML libraries which utilise collaborative filtering are ideal for this approach. Essentially given some user-other combination, we want to assign a "score" that is predictive of their likelihood to match.

## Train-Test Split & Evaluate
"""

# Hyperparameters (somewhat arbitrary/relative) - could use grid-search in future to optimise.
FACTORS = 64
REGULARIZATION = 0.01
ITERATIONS = 20
ALPHA = 40
TEST_SIZE = 0.2

# Split data into train and test
print(f"\nSplitting data: {int((1-TEST_SIZE)*100)}% train, {int(TEST_SIZE*100)}% test")
train_data, test_data = train_test_split(
    interaction_counts,
    test_size=TEST_SIZE,
    random_state=42
)

# Create train matrix with confidence weighting
train_confidence = 1 + ALPHA * train_data['interaction_count']
train_matrix = csr_matrix(
    (train_confidence,
     (train_data['user_idx'], train_data['item_idx'])),
    shape=(len(user_mapping), len(item_mapping))
)

# Create test matrix
test_confidence = 1 + ALPHA * test_data['interaction_count']
test_matrix = csr_matrix(
    (test_confidence,
     (test_data['user_idx'], test_data['item_idx'])),
    shape=(len(user_mapping), len(item_mapping))
)

# Train the ALS model
print(f"\nTraining ALS model...")
print(f"Parameters: factors={FACTORS}, regularization={REGULARIZATION}, "
      f"iterations={ITERATIONS}, alpha={ALPHA}")

model = AlternatingLeastSquares(
    factors=FACTORS,
    regularization=REGULARIZATION,
    iterations=ITERATIONS,
    random_state=42
)

model.fit(train_matrix)
print("Training complete!")

# Evaluate model performance
print(f"\nEvaluating model (Top-10 recommendations)...")

K = 10
precisions = []
recalls = []

# Group test data by user
test_grouped = test_data.groupby('user_idx')['item_idx'].apply(list).to_dict()

for user_idx, actual_items in test_grouped.items():
    if user_idx >= train_matrix.shape[0]:
        continue

    # Get recommendations
    user_items = train_matrix[user_idx]
    ids, scores = model.recommend(
        user_idx,
        user_items,
        N=K,
        filter_already_liked_items=True
    )

    recommended_items = ids.tolist()

    # Calculate metrics
    hits = len(set(recommended_items) & set(actual_items))
    precision = hits / K if K > 0 else 0
    recall = hits / len(actual_items) if len(actual_items) > 0 else 0

    precisions.append(precision)
    recalls.append(recall)

avg_precision = np.mean(precisions)
avg_recall = np.mean(recalls)

print(f"Precision@{K}: {avg_precision:.4f}")
print(f"Recall@{K}: {avg_recall:.4f}")

"""## Visualisation"""

# Visualize results
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. Precision and Recall Distribution
ax1 = axes[0, 0]
ax1.hist(precisions, bins=10, alpha=0.7, label='Precision', color='blue')
ax1.hist(recalls, bins=10, alpha=0.7, label='Recall', color='orange')
ax1.axvline(avg_precision, color='blue', linestyle='--',
            label=f"Avg Precision: {avg_precision:.3f}")
ax1.axvline(avg_recall, color='orange', linestyle='--',
            label=f"Avg Recall: {avg_recall:.3f}")
ax1.set_xlabel('Score')
ax1.set_ylabel('Frequency')
ax1.set_title('Precision & Recall Distribution')
ax1.legend()
ax1.grid(alpha=0.3)

# 2. Interaction density heatmap (sample)
ax2 = axes[0, 1]
sample_size = min(50, train_matrix.shape[0])
sample_matrix = train_matrix[:sample_size, :sample_size].toarray()
sns.heatmap(sample_matrix, cmap='YlOrRd', ax=ax2, cbar_kws={'label': 'Confidence'})
ax2.set_title(f'User-Item Interaction Heatmap (First {sample_size}x{sample_size})')
ax2.set_xlabel('Items')
ax2.set_ylabel('Users')

# 3. Top users by interaction count
ax3 = axes[1, 0]
user_interactions = test_data.groupby('user_idx').size().sort_values(ascending=False).head(20)
ax3.barh(range(len(user_interactions)), user_interactions.values, color='steelblue')
ax3.set_yticks(range(len(user_interactions)))
ax3.set_yticklabels([f"User {idx}" for idx in user_interactions.index])
ax3.set_xlabel('Number of Interactions')
ax3.set_title('Top 20 Most Active Users (Test Set)')
ax3.invert_yaxis()
ax3.grid(axis='x', alpha=0.3)

# 4. Model performance summary
ax4 = axes[1, 1]
ax4.axis('off')

summary_text = f"""
Model Performance Summary
══════════════════════════

Hyperparameters:
• Factors: {FACTORS}
• Regularization: {REGULARIZATION}
• Iterations: {ITERATIONS}
• Alpha (confidence): {ALPHA}

Dataset:
• Total Users: {len(user_mapping):,}
• Total Items: {len(item_mapping):,}
• Train Interactions: {train_matrix.nnz:,}
• Test Interactions: {len(test_data):,}
• Sparsity: {(1 - train_matrix.nnz / (train_matrix.shape[0] * train_matrix.shape[1])) * 100:.2f}%

Evaluation Metrics (Top-10):
• Precision@10: {avg_precision:.4f}
• Recall@10: {avg_recall:.4f}

Interpretation:
• Precision indicates how many recommended
  items are relevant
• Recall indicates coverage of relevant items
"""

ax4.text(0.1, 0.5, summary_text, fontsize=11, verticalalignment='center',
        fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.tight_layout()
plt.savefig('recommendation_results.png', dpi=300, bbox_inches='tight')
print("\nVisualization saved as 'recommendation_results.png'")
plt.show()

"""# Explainability

This section is not necessary for the model to work. However, doing Principle Components Analysis creates a vectorised represenation of the dataset such that we can account for which "axis" is responsible for some proportion of variation in a dataset. Plotting these components using pre-built t-SNE methods, allows to see how certain behaviours cluster. And visualising clusters allows us to see if one group can be distinguised from another group.
"""

# Get the learned user factor matrix (embeddings): model.user_factors shape: (n_users, n_factors)
user_embeddings = model.user_factors
print(f"Embedding dimensions: {user_embeddings.shape}")
print(f"Number of users: {len(user_mapping)}")
print(f"Embedding size (latent factors): {user_embeddings.shape[1]}")

# Create a dataframe with user IDs and their embeddings
embedding_df = pd.DataFrame(user_embeddings)
embedding_df['user_idx'] = range(len(user_embeddings))
embedding_df['user_id'] = embedding_df['user_idx'].map(reverse_user_mapping)

# Merge with demographic information
user_demographics = df_filtered[['decidermemberid', 'decidergender', 'deciderdobyear', 'decidersignuptimestamp']].copy()
user_demographics = user_demographics.drop_duplicates('decidermemberid')
user_demographics.columns = ['user_id', 'gender', 'dob_year', 'signup_timestamp']

embedding_df = embedding_df.merge(user_demographics, on='user_id', how='left')

# Calculate age and account age
current_year = 2025
embedding_df['age'] = current_year - embedding_df['dob_year']
embedding_df['signup_timestamp'] = pd.to_datetime(embedding_df['signup_timestamp'])
embedding_df['account_age_days'] = (pd.Timestamp('2025-01-01') - embedding_df['signup_timestamp']).dt.days

# Create age groups
embedding_df['age_group'] = pd.cut(embedding_df['age'],
                                    bins=[0, 22, 26, 30, 35, 40, 100],
                                    labels=['18-22', '23-26', '27-30', '31-35', '36-40', '40+'])

# Sample users if dataset is large (for speed)
max_users_to_plot = 5000
if len(user_embeddings) > max_users_to_plot:
    print(f"Sampling {max_users_to_plot} users from {len(user_embeddings)} for visualization speed...")
    sample_idx = np.random.choice(len(user_embeddings), max_users_to_plot, replace=False)
    user_embeddings_sample = user_embeddings[sample_idx]
    embedding_df = embedding_df.iloc[sample_idx].reset_index(drop=True)
else:
    user_embeddings_sample = user_embeddings

# Use t-SNE to reduce to 2D (captures non-linear relationships: Using faster parameters: lower perplexity and iterations
tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(user_embeddings_sample)//4),
            n_iter=500, n_jobs=-1, verbose=0)
embeddings_2d_tsne = tsne.fit_transform(user_embeddings_sample)

# Use PCA to reduce to 2D (captures linear relationships)
pca = PCA(n_components=2, random_state=42)
embeddings_2d_pca = pca.fit_transform(user_embeddings_sample)

# Add to dataframe
embedding_df['tsne_x'] = embeddings_2d_tsne[:, 0]
embedding_df['tsne_y'] = embeddings_2d_tsne[:, 1]
embedding_df['pca_x'] = embeddings_2d_pca[:, 0]
embedding_df['pca_y'] = embeddings_2d_pca[:, 1]

print("Creating visualizations...")
# Create comprehensive visualization
fig = plt.figure(figsize=(20, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# 1. t-SNE colored by gender
ax1 = fig.add_subplot(gs[0, 0])
for gender in embedding_df['gender'].dropna().unique():
    mask = embedding_df['gender'] == gender
    ax1.scatter(embedding_df[mask]['tsne_x'],
               embedding_df[mask]['tsne_y'],
               alpha=0.6, s=20, label=gender)
ax1.set_title('t-SNE: Colored by Gender', fontsize=12, fontweight='bold')
ax1.set_xlabel('t-SNE Dimension 1')
ax1.set_ylabel('t-SNE Dimension 2')
ax1.legend()
ax1.grid(alpha=0.3)

# 2. t-SNE colored by age
ax2 = fig.add_subplot(gs[0, 1])
scatter = ax2.scatter(embedding_df['tsne_x'],
                      embedding_df['tsne_y'],
                      c=embedding_df['age'],
                      cmap='viridis',
                      alpha=0.6, s=20)
plt.colorbar(scatter, ax=ax2, label='Age')
ax2.set_title('t-SNE: Colored by Age', fontsize=12, fontweight='bold')
ax2.set_xlabel('t-SNE Dimension 1')
ax2.set_ylabel('t-SNE Dimension 2')
ax2.grid(alpha=0.3)

# 3. t-SNE colored by age group
ax3 = fig.add_subplot(gs[0, 2])
age_colors = {'18-22': 'purple', '23-26': 'blue', '27-30': 'green',
              '31-35': 'orange', '36-40': 'red', '40+': 'brown'}
for age_group in embedding_df['age_group'].dropna().unique():
    mask = embedding_df['age_group'] == age_group
    ax3.scatter(embedding_df[mask]['tsne_x'],
               embedding_df[mask]['tsne_y'],
               alpha=0.6, s=20, label=age_group,
               color=age_colors.get(age_group, 'gray'))
ax3.set_title('t-SNE: Colored by Age Group', fontsize=12, fontweight='bold')
ax3.set_xlabel('t-SNE Dimension 1')
ax3.set_ylabel('t-SNE Dimension 2')
ax3.legend(loc='best', fontsize=8)
ax3.grid(alpha=0.3)

# 4. PCA colored by gender
ax4 = fig.add_subplot(gs[1, 0])
for gender in embedding_df['gender'].dropna().unique():
    mask = embedding_df['gender'] == gender
    ax4.scatter(embedding_df[mask]['pca_x'],
               embedding_df[mask]['pca_y'],
               alpha=0.6, s=20, label=gender)
ax4.set_title('PCA: Colored by Gender', fontsize=12, fontweight='bold')
ax4.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
ax4.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
ax4.legend()
ax4.grid(alpha=0.3)

# 5. PCA colored by age
ax5 = fig.add_subplot(gs[1, 1])
scatter = ax5.scatter(embedding_df['pca_x'],
                      embedding_df['pca_y'],
                      c=embedding_df['age'],
                      cmap='viridis',
                      alpha=0.6, s=20)
plt.colorbar(scatter, ax=ax5, label='Age')
ax5.set_title('PCA: Colored by Age', fontsize=12, fontweight='bold')
ax5.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
ax5.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
ax5.grid(alpha=0.3)

# 6. t-SNE colored by account age
ax6 = fig.add_subplot(gs[1, 2])
scatter = ax6.scatter(embedding_df['tsne_x'],
                      embedding_df['tsne_y'],
                      c=embedding_df['account_age_days'],
                      cmap='plasma',
                      alpha=0.6, s=20)
plt.colorbar(scatter, ax=ax6, label='Account Age (Days)')
ax6.set_title('t-SNE: Colored by Account Age', fontsize=12, fontweight='bold')
ax6.set_xlabel('t-SNE Dimension 1')
ax6.set_ylabel('t-SNE Dimension 2')
ax6.grid(alpha=0.3)

# 7. Gender-Age interaction (t-SNE)
ax7 = fig.add_subplot(gs[2, 0])
for gender in embedding_df['gender'].dropna().unique():
    for age_group in ['18-22', '23-26', '27-30', '31-35']:
        mask = (embedding_df['gender'] == gender) & (embedding_df['age_group'] == age_group)
        if mask.sum() > 0:
            ax7.scatter(embedding_df[mask]['tsne_x'],
                       embedding_df[mask]['tsne_y'],
                       alpha=0.5, s=15, label=f'{gender} {age_group}')
ax7.set_title('t-SNE: Gender-Age Segments', fontsize=12, fontweight='bold')
ax7.set_xlabel('t-SNE Dimension 1')
ax7.set_ylabel('t-SNE Dimension 2')
ax7.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=7)
ax7.grid(alpha=0.3)

# 8. Density plot
ax8 = fig.add_subplot(gs[2, 1])
from scipy.stats import gaussian_kde
xy = np.vstack([embedding_df['tsne_x'].dropna(), embedding_df['tsne_y'].dropna()])
z = gaussian_kde(xy)(xy)
ax8.scatter(embedding_df['tsne_x'], embedding_df['tsne_y'],
           c=z, s=20, cmap='hot', alpha=0.6)
ax8.set_title('t-SNE: Density Heatmap', fontsize=12, fontweight='bold')
ax8.set_xlabel('t-SNE Dimension 1')
ax8.set_ylabel('t-SNE Dimension 2')
ax8.grid(alpha=0.3)

# 9. Summary statistics
ax9 = fig.add_subplot(gs[2, 2])
ax9.axis('off')

summary_text = f"""
• Gender breakdown:
{embedding_df['gender'].value_counts().to_string()}

• Age distribution:
  Mean: {embedding_df['age'].mean():.1f} years
  Median: {embedding_df['age'].median():.1f} years
  Range: {embedding_df['age'].min():.0f}-{embedding_df['age'].max():.0f}

• Age groups:
{embedding_df['age_group'].value_counts().to_string()}

PCA Variance Explained:
• PC1: {pca.explained_variance_ratio_[0]:.2%}
• PC2: {pca.explained_variance_ratio_[1]:.2%}
• Total: {pca.explained_variance_ratio_[:2].sum():.2%}
"""

ax9.text(0.05, 0.5, summary_text, fontsize=6, verticalalignment='center',
        fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.suptitle('ALS Embedding Space: Demographic Clustering Analysis',
             fontsize=16, fontweight='bold', y=0.995)

plt.savefig('embedding_space_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# Statistical analysis of clustering
print("\n" + "="*60)
print("CLUSTERING ANALYSIS")
print("="*60)

# Analyze separation between genders in embedding space
from scipy.spatial.distance import cdist

for gender1 in embedding_df['gender'].dropna().unique():
    for gender2 in embedding_df['gender'].dropna().unique():
        if gender1 < gender2:  # Avoid duplicates
            emb1 = user_embeddings_sample[embedding_df['gender'] == gender1]
            emb2 = user_embeddings_sample[embedding_df['gender'] == gender2]

            # Sample for efficiency
            if len(emb1) > 1000:
                emb1 = emb1[np.random.choice(len(emb1), 1000, replace=False)]
            if len(emb2) > 1000:
                emb2 = emb2[np.random.choice(len(emb2), 1000, replace=False)]

            # Calculate average distance
            if len(emb1) > 0 and len(emb2) > 0:
                distances = cdist(emb1, emb2, metric='euclidean')
                avg_distance = np.mean(distances)
                print(f"\nAvg distance between {gender1} and {gender2}: {avg_distance:.3f}")

# Analyze age clustering
print("\n" + "="*60)
print("AGE GROUP SEPARATION")
print("="*60)

age_groups = embedding_df['age_group'].dropna().unique()
for i, ag1 in enumerate(age_groups):
    for ag2 in age_groups[i+1:]:
        emb1 = user_embeddings_sample[embedding_df['age_group'] == ag1]
        emb2 = user_embeddings_sample[embedding_df['age_group'] == ag2]

        if len(emb1) > 500:
            emb1 = emb1[np.random.choice(len(emb1), 500, replace=False)]
        if len(emb2) > 500:
            emb2 = emb2[np.random.choice(len(emb2), 500, replace=False)]

        if len(emb1) > 0 and len(emb2) > 0:
            distances = cdist(emb1, emb2, metric='euclidean')
            avg_distance = np.mean(distances)
            print(f"Avg distance between {ag1} and {ag2}: {avg_distance:.3f}")

"""# Deployment

## Local Inference
"""

def generate_match_queue(user_id, queue_length=20):
    # Check if user exists in training data
    if user_id not in user_mapping:
        print(f"New user {user_id} - using popularity-based recommendations")
        # For cold start users, return most popular items
        item_popularity = np.array(train_matrix.sum(axis=0)).flatten()
        top_items_idx = np.argsort(item_popularity)[::-1][:queue_length]

        results = []
        for item_idx in top_items_idx:
            if item_idx in reverse_item_mapping:
                original_id = reverse_item_mapping[item_idx]
                score = float(item_popularity[item_idx])
                results.append((original_id, score))
        return results

    # Get user index
    user_idx = user_mapping[user_id]
    user_items = train_matrix[user_idx]

    # Get recommendations from model
    ids, scores = model.recommend(
        user_idx,
        user_items,
        N=queue_length,
        filter_already_liked_items=True
    )

    # Convert back to original IDs
    results = []
    for item_idx, score in zip(ids, scores):
        original_id = reverse_item_mapping[item_idx]
        results.append((original_id, float(score)))
    return results


# Example usage: Generate a match queue
user_to_serve = df_filtered['decidermemberid'].iloc[0]
desired_queue_length = 20

match_queue = generate_match_queue(user_to_serve, queue_length=desired_queue_length)
for rank, (other_user_id, score) in enumerate(match_queue[:desired_queue_length], 1):
    user_data = df_filtered[df_filtered['decidermemberid'] == other_user_id].iloc[0] if len(df_filtered[df_filtered['decidermemberid'] == other_user_id]) > 0 else None

    if user_data is not None:
        gender = user_data.get('decidergender', 'Unknown')
        age = 2025 - user_data.get('deciderdobyear', 2000)
        print(f"{rank:2d}. User {other_user_id} | Score: {score:.4f} | {gender}, Age {age}")
    else:
        print(f"{rank:2d}. User {other_user_id} | Score: {score:.4f}")

# Test with different queue lengths
print(f"\n{'='*60}")
print("Queue Length Comparison:")
print(f"{'='*60}\n")

for length in [5, 10, 20, 50]:
    queue = generate_match_queue(user_to_serve, queue_length=length)
    avg_score = np.mean([score for _, score in queue])
    print(f"Queue length {length:2d}: {len(queue)} matches | Avg score: {avg_score:.4f}")

"""## Stastical Test of Users (Q2)

NOTE! To gain access to cold-start users we cannot use df_filtered as we excluded where like_rate == 0. Hence, we must refer to the original df containing all interactions to make comparison to see how they are handled.
"""

# ============================================================
# UPDATED GENERATE_MATCH_QUEUE WITH NORMALIZED SCORES
# ============================================================

def generate_match_queue_normalized(user_id, queue_length=20):
    # Check if user exists in training data
    if user_id not in user_mapping:
        # Cold-start: use popularity-based recommendations
        item_popularity = np.array(train_matrix.sum(axis=0)).flatten()
        top_items_idx = np.argsort(item_popularity)[::-1][:queue_length]

        # Normalize popularity scores to 0-10 scale
        max_popularity = item_popularity.max()
        min_popularity = item_popularity[item_popularity > 0].min() if np.any(item_popularity > 0) else 0

        results = []
        for item_idx in top_items_idx:
            if item_idx in reverse_item_mapping:
                original_id = reverse_item_mapping[item_idx]
                # Min-max normalization to 0-10 scale
                if max_popularity > min_popularity:
                    normalized_score = ((item_popularity[item_idx] - min_popularity) /
                                       (max_popularity - min_popularity)) * 10
                else:
                    normalized_score = 5.0  # Default mid-range score
                results.append((original_id, float(normalized_score)))
        return results, 'cold_start'

    # Active user: use collaborative filtering
    user_idx = user_mapping[user_id]
    user_items = train_matrix[user_idx]

    # Get recommendations from model
    ids, scores = model.recommend(
        user_idx,
        user_items,
        N=queue_length,
        filter_already_liked_items=True
    )

    # Convert back to original IDs
    results = []
    for item_idx, score in zip(ids, scores):
        original_id = reverse_item_mapping[item_idx]
        results.append((original_id, float(score)))

    return results, 'active'


# ============================================================
# IDENTIFY USER GROUPS
# ============================================================

# Users with interactions (in training data)
active_users = list(user_mapping.keys())

# Users without interactions (new/cold-start users) - USE ORIGINAL DF
all_users = df['decidermemberid'].unique()
cold_start_users = [u for u in all_users if u not in user_mapping]

print(f"Active users (in training): {len(active_users):,}")
print(f"Cold-start users (not in training): {len(cold_start_users):,}")
print(f"Percentage of users that are cold-start: {len(cold_start_users) / len(all_users) * 100:.1f}%")

# Sample users from each group
np.random.seed(42)
sample_active = np.random.choice(active_users, min(10, len(active_users)), replace=False)
sample_cold_start = np.random.choice(cold_start_users, min(10, len(cold_start_users)), replace=False)

print(f"\nSampled {len(sample_active)} active users and {len(sample_cold_start)} cold-start users")

# ============================================================
# GENERATE RECOMMENDATIONS FOR BOTH GROUPS
# ============================================================

def get_user_info(user_id):
    user_data = df_filtered[df_filtered['decidermemberid'] == user_id]
    if len(user_data) > 0:
        user_data = user_data.iloc[0]
        return {
            'gender': user_data.get('decidergender', 'Unknown'),
            'age': 2025 - user_data.get('deciderdobyear', 2000),
            'num_likes_given': len(df_filtered[(df_filtered['decidermemberid'] == user_id) & (df_filtered['like'] == 1)])
        }
    return {'gender': 'Unknown', 'age': None, 'num_likes_given': 0}

# Generate recommendations for active users
print("\n" + "="*70)
print("ACTIVE USERS (Have Interaction History)")
print("="*70)

active_results = []
for i, user_id in enumerate(sample_active, 1):
    user_info = get_user_info(user_id)
    queue, method = generate_match_queue_normalized(user_id, queue_length=20)
    avg_score = np.mean([score for _, score in queue])

    active_results.append({
        'user_id': user_id,
        'type': 'active',
        'gender': user_info['gender'],
        'age': user_info['age'],
        'num_likes_given': user_info['num_likes_given'],
        'avg_score': avg_score,
        'num_recommendations': len(queue)
    })

    print(f"{i}. User {user_id} ({user_info['gender']}, {user_info['age']}) | "
          f"Given {user_info['num_likes_given']} likes | Avg score: {avg_score:.4f}")
    print(f"   Top 3 recommendations:")
    for rank, (rec_id, score) in enumerate(queue[:3], 1):
        rec_info = get_user_info(rec_id)
        print(f"      {rank}. User {rec_id} ({rec_info['gender']}, {rec_info['age']}) - Score: {score:.4f}")
    print()

# Generate recommendations for cold-start users
print("\n" + "="*70)
print("COLD-START USERS (No Interaction History)")
print("="*70)

cold_start_results = []
for i, user_id in enumerate(sample_cold_start, 1):
    user_info = get_user_info(user_id)
    queue, method = generate_match_queue_normalized(user_id, queue_length=20)
    avg_score = np.mean([score for _, score in queue]) if len(queue) > 0 else 0

    cold_start_results.append({
        'user_id': user_id,
        'type': 'cold_start',
        'gender': user_info['gender'],
        'age': user_info['age'],
        'num_likes_given': 0,
        'avg_score': avg_score,
        'num_recommendations': len(queue)
    })

    print(f"{i}. User {user_id} ({user_info['gender']}, {user_info['age']}) | "
          f"NEW USER | Avg score: {avg_score:.4f}")
    print(f"   Top 3 recommendations:")
    for rank, (rec_id, score) in enumerate(queue[:3], 1):
        rec_info = get_user_info(rec_id)
        print(f"      {rank}. User {rec_id} ({rec_info['gender']}, {rec_info['age']}) - Score: {score:.4f}")
    print()

# ============================================================
# VISUALIZE COMPARISON (WITH PROPER SCALING)
# ============================================================

results_df = pd.DataFrame(active_results + cold_start_results)

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Model Performance: Active Users vs Cold-Start Users (Normalized Scores)',
             fontsize=16, fontweight='bold')

# 1. Average Score Distribution
ax1 = axes[0, 0]
active_scores = [r['avg_score'] for r in active_results]
cold_scores = [r['avg_score'] for r in cold_start_results]

# Use same bins for fair comparison
bins = np.linspace(0, 10, 15)
ax1.hist(active_scores, bins=bins, alpha=0.7, label='Active Users', color='steelblue', edgecolor='black')
ax1.hist(cold_scores, bins=bins, alpha=0.7, label='Cold-Start Users', color='coral', edgecolor='black')
ax1.axvline(np.mean(active_scores), color='steelblue', linestyle='--', linewidth=2,
            label=f'Active Mean: {np.mean(active_scores):.3f}')
ax1.axvline(np.mean(cold_scores), color='coral', linestyle='--', linewidth=2,
            label=f'Cold-Start Mean: {np.mean(cold_scores):.3f}')
ax1.set_xlabel('Average Recommendation Score (0-10 scale)')
ax1.set_ylabel('Frequency')
ax1.set_title('Score Distribution Comparison')
ax1.set_xlim(0, 10)
ax1.legend(fontsize=9)
ax1.grid(alpha=0.3)

# 2. Score Comparison Box Plot
ax2 = axes[0, 1]
data_to_plot = [active_scores, cold_scores]
bp = ax2.boxplot(data_to_plot, labels=['Active Users', 'Cold-Start Users'],
                 patch_artist=True, showmeans=True)
bp['boxes'][0].set_facecolor('steelblue')
bp['boxes'][1].set_facecolor('coral')
ax2.set_ylabel('Average Recommendation Score')
ax2.set_title('Score Distribution (Box Plot)')
ax2.set_ylim(0, 10)
ax2.grid(axis='y', alpha=0.3)

# 3. Number of Likes Given vs Score (Active users only)
ax3 = axes[0, 2]
active_df = results_df[results_df['type'] == 'active']
if len(active_df) > 0 and active_df['num_likes_given'].sum() > 0:
    ax3.scatter(active_df['num_likes_given'], active_df['avg_score'],
               s=100, alpha=0.6, color='steelblue')
    ax3.set_xlabel('Number of Likes Given')
    ax3.set_ylabel('Average Recommendation Score')
    ax3.set_title('User Activity vs Recommendation Quality')
    ax3.set_ylim(0, 10)
    ax3.grid(alpha=0.3)

    # Add trend line
    if len(active_df) > 1:
        z = np.polyfit(active_df['num_likes_given'], active_df['avg_score'], 1)
        p = np.poly1d(z)
        x_trend = np.linspace(active_df['num_likes_given'].min(), active_df['num_likes_given'].max(), 100)
        ax3.plot(x_trend, p(x_trend), "r--", alpha=0.8, linewidth=2, label='Trend')
        ax3.legend()
else:
    ax3.text(0.5, 0.5, 'No activity data available', ha='center', va='center', transform=ax3.transAxes)
    ax3.set_title('User Activity vs Recommendation Quality')

# 4. Score by User Type (Bar Chart)
ax4 = axes[1, 0]
type_means = results_df.groupby('type')['avg_score'].agg(['mean', 'std'])
x_pos = np.arange(len(type_means))
bars = ax4.bar(x_pos, type_means['mean'], yerr=type_means['std'],
               color=['steelblue', 'coral'], alpha=0.7, capsize=10, edgecolor='black')
ax4.set_xticks(x_pos)
ax4.set_xticklabels(['Active Users', 'Cold-Start Users'])
ax4.set_ylabel('Average Score (0-10 scale)')
ax4.set_title('Mean Score by User Type (with Std Dev)')
ax4.set_ylim(0, 10)
ax4.grid(axis='y', alpha=0.3)

# Add value labels on bars
for i, bar in enumerate(bars):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.2f}', ha='center', va='bottom', fontweight='bold')

# 5. Score Range Comparison
ax5 = axes[1, 1]
categories = ['Min', 'Median', 'Max']
active_stats = [np.min(active_scores), np.median(active_scores), np.max(active_scores)]
cold_stats = [np.min(cold_scores), np.median(cold_scores), np.max(cold_scores)]

x = np.arange(len(categories))
width = 0.35

ax5.bar(x - width/2, active_stats, width, label='Active Users', color='steelblue', alpha=0.7, edgecolor='black')
ax5.bar(x + width/2, cold_stats, width, label='Cold-Start Users', color='coral', alpha=0.7, edgecolor='black')

ax5.set_ylabel('Score')
ax5.set_title('Score Statistics Comparison')
ax5.set_xticks(x)
ax5.set_xticklabels(categories)
ax5.set_ylim(0, 10)
ax5.legend()
ax5.grid(axis='y', alpha=0.3)

# 6. Summary Statistics Table
ax6 = axes[1, 2]
ax6.axis('off')

# Calculate percentage difference safely
if np.mean(cold_scores) > 0:
    pct_diff = ((np.mean(active_scores) - np.mean(cold_scores)) / np.mean(cold_scores) * 100)
else:
    pct_diff = 0

summary_text = f"""
Performance Comparison Summary
{'='*40}

Active Users (n={len(active_results)}):
- Mean score: {np.mean(active_scores):.4f}
- Median score: {np.median(active_scores):.4f}
- Std dev: {np.std(active_scores):.4f}
- Score range: {np.min(active_scores):.2f} - {np.max(active_scores):.2f}
- Avg likes given: {np.mean([r['num_likes_given'] for r in active_results]):.1f}
- Method: Collaborative Filtering

Cold-Start Users (n={len(cold_start_results)}):
- Mean score: {np.mean(cold_scores):.4f}
- Median score: {np.median(cold_scores):.4f}
- Std dev: {np.std(cold_scores):.4f}
- Score range: {np.min(cold_scores):.2f} - {np.max(cold_scores):.2f}
- Method: Popularity-based (normalized)

Score Difference:
- Δ = {np.mean(active_scores) - np.mean(cold_scores):.4f}
- % difference: {pct_diff:.1f}%

Key Insight:
{'• Active users: PERSONALIZED matches' if np.mean(active_scores) != np.mean(cold_scores) else '• Scores are similar'}
{'• Cold-start: POPULAR matches' if np.mean(cold_scores) > 0 else ''}
- All scores normalized to 0-10 scale
"""

ax6.text(0.05, 0.5, summary_text, fontsize=9.5, verticalalignment='center',
        fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

plt.tight_layout()
plt.savefig('active_vs_coldstart_normalized.png', dpi=300, bbox_inches='tight')
print("\n✓ Visualization saved as 'active_vs_coldstart_normalized.png'")
plt.show()

# ============================================================
# STATISTICAL COMPARISON
# ============================================================

print("\n" + "="*70)
print("STATISTICAL COMPARISON (Normalized Scores)")
print("="*70)

# T-test to see if scores are significantly different
if len(active_scores) > 1 and len(cold_scores) > 1:
    t_stat, p_value = stats.ttest_ind(active_scores, cold_scores)
    print(f"\nT-test Results:")
    print(f"  t-statistic: {t_stat:.4f}")
    print(f"  p-value: {p_value:.4f}")
    if p_value < 0.05:
        print(f"  ✓ Scores are SIGNIFICANTLY different (p < 0.05)")
        print(f"    → {'Active users get higher scores' if np.mean(active_scores) > np.mean(cold_scores) else 'Cold-start users get higher scores'}")
    else:
        print(f"  ✗ No significant difference in scores (p >= 0.05)")
        print(f"    → Both methods produce similar quality recommendations")

print(f"\nEffect Size (Cohen's d):")
pooled_std = np.sqrt((np.std(active_scores)**2 + np.std(cold_scores)**2) / 2)
cohens_d = (np.mean(active_scores) - np.mean(cold_scores)) / pooled_std if pooled_std > 0 else 0
print(f"  Cohen's d: {cohens_d:.4f}")
if abs(cohens_d) < 0.2:
    print(f"  → Small effect size (scores very similar)")
elif abs(cohens_d) < 0.5:
    print(f"  → Medium effect size (noticeable difference)")
else:
    print(f"  → Large effect size (substantial difference)")

print(f"\nKey Findings:")
print(f"  • {len(cold_start_users):,} / {len(all_users):,} users are cold-start ({len(cold_start_users) / len(all_users) * 100:.1f}%)")
print(f"  • Cold-start handling is CRITICAL for your user base")
print(f"  • {'Personalization adds value' if np.mean(active_scores) > np.mean(cold_scores) + 0.5 else 'Popularity baseline is strong'}")

print("\n✓ Analysis complete!")

"""## Serving via API"""

try:
  if type(userdata.get('NGROK')) == str:
    print('Found secret')
except:
  print('Check environment for secrets')

# (get free token from https://dashboard.ngrok.com/get-started/your-authtoken)
ngrok.set_auth_token(userdata.get('NGROK'))

# Create Flask app
app = Flask(__name__)

@app.route('/recommend', methods=['GET'])
def recommend():
    """
    Get recommendations via GET request
    Usage: curl "http://your-url.ngrok-free.app/recommend?user_id=12345&queue_length=10"
    """
    try:
        user_id = int(request.args.get('user_id'))
        queue_length = int(request.args.get('queue_length', 20))

        # Generate recommendations
        match_queue = generate_match_queue(user_id, queue_length=queue_length)

        # Format response
        recommendations = []
        for rank, (other_user_id, score) in enumerate(match_queue, 1):
            user_data = df_filtered[df_filtered['decidermemberid'] == other_user_id]

            if len(user_data) > 0:
                user_data = user_data.iloc[0]
                recommendations.append({
                    'rank': rank,
                    'user_id': int(other_user_id),
                    'score': round(float(score), 4),
                    'gender': user_data.get('decidergender', 'Unknown'),
                    'age': int(2025 - user_data.get('deciderdobyear', 2000))
                })
            else:
                recommendations.append({
                    'rank': rank,
                    'user_id': int(other_user_id),
                    'score': round(float(score), 4)
                })

        return jsonify({
            'user_id': user_id,
            'queue_length': queue_length,
            'recommendations': recommendations
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400

# Start Flask in background thread
def run_flask():
    app.run(port=5000, use_reloader=False)

threading.Thread(target=run_flask, daemon=True).start()

# Create ngrok tunnel
public_url = ngrok.connect(5000)
print(f"\n{'='*70}")
print(f"API IS LIVE!")
print(f"{'='*70}")
print(f"Public URL: {public_url}")
print(f"\Test from CMD:")
print(f'curl "{public_url}/recommend?user_id={user_to_serve}&queue_length=10"')
print(f"\n{'='*70}\n")

"""## CURL"""

!curl "https://vegetative-lulu-urogenital.ngrok-free.dev/recommend?user_id=3847776&queue_length=10"
