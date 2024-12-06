
# ### Tree Constrution
#
# The clustering approach in tree construction includes a few interesting ideas.
#
# **GMM (Gaussian Mixture Model)**
#
# - Model the distribution of data points across different clusters
# - Optimal number of clusters by evaluating the model's Bayesian Information Criterion (BIC)
#
# **UMAP (Uniform Manifold Approximation and Projection)**
#
# - Supports clustering
# - Reduces the dimensionality of high-dimensional data
# - UMAP helps to highlight the natural grouping of data points based on their similarities
#
# **Local and Global Clustering**
#
# - Used to analyze data at different scales
# - Both fine-grained and broader patterns within the data are captured effectively
#
# **Thresholding**
#
# - Apply in the context of GMM to determine cluster membership
# - Based on the probability distribution (assignment of data points to ≥ 1 cluster)
# ---
#
# Code for GMM and thresholding is from Sarthi et al, as noted in the below two sources:
#
# * [Origional repo]()
# * [Minor tweaks]()
#
# Full credit to both authors.
https://github.com/parthsarthi03/raptor/blob/master/raptor/cluster_tree_builder.py
https://github.com/run-llama/llama_index/blob/main/llama-index-packs/llama-index-packs-raptor/llama_index/packs/raptor/clustering.py
https://arxiv.org/abs/2401.18059
https://github.com/langchain-ai/langchain/blob/master/cookbook/RAPTOR.ipynb
https://medium.com/p/092f23f6d230
