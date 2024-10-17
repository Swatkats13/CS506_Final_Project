# CS506_Final_Project

## Description

This project aims to predict the presence of cancer based on gene expression data from the Gene Expression Omnibus (GEO) database. RNA sequencing data from cancer and non-cancer samples will be analyzed to identify differences in expression profiles. By focusing on a subset of key genes related to cancer development, the project will explore how gene expression patterns differ between cancerous and normal tissues, and whether these patterns can be used to predict the likelihood of cancer.

## Background

Proteins are essential for the fulfillment of critical processes throughout the body. The instructions for making these proteins are encoded in our genes, and proteins are produced through the expression of these genes. When there is a malfunction in this protein-making process, it can lead to diseases such as cancer.

Abnormal gene expression is often a sign of cancer and has been used in personalized medicine to identify targets for precision treatments. The data used to detect abnormal gene expression comes from RNA sequencing (RNA-seq), which measures the frequency of gene expression in cells. RNA-seq provides information about which genes are being expressed and at what levels, based on the presence of RNA. The actual values in RNA-seq data represent counts of RNA fragments that have been mapped (or connected) to specific genes, reflecting how actively a gene is being expressed in a given sample.

## Goals

The main goal is to successfully predict cancer based on the RNA expression levels of a few genes of interest. These genes will be selected based on their known association with cancer-related pathways or their differential expression between cancer and normal tissue samples. The project will -
* Identify key genes with significant expression changes between cancerous and healthy tissues.
* Build predictive models to classify samples as cancerous or non-cancerous based on RNA expression levels.

## Data Collection

Data will be sourced from the publicly available Gene Expression Omnibus (GEO) repository. Specific datasets containing RNA sequencing data from cancer patients and healthy controls will be identified and downloaded. These datasets typically include thousands of genes, but the focus will be on a subset of genes that are strongly linked to cancer progression, based on existing literature or feature selection techniques.
Steps to collect data - 
1. Identify suitable GEO datasets containing both cancer and healthy control samples.
2. Filter the data to focus on a few key genes (approximately 10-20 genes) related to cancer biology.
3. Preprocess the data to remove noise and normalize gene expression values for further analysis.

## Modeling Approach

To model the data, several machine learning methods will be explored -
* Logistic Regression - A simple linear model will be used as a baseline to assess how well gene expression levels can predict cancer.
* Decision Trees or Random Forests - Non-linear models will be used to capture complex interactions between gene expression levels and cancer classification.
* Support Vector Machines (SVM) - SVM is a method to classify samples based on their gene expression profiles, particularly useful for binary classification tasks like cancer detection.
* XGBoost or Gradient Boosting Machines - This are tree-based ensemble methods that could potentially outperform simpler models.
* Deep Learning (optional) - If needed, a neural network might be employed to identify complex patterns in gene expression data.

The choice of the model will depend on the performance in predicting cancer from the training data.

## Data Visualization

To visualize the differences in gene expression between cancerous and normal tissues, the following techniques will be used - 
* Box plots to compare the distribution of expression levels for each selected gene between cancer and non-cancer groups.
* Heatmaps to display the relative expression levels of multiple genes across different samples, highlighting patterns of differential expression.
* t-SNE or PCA plots to project high-dimensional gene expression data into a lower-dimensional space, providing visual insights into how cancerous and non-cancerous samples cluster based on their gene expression profiles.
* Interactive scatter plots to visualize the relationship between two or more gene expression levels and how these relate to cancer classification.

## Test Plan

The test plan involves splitting the data into training and testing sets - 
* 80-20 split - 80% of the dataset will be used for training the model, and the remaining 20% will be withheld for testing the model’s performance.
* Cross-validation - To ensure the robustness of the model, k-fold cross-validation (with k=5 or 10) will be employed during training.
* Performance Metrics - The model will be evaluated based on accuracy, precision, recall and F1-score, as cancer detection often involves trade-offs between false positives and false negatives.
* Test on external data - If possible, after building the model on one dataset, it will be tested on a different GEO dataset to check the generalization of the model.

## Conclusion

This project will provide insights into how well RNA expression profiles can predict cancer. By focusing on specific genes known to play a role in cancer development, the project will seek to identify patterns in gene expression that can be used as biomarkers for cancer detection. The work will contribute to the understanding of cancer biology and the potential application of machine learning techniques in genomics-based cancer prediction.
