import pickle
import pandas as pd
import seaborn as sns
from os.path import exists
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def main():
    print('Processing GSE68086_series_matrix...')
    series_matrix = pd.read_csv('data/GSE68086_series_matrix.csv')
    
    # Keep only the sample source and cancer type columns
    series_matrix_source_cancer = series_matrix[['!Sample_source_name_ch1', '!Sample_characteristics_ch1.3']]
    # Rename the columns
    series_matrix_source_cancer.columns = ['Sample', 'Cancer_Type']
    # Delete the double quotes in the source
    series_matrix_source_cancer['Sample'] = series_matrix_source_cancer['Sample'].str.replace('"', '')
    # Delete the cancer type in str
    series_matrix_source_cancer['Cancer_Type'] = series_matrix_source_cancer['Cancer_Type'].str.replace('cancer type: ', '')
    series_matrix_source_cancer['Cancer_Type'] = series_matrix_source_cancer['Cancer_Type'].str.replace('"', '')
    # One hot encode the cancer type
    patient_data = pd.get_dummies(series_matrix_source_cancer, columns=['Cancer_Type'])
    
    print('Making the binary cancer column...')
    patient_data['Cancer'] = ~patient_data['Cancer_Type_HC']
    
    cancer_type_columns = [col for col in patient_data.columns if col.startswith('Cancer_Type_')]
    patient_data.drop(columns=cancer_type_columns, inplace=True)

    print('Splitting the data...')
    # Split the data, keep the portion of the data with cancer type
    X_train, X_test, y_train, y_test = train_test_split(patient_data.drop(columns=['Cancer']), patient_data['Cancer'], test_size=0.2, random_state=42)

    # Combine the features and target back into DataFrames for training and testing sets
    train_data = pd.concat([X_train, y_train], axis=1)
    test_data = pd.concat([X_test, y_test], axis=1)

    print('Processing GSE68086_TEP_data_matrix...')
    rna_expression = pd.read_csv('data/GSE68086_TEP_data_matrix.csv', index_col=0)

    # Reset the index to make the index a regular column
    rna_expression.reset_index(inplace=True)
    # Rename the index column to 'gene_ID'
    rna_expression.rename(columns={'index': 'gene_ID'}, inplace=True)

    print('Merging the data...')
    # Reshape data from wide to long format
    rna_expression_long = rna_expression.melt(id_vars=['gene_ID'], var_name='Sample', value_name='FPKM')

    # Merge the patient data with the rna expression data
    train_data = pd.merge(rna_expression_long, train_data, on='Sample')
    test_data = pd.merge(rna_expression_long, test_data, on='Sample')
    
    print('Normalizing the data...')
    # Calculate the mean and variance of each gene in the training set
    gene_stats = train_data.groupby('gene_ID')['FPKM'].agg(['mean', 'var'])
    # Use the mean and variance to normalize the gene expression values to -1 to 1
    train_data = pd.merge(train_data, gene_stats, on='gene_ID')
    train_data['FPKM_normalized'] = (train_data['FPKM'] - train_data['mean']) / np.sqrt(train_data['var'] + 1e-10)
    test_data = pd.merge(test_data, gene_stats, on='gene_ID')
    test_data['FPKM_normalized'] = (test_data['FPKM'] - test_data['mean']) / np.sqrt(test_data['var'] + 1e-10)
    # Drop the original FPKM column
    train_data.drop(columns=['FPKM'], inplace=True)
    test_data.drop(columns=['FPKM'], inplace=True)
    # Display the first few rows of the training data
    # Write the data to a csv file
    cleaned_data_train = pd.DataFrame(columns=['Sample', 'Cancer'])
    cleaned_data_test = pd.DataFrame(columns=['Sample', 'Cancer'])

    # Pivot the train_data DataFrame
    pivoted_train_data = train_data.pivot_table(index=['Sample', 'Cancer'], columns='gene_ID', values='FPKM_normalized', fill_value=0).reset_index()
    pivoted_test_data = test_data.pivot_table(index=['Sample', 'Cancer'], columns='gene_ID', values='FPKM_normalized', fill_value=0).reset_index()

    # Merge the pivoted data with the cleaned_data_train DataFrame
    cleaned_data_train = pd.merge(cleaned_data_train, pivoted_train_data, on=['Sample', 'Cancer'], how='outer')
    cleaned_data_test = pd.merge(cleaned_data_test, pivoted_test_data, on=['Sample', 'Cancer'], how='outer')

    print('Writing the data to csv...')
    # write to csv
    cleaned_data_train.to_csv('data/cleaned_data_train.csv', index=False)
    cleaned_data_test.to_csv('data/cleaned_data_test.csv', index=False)
    print('Done!')


if __name__ == '__main__':
    main()