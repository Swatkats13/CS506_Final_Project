import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay, f1_score, classification_report
import matplotlib.pyplot as plt
from PIL import Image
import xgboost as xgb

def main():
    # Load the training and test datasets
    print('Loading the training and test datasets...')
    train_data = pd.read_csv('data/cleaned_data_train.csv')
    test_data = pd.read_csv('data/cleaned_data_test.csv')
    print('Processing the data...')
    # Convert 'Cancer' column to numeric
    train_data['Cancer'] = train_data['Cancer'].astype(int)
    test_data['Cancer'] = test_data['Cancer'].astype(int)

    # Separate features (FPKM) and target (Cancer)
    X_train = train_data.drop(columns=['Cancer', 'Sample'])
    y_train = train_data['Cancer']

    X_test = test_data.drop(columns=['Cancer', 'Sample'])
    y_test = test_data['Cancer']

    # Standardize the features (important for Logistic Regression)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Linear Regression
    print("### Linear Regression ###")
    linear_model = LinearRegression()
    linear_model.fit(X_train_scaled, y_train)
    y_pred_linear = linear_model.predict(X_test_scaled)

    # Evaluate Linear Regression
    mse = mean_squared_error(y_test, y_pred_linear)
    r2 = r2_score(y_test, y_pred_linear)
    # Calculate f1 score
    f1 = f1_score(y_test, (y_pred_linear > 0.5).astype(int))
    print(f"F1 Score: {f1}")
    print(f"Mean Squared Error: {mse}")
    print(f"R^2 Score: {r2}")

    # Coefficient of FPKM for Linear Regression
    print("Linear Regression Coefficients:")
    print(f"FPKM Coefficient: {linear_model.coef_[0]}")
    print(f"Intercept: {linear_model.intercept_}")

    # Confusion Matrix
    y_pred_prob = linear_model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype(int)

    # Compute confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Plot confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[False, True])
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix for Linear Regression')
    # Add the f1 score to the plot
    # plt.text(0, 1, f'F1 Score: {f1}', color='black', fontsize=12, fontweight='bold', ha='bottom')
    # Save the plot
    plt.savefig('confusion_matrix_linear_regression.png')



    # Logistic Regression
    print("\n### Logistic Regression ###")
    logistic_model = LogisticRegression(random_state=0)
    logistic_model.fit(X_train_scaled, y_train)
    y_pred_logistic = logistic_model.predict(X_test_scaled)

    # Evaluate Logistic Regression
    accuracy = accuracy_score(y_test, y_pred_logistic)
    conf_matrix = confusion_matrix(y_test, y_pred_logistic)
    # Calculate f1 score
    f1 = f1_score(y_test, y_pred_logistic)
    print(f"F1 Score: {f1}")
    print(f"Accuracy: {accuracy}")

    # Coefficient of FPKM for Logistic Regression
    print("Logistic Regression Coefficients:")
    print(f"FPKM Coefficient: {logistic_model.coef_[0][0]}")
    print(f"Intercept: {logistic_model.intercept_[0]}")

    y_pred_prob = logistic_model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype(int)

    # Compute confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Plot confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[False, True])
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix for Logistic Regression')
    # Add the f1 score to the plot
    # plt.text(0, 1, f'F1 Score: {f1}', color='black', fontsize=12, fontweight='bold', ha='bottom')
    # Save the plot
    plt.savefig('confusion_matrix_logistic_regression.png')
    
    
    # XGBoost Model
    print("\n### XGBoost Model ###")
    xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    xgb_model.fit(X_train_scaled, y_train)
    
    # Predictions
    y_pred_xgb = xgb_model.predict(X_test_scaled)
    y_pred_prob_xgb = xgb_model.predict_proba(X_test_scaled)[:, 1]  # Probability of positive class
    
    # Evaluate XGBoost
    accuracy = accuracy_score(y_test, y_pred_xgb)
    f1 = f1_score(y_test, y_pred_xgb)
    print(f"Accuracy: {accuracy}")
    print(f"F1 Score: {f1}")

    # Classification Report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_xgb))
    
    # Save the confusion matrix plot for XGBoost
    cm = confusion_matrix(y_test, y_pred_xgb)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[False, True])
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix for XGBoost')
    plt.savefig('confusion_matrix_xgboost.png')
    
    # Random Forest Model
    print("\n### Random Forest Model ###")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train_scaled, y_train)
    
    # Predictions
    y_pred_rf = rf_model.predict(X_test_scaled)
    y_pred_prob_rf = rf_model.predict_proba(X_test_scaled)[:, 1]  # Probability of positive class
    
    # Evaluate Random Forest
    accuracy = accuracy_score(y_test, y_pred_rf)
    f1 = f1_score(y_test, y_pred_rf)
    print(f"Accuracy: {accuracy}")
    print(f"F1 Score: {f1}")

    # Classification Report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_rf))

    # Logistic Regression which genes positively and negatively affect the probability of cancer
    genes = X_train.columns
    gene_weights = logistic_model.coef_[0]
    gene_weights = pd.Series(gene_weights, index=genes)
    gene_weights = gene_weights.sort_values(ascending=False)

    print("\nGenes that positively affect the probability of cancer:")
    print(gene_weights.head(20))

    # XGBoost Feature Importance
    feature_importances = xgb_model.feature_importances_
    feature_importances = pd.Series(feature_importances, index=X_train.columns).sort_values(ascending=False)
    
    # Top 20 most important features
    print("\nTop 20 Features by Importance:")
    print(feature_importances.head(20))

    # Save the confusion matrix plot for Random Forest
    cm = confusion_matrix(y_test, y_pred_rf)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[False, True])
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix for Random Forest')
    plt.savefig('confusion_matrix_random_forest.png')

    # Show both confusion_matrix_linear_regression.png and confusion_matrix_logistic_regression.png
    # Combine the two images
    img1 = Image.open('confusion_matrix_linear_regression.png')
    img2 = Image.open('confusion_matrix_logistic_regression.png')
    img3 = Image.open('confusion_matrix_xgboost.png')
    img4 = Image.open('confusion_matrix_random_forest.png')
    
    # Create a new image with twice the width of the two images
    new_img = Image.new('RGB', (img1.width + img2.width, img1.height + img3.height))
    # Paste the two images side by side
    new_img.paste(img1, (0, 0))
    new_img.paste(img2, (img1.width, 0))
    new_img.paste(img3, (0, img1.height))
    new_img.paste(img4, (img3.width, img1.height))
    # Display the combined image
    new_img.show()
    
    
if __name__ == '__main__':
    main()


