import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay, f1_score
import matplotlib.pyplot as plt
from PIL import Image

def main():
    # Load the training and test datasets
    print('Loading the training and test datasets...')
    train_data = pd.read_csv('test/cleaned_data_train.csv')
    test_data = pd.read_csv('test/cleaned_data_test.csv')
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
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

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

    # Show both confusion_matrix_linear_regression.png and confusion_matrix_logistic_regression.png
    # Combine the two images
    img1 = Image.open('confusion_matrix_linear_regression.png')
    img2 = Image.open('confusion_matrix_logistic_regression.png')
    # Create a new image with twice the width of the two images
    new_img = Image.new('RGB', (img1.width + img2.width, img1.height))
    # Paste the two images side by side
    new_img.paste(img1, (0, 0))
    new_img.paste(img2, (img1.width, 0))
    # Display the combined image
    new_img.show()
    
    
if __name__ == '__main__':
    main()


