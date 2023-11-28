

# Importing Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
import tensorflow as tf
from wordcloud import WordCloud
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, BaggingClassifier
import warnings
warnings.simplefilter('ignore')
sns.set_theme(style="dark")

# Utility Functions
def remove_outliers(df, feature):
    """
    Remove Outliers using IRQ method

    df: dataframe
    feature: dataframe column"""
    q1 = df[feature].quantile(0.25)
    q3 = df[feature].quantile(0.75)
    iqr = q3 - q1
    upper_bound = q3 + 1.5 * iqr
    lower_bound = q1 - 1.5 * iqr
    df = df[(df[feature] >= lower_bound) & (df[feature] <= upper_bound)]
    return df

def histogram_boxplot(data, feature, figsize=(15, 10), kde=False, bins=None):

    f2, (ax_box2, ax_hist2) = plt.subplots(
        nrows=2,
        sharex=True,
        gridspec_kw={"height_ratios": (0.25, 0.75)},
        figsize=figsize,
    )
    sns.boxplot(
        data=data, x=feature, ax=ax_box2, showmeans=True, color="violet"
    )
    sns.histplot(
        data=data, x=feature, kde=kde, ax=ax_hist2, bins=bins
    ) if bins else sns.histplot(
        data=data, x=feature, kde=kde, ax=ax_hist2
    )
    ax_hist2.axvline(
        data[feature].mean(), color="green", linestyle="--"
    )
    ax_hist2.axvline(
        data[feature].median(), color="black", linestyle="-"
    )

def labeled_barplot(data, feature, perc=False, n=None):


    total = len(data[feature])
    count = data[feature].nunique()
    if n is None:
        plt.figure(figsize=(count + 2, 6))
    else:
        plt.figure(figsize=(n + 2, 6))

    plt.xticks(rotation=90, fontsize=15)
    ax = sns.countplot(
        data=data,
        x=feature,
        palette="Paired",
        order=data[feature].value_counts().index[:n],
    )

    for p in ax.patches:
        if perc == True:
            label = "{:.1f}%".format(
                100 * p.get_height() / total
            )
        else:
            label = p.get_height()

        x = p.get_x() + p.get_width() / 2
        y = p.get_height()

        ax.annotate(
            label,
            (x, y),
            ha="center",
            va="center",
            size=12,
            xytext=(0, 5),
            textcoords="offset points",
        )

    plt.show()

def stacked_barplot(data, predictor, target):

    count = data[predictor].nunique()
    sorter = data[target].value_counts().index[-1]
    tab1 = pd.crosstab(data[predictor], data[target], margins=True).sort_values(
        by=sorter, ascending=False
    )
    print(tab1)
    print("-" * 120)
    tab = pd.crosstab(data[predictor], data[target], normalize="index").sort_values(
        by=sorter, ascending=False
    )
    tab.plot(kind="bar", stacked=True, figsize=(count + 5, 5))
    plt.legend(
        loc="lower left", frameon=False,
    )
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.show()

def distribution_plot_wrt_target(data, predictor, target):

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    target_uniq = data[target].unique()

    axs[0, 0].set_title("Distribution of target for target=" + str(target_uniq[0]))
    sns.histplot(
        data=data[data[target] == target_uniq[0]],
        x=predictor,
        kde=True,
        ax=axs[0, 0],
        color="teal",
        stat="density",
    )

    axs[0, 1].set_title("Distribution of target for target=" + str(target_uniq[2]))
    sns.histplot(
        data=data[data[target] == target_uniq[2]],
        x=predictor,
        kde=True,
        ax=axs[0, 1],
        color="orange",
        stat="density",
    )

    axs[1, 0].set_title("Boxplot w.r.t target")
    sns.boxplot(data=data, x=target, y=predictor, ax=axs[1, 0], palette="gist_rainbow")

    axs[1, 1].set_title("Boxplot (without outliers) w.r.t target")
    sns.boxplot(
        data=data,
        x=target,
        y=predictor,
        ax=axs[1, 1],
        showfliers=False,
        palette="gist_rainbow",
    )

    plt.tight_layout()
    plt.show()

def checking_overfitting_undefitting(y_train, y_train_pred, y_test, y_test_pred):
    """
    Print whether the model is underfit, overfit or good fit.

    y_train = training data
    y_train_pred = predictions on training data
    y_test = testing data
    y_test_pred = predictions on testing data
    """
    training_accuracy = accuracy_score(y_train, y_train_pred)
    testing_accuracy = accuracy_score(y_test, y_test_pred)
    if training_accuracy<=0.65:
        print("Model is underfitting.")
    elif training_accuracy>0.65 and abs(training_accuracy-testing_accuracy)>0.15:
        print("Model is overfitting.")
    else:
        print("Model is not underfitting/overfitting.")

def calculate_classification_metrics(y_true, y_pred, algorithm):
    """
    Return the classification Metrics

    y_true = actual values
    y_pred = predicted values
    y_pred_probability = probability values
    algorithm = algorithm name
    """
    accuracy = round(accuracy_score(y_true, y_pred), 3)
    precision = round(precision_score(y_true, y_pred, average='weighted'), 3)
    recall = round(recall_score(y_true, y_pred, average='weighted'), 3)
    f1 = round(f1_score(y_true, y_pred, average='weighted'), 3)
    print("Algorithm: ", algorithm)
    print()
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    print()
    cm = confusion_matrix(y_true, y_pred)
    labels = ['Overcast', 'Clear','Foggy']
    plt.figure(figsize=(10, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
    return accuracy, precision, recall, f1

# Callback function to avoid overfitting
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('val_accuracy')>0.90) and (logs.get('accuracy')>0.95):
            print("\nValidation and training accuracies are high so cancelling training!")
            self.model.stop_training = True

"""---

### 1. Data Inspection
"""

# Fetching data
data = pd.read_csv("/content/weatherHistory.csv")
data.head()

# Analyzing the Data types and Exploring the number of entities in a feature
data.info()

# Checking Statistical Summary
data.describe()

# Checking Target Variable
print(data["Summary"].value_counts())

# Reduced Data (Using only 3 classes)
data = data[(data["Summary"] == "Overcast") | (data["Summary"] == "Clear") | (data["Summary"] == "Foggy")]
data.info()

"""---

### 2. Data Cleaning

* #### Missing Values Treatment
"""

# Calculating Missing Values
missing_values_count = data.isnull().sum()
missing_values_count

# Since 359 is a reasonable count. Dropping the respective rows. If the count were smaller we would've filled it up with dummy values
data.dropna(inplace=True)
# Again checking for values
missing_values_count = data.isnull().sum()
missing_values_count

"""* #### Duplicated Values Treatment"""

# Calculating number of duplicated values
print("Duplicated Values: ",data.duplicated().sum())

# Removing duplicated values
data.drop_duplicates(inplace=True)
# Again checking for duplicated values
print("Duplicated Values: ", data.duplicated().sum())

"""* #### Data Formatting"""

# Rounding up the float64 data upto 2 decimals.
float_cols = data.select_dtypes(include='float')
data[float_cols.columns] = float_cols.round(2)
data.head()

# Formatting Date Column. This can be used to identify any seasonality and trends
data['Formatted Date'] = pd.to_datetime(data['Formatted Date'], errors='coerce')
# Extracting the relevant components
data["Time"] = [d.time() for d in data['Formatted Date']]
data["Time"] = data["Time"].astype(str)
data["Time"] = data["Time"].str.split(':').str[0].astype(int)
data["Date"] = [d.date() for d in data['Formatted Date']]
data["Date"]= data["Date"].astype(str)
data["Year"] = data["Date"].str.split('-').str[0].astype(int)
data["Month"] = data["Date"].str.split('-').str[1].astype(int)
data["Day"] = data["Date"].str.split('-').str[2].astype(int)
# Dropping the original column
data = data.drop(columns=['Formatted Date','Date'], axis=1)

"""* #### Redundant Columns Treatment"""

# It can be seen that the feature "Loud Cover" have only value '0' and mean and other statistical overview also support the deduction. Hence, it is the redundant column
data["Loud Cover"].value_counts()

# Removing 'Loud Cover'
data.drop(columns=["Loud Cover"], axis=1, inplace=True)
data.head()

"""* #### Outlier Treatment"""

# Different types of columns
numeric_columns = list(data.select_dtypes(include=['float64', 'int64']).columns)
categorical_columns = list(data.select_dtypes(include=['object']).columns)
continuous_columns = [i for i in numeric_columns if len(list(data[i].unique()))>=25]
discrete_columns = [i for i in numeric_columns if len(list(data[i].unique()))<25]
print("Numerical Columns: ", numeric_columns)
print()
print("Categorical Columns: ", categorical_columns)
print()
print("Continuous Columns: ", continuous_columns)
print()
print("Discrete Columns: ", discrete_columns)

# Plotting for outliers in the data.
plt.figure(figsize=(18, 8))
sns.boxplot(data=data[numeric_columns])
plt.show()

# Removing Outliers
data = remove_outliers(data, "Pressure (millibars)")
data = remove_outliers(data, "Wind Speed (km/h)")
data = remove_outliers(data, "Humidity")
data = remove_outliers(data, "Temperature (C)")
data = remove_outliers(data, "Apparent Temperature (C)")

plt.figure(figsize=(18, 8))
sns.boxplot(data=data[numeric_columns])
plt.show()

"""---

### 3. Exploratory Data Analysis

* #### Checking skewness of the numerical features
"""

# Numerical columns analysis
for i in numeric_columns:
    histogram_boxplot(data,i)

"""* #### Checking distribution of categorical features(Summary and Precip Type)"""

# Categorical columns analysis
for i in categorical_columns:
    if i in ['Daily Summary','Time']:
        pass
    else:
        labeled_barplot(data, i)

"""* #### Relations between numerical features & Target variable "Summary"
"""

# Multivariate analysis
for i in numeric_columns:
    distribution_plot_wrt_target(data, i, "Summary")

"""* #### Relations between categorical features & Target variable "Summary"
"""

# Stacked barplot
stacked_barplot(data,"Precip Type" , 'Summary')

"""---

### 4. Data Pre-processing & Feature Engineering

* #### Classes Distribution
"""

# Checking whether the target variable is balanced or unbalanced
counts = data["Summary"].value_counts()
total = counts.sum()
percentages = (counts / total) * 100
print(percentages)
print()
print("The classes are satifactory balanced")

"""* #### Dataset Split"""

# Input features dataset
input_df = data.drop(columns="Summary", axis=1)
input_df.head()

# Target variable
# Applying mapping
encoder = LabelEncoder()
y = data["Summary"]
y = encoder.fit_transform(y)

# Checking the mapping of the classes
class_mapping = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))
for class_label, class_number in class_mapping.items():
    print(f"Class '{class_label}' is labeled as {class_number}")

"""* #### Categorical Features Encoding"""

# As "Precip Type" have only 2 values, hence applying binary encoding
mapping = {'rain': 0, 'snow': 1}
input_df['Precip Type'] = input_df['Precip Type'].map(mapping)

# As "Daily Summary" have 221 unique values, hence applying Frequency encoding
# Creating a new column for frequency encoding and removing previous column
input_df['Daily Summary Frequency'] = input_df['Daily Summary'].map(input_df['Daily Summary'].value_counts(normalize=True))
input_df.drop(columns=['Daily Summary'], axis=1, inplace=True)
# Checking data
input_df.head()

"""* #### Checking Multicollinearity"""

# Confirming multicollinearity using heatmap
sns.set(style="white")
plt.figure(figsize=(12,8))
sns.heatmap(input_df.corr(), annot=True, cmap='coolwarm', linewidths=.5)
plt.title('Correlation Heatmap')
plt.show()

# As Daily Summary Frequency has a negligible correlation with the other features and it is apparent that the final target is not going to be affected by this, removing it.
input_df.drop(['Daily Summary Frequency'], axis=1, inplace=True)

# As VIF of Temperature (C) is the highest and Temperature is highly correlated with Apparent Temperature, removing it
input_df.drop(['Temperature (C)'], axis=1, inplace=True)

"""* #### Train-Test Split"""

# Creating X input set
X = input_df.values
X

# Splitting the data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

"""* #### Normalizing Input Features"""

# Apply scaling on the input_df DataFrame
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
joblib.dump(scaler, "scaler.pkl")

"""---

### 5. Model Building

* #### Parametric Algorithm 1: Logistic Regression Classifier
"""

# Hyperparameter tuning
parameters = {'solver': ['liblinear', 'saga'],
              'multi_class':['ovr', 'multinomial'],
              'C':[0.001, 0.01, 10.0],
              'penalty': ['l1', 'l2']}
# Model Creation and Training
model_lr = LogisticRegression(n_jobs=-1)
models_lr = GridSearchCV(estimator=model_lr, param_grid=parameters, cv=4)
models_lr.fit(x_train, y_train)
best_parameters = models_lr.best_params_
print("Best Hyperparameters:", best_parameters)
print()
# Predictions for train
best_model_lr = models_lr.best_estimator_
y_pred_lr = best_model_lr.predict(x_train)
# Predictions for test
y_pred_lr_new = best_model_lr.predict(x_test)
checking_overfitting_undefitting(y_train, y_pred_lr, y_test, y_pred_lr_new)

# Evaluation Metrics Calculation
print("Testing Performance")
accuracy_lr, precision_lr, recall_lr, f1_lr = calculate_classification_metrics(y_test, y_pred_lr_new, "Logistic Regression")

"""* #### Parametric Algorithm 2: Gaussian Naive Bayes Classifier"""

# Hyperparameter tuning
parameters = {'var_smoothing':[1e-9, 1e-8, 1e-10]}
# Model Creation and Training
model_nb = GaussianNB()
models_nb = GridSearchCV(estimator=model_nb, param_grid=parameters, cv=4)
models_nb.fit(x_train, y_train)
best_parameters = models_nb.best_params_
print("Best Hyperparameters:", best_parameters)
print()
# Predictions on training data
best_model_nb = models_nb.best_estimator_
y_pred_nb = best_model_nb.predict(x_train)
# Predictions on test data
y_pred_nb_new = best_model_nb.predict(x_test)
checking_overfitting_undefitting(y_train, y_pred_nb, y_test, y_pred_nb_new)

# Evaluation Metrics Calculation
print("Testing Performance")
accuracy_nb, precision_nb, recall_nb, f1_nb = calculate_classification_metrics(y_test, y_pred_nb_new, "Gaussian NB")

"""* #### Parametric Algorithm 3: Support Vector Machine (SVM) Classifier"""

# Hyperparameter tuning
parameters = {'loss':['log_loss','perceptron','hinge','squared_epsilon_insensitive'],
              'penalty': ['l1', 'l2'],
              'alpha':[0.001,0.01,0.0001],
              'learning_rate':['optimal','adaptive','invscaling']}
# Model Creation and Training
model_svc = SGDClassifier()
models_svc = GridSearchCV(estimator=model_svc, param_grid=parameters, cv=4)
models_svc.fit(x_train, y_train)
best_parameters = models_svc.best_params_
print("Best Hyperparameters:", best_parameters)
print()
# Predictions on train data
best_model_svc = models_svc.best_estimator_
y_pred_svc = best_model_svc.predict(x_train)
# Predictions on test data
y_pred_svc_new = best_model_svc.predict(x_test)
checking_overfitting_undefitting(y_train, y_pred_svc, y_test, y_pred_svc_new)

# Evaluation Metrics Calculation
print("Testing Performance")
accuracy_svc, precision_svc, recall_svc, f1_svc = calculate_classification_metrics(y_test, y_pred_svc_new, "SVC")

"""* #### Parametric Algorithm 4: SGD Classifier"""

# Hyperparameter tuning
parameters = {'loss':['log_loss','perceptron','hinge','squared_epsilon_insensitive'],
              'penalty': ['l1', 'l2'],
              'alpha':[0.001,0.01,0.0001],
              'learning_rate':['optimal','adaptive','invscaling']}
# Model Creation and Training
model_sgd = SGDClassifier()
models_sgd = GridSearchCV(estimator=model_sgd, param_grid=parameters, cv=4)
models_sgd.fit(x_train, y_train)
best_parameters = models_sgd.best_params_
print("Best Hyperparameters:", best_parameters)
print()
# Predictions on train data
best_model_sgd = models_sgd.best_estimator_
y_pred_sgd = best_model_sgd.predict(x_train)
# Predictions on test data
y_pred_sgd_new = best_model_sgd.predict(x_test)
checking_overfitting_undefitting(y_train, y_pred_sgd, y_test, y_pred_sgd_new)

# Evaluation Metrics Calculation
print("Testing Performance")
accuracy_sgd, precision_sgd, recall_sgd, f1_sgd = calculate_classification_metrics(y_test, y_pred_sgd_new, "SGD Classifier")

"""* #### Non-Parametric Algorithm 1: Decision Tree Classifier"""

# Hyperparameter tuning
parameters = {'criterion':['gini', 'entropy', 'log_loss'],
              'max_depth': [None, 5, 10],
              'min_samples_split': [None, 2, 5],
              'splitter':['best','random']}
# Model Creation and Training
model_dt = DecisionTreeClassifier()
models_dt = GridSearchCV(estimator=model_dt, param_grid=parameters, cv=4)
models_dt.fit(x_train, y_train)
best_parameters = models_dt.best_params_
print("Best Hyperparameters:", best_parameters)
print()
# Predictions on train data
best_model_dt = models_dt.best_estimator_
y_pred_dt = best_model_dt.predict(x_train)
# Predictions on test data
y_pred_dt_new = best_model_dt.predict(x_test)
checking_overfitting_undefitting(y_train, y_pred_dt, y_test, y_pred_dt_new)

# Evaluation Metrics Calculation
print("Testing Performance")
accuracy_dt, precision_dt, recall_dt, f1_dt = calculate_classification_metrics(y_test, y_pred_dt_new, "Decision Tree")

"""* #### Non-Parametric Algorithm 2: K Nearest Neighbours Classifier"""

# Hyperparameter tuning
parameters = {'weights': ['uniform', 'distance'],
            'algorithm': ['auto', 'ball_tree','kd_tree','brute'],
            'p': [1,2]}
# Model Creation and Training
model_knn = KNeighborsClassifier(n_neighbors=5)
models_knn = GridSearchCV(estimator=model_knn, param_grid=parameters, cv=4)
models_knn.fit(x_train, y_train)
best_parameters = models_knn.best_params_
print("Best Hyperparameters:", best_parameters)
print()
# Predictions on train data
best_model_knn = models_knn.best_estimator_
y_pred_knn = best_model_knn.predict(x_train)
# Predictions on test data
y_pred_knn_new = best_model_knn.predict(x_test)
checking_overfitting_undefitting(y_train, y_pred_knn, y_test, y_pred_knn_new)

# Evaluation Metrics Calculation
print("Testing Performance")
accuracy_knn, precision_knn, recall_knn, f1_knn = calculate_classification_metrics(y_test, y_pred_knn_new, "KNN")

"""* #### Non-Parametric Algorithm 3: Random Forest Classifier"""

# Hyperparameter tuning
parameters = {'max_depth': [None, 5],
            'class_weight': [None, 'balanced'],
            'min_samples_split': [None, 2, 5]}
# Model Creation and Training
model_rf = RandomForestClassifier()
models_rf = GridSearchCV(estimator=model_rf, param_grid=parameters, cv=4)
models_rf.fit(x_train, y_train)
best_parameters = models_rf.best_params_
print("Best Hyperparameters:", best_parameters)
print()
# Predictions on train data
best_model_rf = models_rf.best_estimator_
y_pred_rf = best_model_rf.predict(x_train)
# Predictions on test data
y_pred_rf_new = best_model_rf.predict(x_test)
checking_overfitting_undefitting(y_train, y_pred_rf, y_test, y_pred_rf_new)

# Evaluation Metrics Calculation
print("Testing Performance")
accuracy_rf, precision_rf, recall_rf, f1_rf = calculate_classification_metrics(y_test, y_pred_rf_new, "Random Forest")

"""* #### Non-Parametric Algorithm 4: Extra Trees Classifier"""

# Hyperparameter tuning
parameters = {'max_depth': [None, 5],
            'class_weight': [None, 'balanced'],
            'min_samples_split': [None, 2, 5],
            'criterion':['gini','log_loss','entropy']}
# Model Creation and Training
model_et = ExtraTreesClassifier()
models_et = GridSearchCV(estimator=model_et, param_grid=parameters, cv=4)
models_et.fit(x_train, y_train)
best_parameters = models_et.best_params_
print("Best Hyperparameters:", best_parameters)
print()
# Predictions on train data
best_model_et = models_et.best_estimator_
y_pred_et = best_model_et.predict(x_train)
# Predictions on test data
y_pred_et_new = best_model_et.predict(x_test)
checking_overfitting_undefitting(y_train, y_pred_et, y_test, y_pred_et_new)

# Evaluation Metrics Calculation
print("Testing Performance")
accuracy_et, precision_et, recall_et, f1_et = calculate_classification_metrics(y_test, y_pred_et_new, "Extra Trees")

"""* #### Non-Parametric Algorithm 5: Gradient Boosting Classifier"""

# Hyperparameter tuning
parameters = {
    'n_estimators': [50, 100, 150],
    'learning_rate': [0.1, 0.01, 0.001],
    'max_depth': [3, 5, 7]
}
# Model creation and training
model_gb = GradientBoostingClassifier()
models_gb = GridSearchCV(estimator=model_gb, param_grid=parameters, cv=4)
models_gb.fit(x_train, y_train)
best_parameters = models_gb.best_params_
print("Best Hyperparameters:", best_parameters)
print()
# Predictions on the training data
best_model_gb = models_gb.best_estimator_
y_pred_gb = best_model_gb.predict(x_train)
# Predictions on the test data
y_pred_gb_new= best_model_gb.predict(x_test)
checking_overfitting_undefitting(y_train, y_pred_gb, y_test, y_pred_gb_new)

# Evaluation Metrics Calculation
print("Testing Performance")
accuracy_gb, precision_gb, recall_gb, f1_gb = calculate_classification_metrics(y_test, y_pred_gb_new, "Gradient Boosting Classifier")

"""* #### Non-Parametric Algorithm 6: Bagging Classifier"""

# Hyperparameter tuning
parameters = {
    'n_estimators': [50, 100, 200],
    'max_samples': [0.5, 0.7, 0.9],
    'max_features': [0.5, 0.7, 0.9]
}
# Model creation and training
model_bagging = BaggingClassifier()
models_bagging = GridSearchCV(estimator=model_bagging, param_grid=parameters, cv=4)
models_bagging.fit(x_train, y_train)
best_parameters = models_bagging.best_params_
print("Best Hyperparameters:", best_parameters)
print()
# Predictions for train
best_model_bagging = models_bagging.best_estimator_
y_pred_bagging = best_model_bagging.predict(x_train)
# Predictions for test
y_pred_bagging_new= best_model_bagging.predict(x_test)
checking_overfitting_undefitting(y_train, y_pred_bagging, y_test, y_pred_bagging_new)

# Evaluation Metrics Calculation
print("Testing Performance")
accuracy_bc, precision_bc, recall_bc, f1_bc = calculate_classification_metrics(y_test, y_pred_bagging_new, "Bagging Classifier")

# Results
print("Testing Performances for Machine Learning Algorithms")
result = pd.DataFrame({"Algorithms":['Logistic Regression', "Gaussian Naive Bayes", "SVC", "SGD Classifier", "Decision Tree", "KNN","Random Forest", "Extra Trees Classifier", "Bagging Classifier","Gradient Boosting Classifier"],
                       "Accuracy":[accuracy_lr, accuracy_nb, accuracy_svc, accuracy_sgd, accuracy_dt, accuracy_knn, accuracy_rf, accuracy_et, accuracy_bc, accuracy_gb],
                       "Precision":[precision_lr, precision_nb, precision_svc, precision_sgd, precision_dt, precision_knn, precision_rf, precision_et, precision_bc, precision_gb],
                       "Recall":[recall_lr, recall_nb, recall_svc, recall_sgd, recall_dt, recall_knn, recall_rf, recall_et, recall_bc, recall_gb],
                       "F1 Score":[f1_lr, f1_nb, f1_svc, f1_sgd, f1_dt, f1_knn, f1_rf, f1_et, f1_bc, f1_gb]}).set_index('Algorithms')
result

# Saving sklearn machine learning models
models = [best_model_dt, best_model_lr, best_model_knn, best_model_et, best_model_nb, best_model_rf, best_model_sgd, best_model_svc, best_model_gb, best_model_bagging]
names = ["dt","lr","knn","et","nb","rf","sgd","svc","gb","bg"]
for i in range(len(models)):
    joblib.dump(models[i],names[i]+".pkl")

"""---"""
