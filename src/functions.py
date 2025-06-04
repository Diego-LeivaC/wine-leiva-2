# We are going to start preparing the data for the model

# 1st function: Split the features variables (X) with the target variable (y) and scale

from sklearn.preprocessing import StandardScaler

def preprocess_data(df):
    X = df.drop(columns=['quality', 'target'])
    y = df['target'].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y

# 2nd function: Data split in training and test set

from sklearn.model_selection import train_test_split

def split_data(X, y, test_size=0.2, random_state=27):
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)