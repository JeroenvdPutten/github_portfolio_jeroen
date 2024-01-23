def preprocess(csv_file):
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.impute import KNNImputer
    from sklearn.impute import KNNImputer

    df = pd.read_csv(csv_file)

    # cabin has quite some unique values (147), so thats difficult to analyze, but I want to try and aggregate them
    # on area (A, B, C, D, etc) and give missings their own category
    df['Cabin_area'] = df['Cabin'].str.extract(r'([A-Za-z])')
    df['Cabin_area'] = df['Cabin_area'].fillna('MISS')
    df = df.drop(['Cabin'], axis=1)

    # impute missings
    # fill Embarked with mode
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

    # fill Fare with median
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())

    # Initialize the KNNImputer with the desired number of neighbors
    # You can adjust the value of n_neighbors as needed
    imputer = KNNImputer(n_neighbors=5)
    # Perform KNN imputation on the 'Age' column
    df['Age'] = imputer.fit_transform(df[['Age']])

    # Outliers
    df.loc[df['Fare'] > 512, 'Fare'] = 263
    df['Fare'].sort_values(ascending=False).head(5)

    # We can group the Name variable based on Title
    df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    df['Title'] = df['Title'].replace(['Dr', 'Rev', 'Mlle', 'Major', 'Col', 'Countess',
                                       'Capt', 'Ms', 'Sir', 'Lady', 'Mme', 'Don', 'Jonkheer'], 'Other')

    # We can group the 'Names' variable based on surname and count the number of relatives on board.
    df['Last_Name'] = df['Name'].str.split(',').str[0].str.strip()
    df['Relatives'] = df.groupby('Last_Name')['Last_Name'].transform('count')
    df = df.drop(['Last_Name'], axis=1)

    df['Relatives'] = df['Relatives'].replace(
        [1, 2, [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]], ['Single', 'Duo', 'More'])

    # onehot encode
    encoder = OneHotEncoder(sparse=False, drop='first')
    categorical_cols = ['Pclass', 'Sex', 'Embarked',
                        'Cabin_area', 'Title', 'Relatives']
    encoded_cols = encoder.fit_transform(df[categorical_cols])

    # Create a new DataFrame with the encoded columns
    df_encoded = pd.DataFrame(
        encoded_cols, columns=encoder.get_feature_names_out(categorical_cols))

    # Concatenate the encoded DataFrame with the original DataFrame
    if 'Cabin_area_T' in df.columns:
        # Drop 'Cabin_area_T' if it exists
        df = pd.concat([df.drop(categorical_cols, axis=1),
                       df_encoded.drop(['Cabin_area_T'], axis=1)], axis=1)
    else:
        # Continue without dropping 'Cabin_area_T'
        df = pd.concat([df.drop(categorical_cols, axis=1), df_encoded], axis=1)

    df = df.drop(['Name', 'Ticket', 'Embarked_S',
                  'Cabin_area_B', 'Cabin_area_C', 'Cabin_area_D', 'Cabin_area_E',
                  'Cabin_area_F', 'Cabin_area_G', 'Title_Other', 'Embarked_Q', 'Title_Mrs'], axis=1)
    return df


def model_train(df):
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score, GridSearchCV

    # create train test split
    features = list(df.drop(['PassengerId', 'Survived'], axis=1))
    X_train, X_val, y_train, y_val = train_test_split(
        df[features], df['Survived'], train_size=0.7, random_state=12)
    print(f"X_train.shape = {X_train.shape}, X_val.shape = {X_val.shape}")
    print(f"y_train.shape = {y_train.shape}, y_val.shape = {y_val.shape}")

    # fit model
    rf = RandomForestClassifier(random_state=9)

    # param grid
    grid_param = {
        'n_estimators': [100, 200, 300],
        'criterion': ['gini', 'entropy'],
        'min_samples_split': [2, 10, 20],
        'min_samples_leaf': [1, 5],
        'bootstrap': [True, False],
    }
    gd_sr = GridSearchCV(estimator=rf,
                         param_grid=grid_param,
                         scoring='accuracy',
                         cv=5,
                         n_jobs=-1)
    gd_sr.fit(X_train, y_train)
    best_parameters = gd_sr.best_params_
    print(best_parameters)

    # train model on optimized parameters
    model = RandomForestClassifier(bootstrap=False, criterion='entropy', min_samples_leaf=1,
                                   min_samples_split=10, n_estimators=200, random_state=9)
    model.fit(X_train, y_train)  # Fit the model with the training data

    return model


def model_test(df, model):
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier

    features = list(df.drop(['PassengerId'], axis=1))
    X_test = df[features]

    y_pred = model.predict(X_test)

    passenger_ids = df['PassengerId'].copy()

    prediction = pd.DataFrame({
        'PassengerId': passenger_ids,
        'Survived': y_pred
    })

    return prediction


def random_forrest(train_file, test_file):
    train = preprocess(train_file)  # preprocess trainset
    test = preprocess(test_file)  # preprocess testset
    model = model_train(train)  # train model
    pred = model_test(test, model)  # apply model to testset

    return pred


train_file = 'C:/Users/jeroe/Documents/Github/Project_Titanic/train.csv'
test_file = 'C:/Users/jeroe/Documents/Github/Project_Titanic/test.csv'
save_location = 'C:/Users/jeroe/Documents/Github/Project_Titanic/rf_prediction_final.csv'

pred = random_forrest(train_file, test_file)
pred.to_csv(save_location, index=False)
