from data_prep import *

def create_iterative_german_bias (X, y,  unprivileged_class_name1="Age_group", unprivileged_class_name2= "Attribute9", unprivileged_class_value1=0, unprivileged_class_value2=0,
                            p_range1=[0.2,0.5,0.8], p_range2=[0.2,0.5,0.8], verbose=True, dataset_name =  'German_credit_biased'):
    
    for p1 in p_range1:
        for p2 in p_range2:
            Xc, yc = add_bias(X=X, unprivileged_class_name=unprivileged_class_name1, unprivileged_class_value= unprivileged_class_value1, y=y, p=p1,
                            verbose=True)  # Age bias
            Xc, yc = add_bias(X=Xc, unprivileged_class_name = unprivileged_class_name2, unprivileged_class_value = unprivileged_class_value2, y=yc, p=p2,
                            verbose=True)  # Gender bias

            num_features = ["Attribute2", "Attribute5", "Attribute8", "Attribute11", "Attribute13", "Attribute16",
                            "Attribute18"]
            cat_features = [col_name for col_name in X.columns if col_name not in num_features]
            Xc = pd.get_dummies(Xc, columns=cat_features, drop_first=True)

            X_train, X_test, y_train, y_test = train_test_split(Xc, yc, test_size=0.4, random_state=123)

            X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5)  ##this make 0.2 for both val and test

            ## Save sensitive attributes

            gender_train = X_train["Attribute9_1"]
            age_train = X_train["Age_group_1"]

            gender_test = X_test["Attribute9_1"]
            age_test = X_test["Age_group_1"]

            gender_val = X_val["Attribute9_1"]
            age_val = X_val["Age_group_1"]

            X_train_with_A = X_train.copy()  # X with sensitive_attributes
            X_train_with_A.drop("Age_group_1", axis=1, inplace=True)
            X_train_without_A = X_train.drop(["Age_group_1", "Attribute9_1"], axis=1)

            X_test_with_A = X_test.copy()  # X with sensitive_attributes
            X_test_with_A.drop("Age_group_1", axis=1, inplace=True)
            X_test_without_A = X_test.drop(["Age_group_1", "Attribute9_1"], axis=1)

            X_val_with_A = X_val.copy()  # X with sensitive_attributes
            X_val_with_A.drop("Age_group_1", axis=1, inplace=True)
            X_val_without_A = X_val.drop(["Age_group_1", "Attribute9_1"], axis=1)

            store_data(dataset_name=dataset_name, X_train_with_A=X_train_with_A,
                       X_train_without_A=X_train_without_A, X_val_with_A=X_val_with_A,
                       X_val_without_A=X_val_without_A, X_test_with_A=X_test_with_A, X_test_without_A=X_test_without_A,
                       y_train=y_train,
                       y_val=y_val, y_test=y_test, age=[age_train, age_val, age_test],
                       gender=[gender_train, gender_val, gender_test], education=None,sufix_name=f"_{p1}_{p2}")

            