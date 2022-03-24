def split_train_test(df, test_size=0.2):
    """
    Split dataframe into train and test dataframes.
    """
    len_dataset = len(df)
    len_test = int(len_dataset * test_size)
    len_train = len_dataset - len_test
    df_train = df.iloc[:len_train]
    df_test = df.iloc[len_train:]
    X_train = df_train.drop(columns=['Close'])
    y_train = df_train['Close']
    x_test = df_test.drop(columns=['Close'])
    y_test = df_test['Close']
    return (X_train, x_test, y_train, y_test)