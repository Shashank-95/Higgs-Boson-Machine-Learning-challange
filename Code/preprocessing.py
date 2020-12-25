import utils
def PreProcessing(trainset, valtest, y_train, y_valtest):
    #Separate PRI_jet_num and DER_mass_MMC from the data to preprocess separately
    jet_num_train =  trainset['PRI_jet_num'].values
    jet_num_val = valtest['PRI_jet_num'].values

    der_mass_train = trainset['DER_mass_MMC'].values
    der_mass_val = valtest['DER_mass_MMC'].values

    X_train_dropped = trainset.drop(['PRI_jet_num'], axis=1)
    X_train_dropped = X_train_dropped.drop(['DER_mass_MMC'], axis=1)
    X_val_dropped = valtest.drop(['PRI_jet_num'], axis=1)
    X_val_dropped = X_val_dropped.drop(['DER_mass_MMC'], axis=1)

    #Step1: Standardize the data by ignoring NaN values to 28 features and assign 0 to all NaN values.
    std = utils.StandardScaler()
    train_preprocessed = std.fit_transform(X_train_dropped)
    val_preprocessed = std.transform(X_val_dropped)
    train_preprocessed_step1 = utils.pd.DataFrame(train_preprocessed, columns = X_train_dropped.columns.values).replace(utils.np.nan, 0)
    val_preprocessed_step1 = utils.pd.DataFrame(val_preprocessed, columns = X_val_dropped.columns.values).replace(utils.np.nan, 0)


    #Step2: Use one hot encoding for jet_num variable
    one_hot_encoder = utils.OneHotEncoder()
    train_preprocessed_step2 = one_hot_encoder.fit_transform(jet_num_train.reshape(-1,1))
    val_preprocessed_step2 = one_hot_encoder.transform(jet_num_val.reshape(-1,1))
    train_preprocessed_step2 = utils.pd.DataFrame(train_preprocessed_step2.toarray(), columns = ['PRI_jet_num_0', 'PRI_jet_num_1', 'PRI_jet_num_2', 'PRI_jet_num_3' ])
    val_preprocessed_step2 = utils.pd.DataFrame(val_preprocessed_step2.toarray(), columns = ['PRI_jet_num_0', 'PRI_jet_num_1', 'PRI_jet_num_2', 'PRI_jet_num_3' ])


    #Step3: Mean imputation on DER_mas_MMC
    imp = utils.SimpleImputer(missing_values=utils.np.nan, strategy='mean')
    mass_imputation_train = imp.fit_transform(der_mass_train.reshape(-1,1))
    mass_imputation_val = imp.transform(der_mass_val.reshape(-1,1))
    mass_preprocessed_train = std.fit_transform(mass_imputation_train)
    mass_preprocessed_val = std.transform(mass_imputation_val)
    mass_preprocessed_train = utils.pd.DataFrame(mass_preprocessed_train, columns = ['DER_mass_MMC'])
    mass_preprocessed_val = utils.pd.DataFrame(mass_preprocessed_val, columns = ['DER_mass_MMC'])

    #Label encoder
    labelenc = utils.LabelEncoder()
    y_train_enc = labelenc.fit_transform(y_train)
    y_val_enc = labelenc.transform(y_valtest)


    #Concatenate all the features
    train_preprocessed_final = utils.pd.concat([train_preprocessed_step1, train_preprocessed_step2, mass_preprocessed_train], axis=1, sort=False)
    val_preprocessed_final = utils.pd.concat([val_preprocessed_step1, val_preprocessed_step2, mass_preprocessed_val], axis=1, sort=False)

    return train_preprocessed_final, val_preprocessed_final, y_train_enc, y_val_enc
