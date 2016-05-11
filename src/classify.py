import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier

df = pd.DataFrame.from_csv("../data/train_risk.csv", index_col=False)
test = pd.DataFrame.from_csv("../data/test_risk.csv", index_col=False)
X, y = df[df.columns[:-1]], df[df.columns[-1]]


def encode_onehot(x_data, column_name, digitize=False):
    """ Encode a catagorical column from a data frame into a
    data frame of one hot features
    """
    data = x_data[[column_name]]

    if digitize:
        data = np.digitize(data, np.arange(data.min(), data.max(), 10))

    enc = preprocessing.OneHotEncoder()
    features = enc.fit_transform(data).toarray()
    names = ['%s_%d' % (column_name, i) for i in enc.active_features_]
    features = pd.DataFrame(features, columns=names, index=x_data.index)
    return features


def create_spiro_features(x_data):
    """ Creare spriometry based features """
    # create new feature FER
    # this is the raito of FEV1 and FVC
    FER = (x_data.PRE5 / x_data.PRE4) * 100
    FER.index = x_data.index

    # create a new feature OBS
    # this is whether the instance has a FER below 70%
    # which implies an obstructive disease.
    OBS = pd.Series(np.zeros(x_data.AGE.shape))
    OBS.index = x_data.index
    OBS.loc[FER < 70] = 1.0

    spiro = pd.concat([FER, OBS], axis=1)
    spiro.columns = ['FER', 'OBS']
    return spiro


def create_poly_features(x_data, names):
    """ Create new features base on Polynomials of the original best two predictors """
    poly = preprocessing.PolynomialFeatures(2, include_bias=False, interaction_only=True)
    poly_features = pd.DataFrame(poly.fit_transform(x_data[names]), index=x_data.index)
    poly_features.columns = ["POLY_%d" % i for i in poly_features.columns]
    return poly_features


def preprocess(x_data, y_data=None):
    # drop zero var PRE32
    Xp = x_data.drop("PRE32", axis=1)

    # remove outliers
    if y_data is not None:
        mask = Xp.PRE5 < 30
        Xp = Xp.loc[mask]
        Yp = y_data.copy()
        Yp = Yp.loc[mask]
    else:
        Yp = None

    # encode catagorical data as one hot vectors
    one_hot_names = ["DGN"]
    encoded = map(lambda name: encode_onehot(Xp, name), one_hot_names)
    # combine into a single data frame
    new_features = pd.concat(encoded, axis=1)

    # drop the catagorical variables that have been encoded
    Xp.drop(["DGN"], inplace=True, axis=1)
    # add new features
    Xp = pd.concat([Xp, new_features], axis=1)

    return Xp, Yp

Xp, Yp = preprocess(X, y)

scaler = preprocessing.StandardScaler()

gbc_params = {
    'min_samples_leaf': 1,
    'min_samples_split': 7,
    'max_depth': 9,
    'max_features': 11,
    'subsample': 0.8,
    'n_estimators': 1000,
    'learning_rate': 0.01
}
gbc = GradientBoostingClassifier(**gbc_params)
gbc_pipe = Pipeline([('scaler', scaler), ('GradientBoostingClassifer', gbc)])

model = {'name': 'GradientBoost', 'model': gbc_pipe}

# Create training features
spiro_features = create_spiro_features(Xp)
poly_features = create_poly_features(Xp, ['PRE4', 'PRE5'])
Xp_all = pd.concat([Xp, poly_features, spiro_features], axis=1)
Xp_all.drop(['DGN_1', 'DGN_8'], axis=1, inplace=True)

# Create testing features
Xtest, _ = preprocess(test, y_data=None)
Xtest = Xtest.drop('test_id', axis=1)

test_spiro_features = create_spiro_features(Xtest)
test_poly_features = create_poly_features(Xtest, ['PRE4', 'PRE5'])
Xtest = pd.concat([Xtest, test_spiro_features, test_poly_features], axis=1)

# Build model
gbc_final = model['model']
gbc_final.fit(Xp_all, Yp)
predicted_prob = pd.Series(gbc_final.predict_proba(Xtest)[:, 0])

# Format output
predicted_label = predicted_prob.copy()
predicted_label[predicted_label >= 0.5] = 1
predicted_label[predicted_label < 0.5] = 0
predicted_label = predicted_label.astype(int)

final_submission = pd.concat([test.test_id, predicted_label, predicted_prob], axis=1)
final_submission.columns = ['test_id', 'predicted_label', 'predicted_output']
final_submission.to_csv('../data/submission.csv', index=False)
