# Reference
# http://zacstewart.com/2014/08/05/pipelines-of-featureunions-of-pipelines.html

import numpy as np
import pandas as pd
from functools import reduce

from sklearn.base import TransformerMixin
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import LabelEncoder, StandardScaler


class DFFeatureUnion(TransformerMixin):
    # FeatureUnion but for pandas DataFrames

    def __init__(self, transformer_list):
        self.transformer_list = transformer_list
        self.Xunion = None

    def fit(self, X, y=None):
        for (name, t) in self.transformer_list:
            t.fit(X, y)
        return self

    def transform(self, X):
        # assumes X is a DataFrame
        Xts = [t.transform(X) for _, t in self.transformer_list]
        self.Xunion = reduce(lambda X1, X2: pd.merge(X1, X2, left_index=True, right_index=True), Xts)
        return self.Xunion

    def get_feature_names(self):
        return self.Xunion.columns.tolist()


class ColumnExtractor(TransformerMixin):

    def __init__(self, cols):
        self.cols = cols

    def fit(self, X, y=None):
        # stateless transformer
        return self

    def transform(self, X):
        # assumes X is a DataFrame
        Xcols = X[self.cols]
        return Xcols


# dates
class DateFormatter(TransformerMixin):

    def fit(self, X, y=None):
        # stateless transformer
        return self

    def transform(self, X):
        # assumes X is a DataFrame
        Xdate = X.apply(pd.to_datetime)
        return Xdate


class DateToMonth(TransformerMixin):

    def fit(self, X, y=None):
        # stateless transformer
        return self

    def transform(self, X):
        # assumes X is a DataFrame
        to_Month = X.columns[0]
        X['sign_month'] = X[to_Month].dt.month
        return X


class DateDiffer(TransformerMixin):

    def fit(self, X, y=None):
        # stateless transformer
        return self

    def transform(self, X):
        # assumes X is a DataFrame
        beg_cols = X.columns[0]
        end_cols = X.columns[1]
        X['interval'] = map(lambda x: x.total_seconds(), X[end_cols] - X[beg_cols])
        return X


class DeleteCol(TransformerMixin):

    def __init__(self, cols):
        self.cols = cols

    def fit(self, X, y=None):
        # stateless transformer
        return self

    def transform(self, X):
        # assumes X is a DataFrame
        X = X.drop(self.cols, axis=1)
        return X


# cat_ohe
class DummyTransformer(TransformerMixin):

    def __init__(self):
        self.dv = None

    def fit(self, X, y=None):
        # assumes all columns of X are strings
        Xdict = X.to_dict('records')
        self.dv = DictVectorizer(sparse=False)
        self.dv.fit(Xdict)
        return self

    def transform(self, X):
        # assumes X is a DataFrame
        Xdict = X.to_dict('records')
        Xt = self.dv.transform(Xdict)
        cols = self.dv.get_feature_names()
        Xdum = pd.DataFrame(Xt, index=X.index, columns=cols)
        # drop column indicating NaNs
        nan_cols = [c for c in cols if '=' not in c]
        Xdum = Xdum.drop(nan_cols, axis=1)
        return Xdum


# cat_lab
class LabelEncoderExt(object):
    def __init__(self):
        self.label_encoder = LabelEncoder()

    def fit(self, data_list):
        # fit(list)
        self.label_encoder = self.label_encoder.fit(np.hstack(data_list.values.tolist()).tolist()+['Unknown'])
        self.classes_ = self.label_encoder.classes_

        return self

    def transform(self, data_list):
        # new_data_list - list
        new_data_list = np.hstack(data_list.values.tolist()).tolist()
        for unique_item in np.unique(data_list):
            if unique_item not in self.label_encoder.classes_:
                new_data_list = ['Unknown' if x==unique_item else x for x in new_data_list]

        return self.label_encoder.transform(new_data_list)


class LabelTransformer(TransformerMixin):

    def __init__(self):
        self.le = None

    def fit(self, X, y=None):
        # assumes all columns of X are strings
        X = X.fillna('NaN')
        self.le = LabelEncoderExt()
        self.le.fit(X)
        return self

    def transform(self, X):
        # assumes X is a DataFrame
        X = X.fillna('NaN')
        Xt = self.le.transform(X)
        X_df = pd.DataFrame(Xt, index=X.index, columns=X.columns)
        return X_df


# add new feature
class AddGroupByCount(TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        group_target = X.columns[-1]
        per_num = X.groupby(group_target).count().reset_index()
        device_num = per_num.rename(columns={X.columns[0]: X.columns[-1]+'_num'})
        X_all = X.reset_index().merge(device_num, how='left', on=group_target).set_index('index')
        X_one = X_all.iloc[:, -1].to_frame(name=X_all.columns[-1])
        return X_one


class DFStandardScaler(TransformerMixin):
    # StandardScaler but for pandas DataFrames

    def __init__(self):
        self.ss = None
        self.mean_ = None
        self.scale_ = None

    def fit(self, X, y=None):
        self.ss = StandardScaler()
        self.ss.fit(X)
        self.mean_ = pd.Series(self.ss.mean_, index=X.columns)
        self.scale_ = pd.Series(self.ss.scale_, index=X.columns)
        return self

    def transform(self, X):
        # assumes X is a DataFrame
        Xss = self.ss.transform(X)
        Xscaled = pd.DataFrame(Xss, index=X.index, columns=X.columns)
        return Xscaled


class ZeroFillTransformer(TransformerMixin):

    def fit(self, X, y=None):
        # stateless transformer
        return self

    def transform(self, X):
        # assumes X is a DataFrame
        Xz = X.fillna(value=0)
        return Xz


class Log1pTransformer(TransformerMixin):

    def fit(self, X, y=None):
        # stateless transformer
        return self

    def transform(self, X):
        # assumes X is a DataFrame
        Xlog = np.log1p(X)
        return Xlog




