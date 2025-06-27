import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

class Encoder:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.gv_dict = {}
        self.feature = None

    def fit(self, train_df, feature, target='Class'):
        self.feature = feature
        value_counts = train_df[feature].value_counts()
        gv_dict = {}
        for i, denominator in value_counts.items():
            vec = []
            for k in range(1, 10):  # class values from 1 to 9
                cls_cnt = train_df[(train_df[target] == k) & (train_df[feature] == i)]
                vec.append((cls_cnt.shape[0] + self.alpha * 10) / (denominator + 90 * self.alpha))
            gv_dict[i] = vec
        self.gv_dict = gv_dict

    def transform(self, df):
        if not self.feature or not self.gv_dict:
            raise ValueError("You must call fit() before transform().")

        gv_fea = []
        for _, row in df.iterrows():
            value = row[self.feature]
            if value in self.gv_dict:
                gv_fea.append(self.gv_dict[value])
            else:
                gv_fea.append([1 / 9] * 9)  # default for unseen
        return np.array(gv_fea)

    def get_response_coding(self, train_df, feature, target='Class'):
        self.fit(train_df, feature, target)
        return self.transform(train_df)
    
    def get_onehotCoding(self, train_df,test_df,cv_df, feature): 
        gene_vectorizer = CountVectorizer()
        train_gene_feature_onehotCoding = gene_vectorizer.fit_transform(train_df[feature])
        test_gene_feature_onehotCoding = gene_vectorizer.transform(test_df[feature])
        cv_gene_feature_onehotCoding   = gene_vectorizer.transform(cv_df[feature])

        return train_gene_feature_onehotCoding, test_gene_feature_onehotCoding, cv_gene_feature_onehotCoding


# from encoding import Encoder
# # Create and use encoder
# rc = Encoder().get_onehotCoding(train_df, feature='Gene')

