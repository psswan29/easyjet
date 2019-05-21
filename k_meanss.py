import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def data_collection(file_path):
    # read the raw data
    airline_data = pd.read_csv(file_path, encoding="gb18030")
    # descriptive study
    # print("the shape of raw data is", airline_data.shape)
    print("the main info of raw data", airline_data.info())
    print("the description of raw data", airline_data.describe())
    # print(airline_data.head(10))
    return airline_data


def data_preprocessing(airline_data):
    # remove the data whose value is null
    exp1 = airline_data["SUM_YR_1"].notnull()
    exp2 = airline_data["SUM_YR_2"].notnull()
    exp = exp1 & exp2
    airline_data_notnull = airline_data.loc[exp, :]

    # remain the data whose price is non-zero
    # or average discount rate is not zero
    # besides, its total mile distance is greater than zero
    index1 = airline_data_notnull['SUM_YR_1'] != 0
    index2 = airline_data_notnull['SUM_YR_2'] != 0
    index3 = (airline_data_notnull['SEG_KM_SUM'] > 0) & (airline_data_notnull['avg_discount'] != 0)
    airline = airline_data_notnull[(index1 | index2) & index3]
    return airline

def data_analysis(airline):
    # shape the main features
    # L: LOAD_TIME  the ending time of study----FFP_DATE	the time got membership
    # R: LAST_TO_END  the length from the last time to ending study
    # F: FLIGHT_COUNT counting the frequency of taking a flight
    # M: SEG_KM_SUM sum of miles
    # C: avg_discount average discount rate

    # select the features(R: LAST_TO_END, F: FLIGHT_COUNT, M: SEG_KM_SUM, C: avg_discount)
    airline_selection = airline[["FFP_DATE", "LOAD_TIME", "LAST_TO_END", "FLIGHT_COUNT", "SEG_KM_SUM", "avg_discount"]]
    airline_selection.head()

    # shape the features for L
    L = pd.to_datetime(airline_selection["LOAD_TIME"]) - pd.to_datetime(airline_selection["FFP_DATE"])
    # print(L)
    # select date
    L = L.astype("str").str.split().str[0]
    # transform into integer
    L = L.astype("int") / 30

    # merge the features
    airline_features = pd.concat([L, airline_selection.iloc[:, 2:]], axis=1)
    # print('the top 5 of the features of LRFMC：\n',airline_features.head())

    airline_features = airline_features.rename(columns={0: 'L'})
    airline_features.head()
    # airline_features.describe()

    # data standardization
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(airline_features)
    # print(scaled_data.head())

    # save the features
    np.savez('./data/airline_features_scaled.npz', scaled_data)

    # data_modeling
    airline_features_scaled = np.load('./data/airline_features_scaled.npz')['arr_0']
    kmeans_model = KMeans(n_clusters=k, random_state=123)
    fit_kmeans = kmeans_model.fit(airline_features_scaled)

    r1 = pd.Series(kmeans_model.labels_).value_counts()
    print('the number of knernals：\n', r1)
    print(kmeans_model.labels_)
    print(kmeans_model.cluster_centers_)
    print()

def data_visulization():
    pass


def main(file_path, k):
    airline_data = data_collection(file_path)
    airline = data_preprocessing(airline_data)
    data_analysis(airline)


if __name__ == '__main__':
    file_path = "./data/air_data.csv"
    k = 5
    main(file_path, k)