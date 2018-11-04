# -*- coding: utf-8 -*-

import pandas as pd
import pyodbc
import sqlalchemy
import urllib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import SGDRegressor

def Stacking_Count (SumRange):

    # Устанавливаем соединение с базой
    connection = pyodbc.connect('Driver={SQL Server};'
                                'Server=*******;'
                                'Database=BI_Analytics;'
                                'Trusted_Connection=yes')


    forecast_sql = """
            SELECT [BG_Year]
              ,[BG_Month]
              ,[RegionId]
              ,[Collaborative]
              ,[OtraslId]
              ,[SumRangeCode3]
              ,[BG_Count_Avg]
              ,[LotCount_1]
              ,[LotCount_2]

          FROM [BI_Analytics].[forecast].[BG_PredictSetOnly_Final]
          WHERE [SumRangeCode3] = """+str(SumRange)

    forecast_data = pd.read_sql_query(forecast_sql, connection)

    # Формирование обучающей и тестовой выборки

    sql = """
            SELECT [BG_Year]
                  ,[BG_Month]
                  ,[RegionId]
                  ,[Collaborative]
                  ,[OtraslId]
                  ,[SumRangeCode3]
                  ,[BG_Count_Avg]
                  ,[BG_Count]
                  ,[LotCount_1]
                  ,[LotCount_2]

              FROM [BI_Analytics].[forecast].[BG_train]
              WHERE [SumRangeCode3] = """ +str(SumRange)


    data = pd.read_sql_query(sql, connection)
    X = data.drop(['BG_Count'], axis=1)
    Y = data['BG_Count'].values
    train_data = X
    train_labels = Y
    # Создаем матрицу категориальных признаков
    train_data_tran = pd.get_dummies(train_data, columns=['RegionId','OtraslId', 'BG_Month', 'BG_Year'])
    forecast_data_tran = pd.get_dummies(forecast_data, columns=['RegionId','OtraslId', 'BG_Month', 'BG_Year'])

    print train_data_tran.columns.values
    print '-----------------------------------------------------------------------------------------'
    print forecast_data_tran.columns.values



    # Обучаем регрессию
    parameters_reg = {
                    "loss": 'huber',  # Функция потерь
                    "n_iter": 100,
                    "penalty": 'l2',  # Тип регуляризации (Ридж, Лассо)
                    "verbose": 2,
                    "random_state": 0
                    }

    print 'START REGRESSION!'
    regressor = SGDRegressor(**parameters_reg)
    regressor.fit(train_data_tran, train_labels)
    # Добавляем результат в обучающую выборку
    train_regressor = regressor.predict(train_data_tran)
    train_data_tran.insert(1, 'Forecast_regressor', train_regressor)
    # Добавляем результаты в прогноз
    forecasting_regressor = regressor.predict(forecast_data_tran)
    forecast_data_tran.insert(1, 'Forecast_regressor', forecasting_regressor)
    print train_data_tran
    print 'END REGRESSION'

    # Обучаем метод ближайших соседей
    parameters_knn = {
                    "n_neighbors": 5,
                    "algorithm": 'auto',
                    }

    print 'START KNN!'
    # Добавляем результаты прогноза в модель
    knn = KNeighborsRegressor(**parameters_knn)
    knn.fit(train_data_tran, train_labels)
    # Добавляем результат в обучающую выборку
    train_knn = np.around(knn.predict(train_data_tran), decimals=0)
    train_data_tran.insert(1, 'Forecast_KNN', train_knn)
    # Добавляем результаты в прогноз
    forecasting_knn = np.around(knn.predict(forecast_data_tran), decimals=0)
    forecast_data_tran.insert(1, 'Forecast_KNN', forecasting_knn)
    print train_data_tran
    print 'END KNN!'

    # Обучаем градиентный бустинг
    parameters_boosting = {
                        "n_estimators": 100,
                        "loss": 'ls',
                        "learning_rate": 0.005,
                        "max_depth": 100,
                        "verbose": 2,
                        "random_state": 0
                        }

    print 'START BOOSTING!'
    # Добавляем результаты прогноза в модель
    boosting = GradientBoostingRegressor(**parameters_boosting)
    boosting.fit(train_data_tran, train_labels)
    # Добавляем результат в обучающую выборку
    train_boosting = np.around(boosting.predict(train_data_tran), decimals=0)
    train_data_tran.insert(1, 'Forecast_Boosting', train_boosting)
    # Добавляем результаты в прогноз
    forecasting_boosting = np.around(boosting.predict(forecast_data_tran), decimals=0)
    forecast_data_tran.insert(1, 'Forecast_Boosting', forecasting_boosting)
    print train_data_tran
    print 'END BOOSTING!'

    # Обучаем случайный лес
    parameters_forest = {
                        "n_estimators": 800,
                        "max_depth": None,
                        "n_jobs": -1,
                        "verbose": 2,
                        "random_state": 0
                        }

    print 'START FOREST!'
    forest = RandomForestRegressor(**parameters_forest)
    forest.fit(train_data_tran, train_labels)
    # Добавляем результат в обучающую выборку
    train_forest = forest.predict(train_data_tran)
    train_data_tran.insert(1, 'Forecast_forest', train_forest)
    # Добавляем результаты в прогноз
    #forecasting_forest = forest.predict(forecast_data_tran)
    #forecast_data_tran.insert(1, 'Forecast_forest', forecasting_forest)
    print train_data_tran
    print 'END FOREST!'


    # МЕТАМОДЕЛЬ
    forecasting = np.around(forest.predict(forecast_data_tran), decimals = 0)


    forecast_data.insert(1, 'BG_Count', forecasting)
    print forecast_data
    params = urllib.quote_plus("DRIVER={SQL Server};SERVER=bi2;DATABASE=BI_Analytics; Trusted_Connection=yes")
    conn = sqlalchemy.create_engine("mssql+pyodbc:///?odbc_connect=%s"%params)
    forecast_data.to_sql("Forecasting_BG_Count", conn, if_exists='append', index=False, schema='forecast')

    print 'FORECAST COMLETE!'

    connection.close()


def main():
    connection_truncate = pyodbc.connect('Driver={SQL Server};'
                                'Server=BI2;'
                                'Database=BI_Analytics;'
                                'Trusted_Connection=yes')

    cursor = connection_truncate.cursor()
    SQLCommand = ("truncate table [forecast].[Forecasting_BG_Count]")
    cursor.execute(SQLCommand)
    connection_truncate.commit()

    SR = [7,6,5,4,3,2,1]

    for s in SR:
        print '------------------------------------------------'
        print 'SUM RANGE {f} STARTED!'.format(f=str(s))
        Stacking_Count(s)
        print '------------------------------------------------'
        print 'SUM RANGE {f} COMPLETE!'.format(f=str(s))

    execute_sql = ('EXEC [forecast].[BG_Count_Final]')
    cursor.execute(execute_sql)


if __name__ == '__main__':
    main()

