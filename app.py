from flask import Flask
from flask import request
from flask import render_template
import os
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import mean_squared_error,confusion_matrix
import math
import argparse

app = Flask('Wind_Farm_Site_Analytics')
@app.route('/')
def show_predict_stock_form():
    return render_template('predictorform.html')
@app.route('/results', methods=['POST'])
def results():
    form = request.form
    if request.method == 'POST':
        x=request.form['x']
        y=request.form['y']
        x = float(x)
        y = float(y)
        data_path="E:/UCC/Wind Farm Site Analytics/shifa/shifa/data"
        loc=""
        station=""
        #Code to find County
        #Max distance for Cork
        center_cork=[51.994200, -8.728500]
        left_cork=[51.716080, -9.959818]
        max_distance_cork= ((center_cork[0]-left_cork[0])*(center_cork[0]-left_cork[0])) + ((center_cork[1]-left_cork[1])*(center_cork[1]-left_cork[1]))
        max_distance_cork= math.sqrt(max_distance_cork)

        #Max distance for dublin
        center_d=[53.330200, -6.310600]
        left_d=[53.292643, -6.445484]
        max_distance_d= ((center_d[0]-left_d[0])*(center_d[0]-left_d[0])) + ((center_d[1]-left_d[1])*(center_d[1]-left_d[1]))
        max_distance_d= math.sqrt(max_distance_d)
        #Clasification of cordinate

        station_distance= ((center_cork[0]-x)*(center_cork[0]-x)) + ((center_cork[1]-y)*(center_cork[1]-y))
        station_distance = math.sqrt(station_distance)
        if station_distance < max_distance_cork:
            loc="The point is in County Cork"
        else:
            station_distance= ((center_d[0]-x)*(center_d[0]-x)) + ((center_d[1]-y)*(center_d[1]-y))
            station_distance = math.sqrt(station_distance)
            if station_distance < max_distance_d:
                loc="The point is in County Dublin"
            else:
                loc="Sorry!Currently not available"
                station="NULL"
                output="NULL"
                return render_template('resultsform.html', location=loc,   station=station,output=output)

        #Code to find the nearest neighbour
        dir_data = os.listdir(data_path)
        length = len(dir_data)
        distance={}
        for i in range (len(dir_data)):
            path = data_path + "/"+ dir_data[i]
            df= pd.read_csv(path)
            latitude = df["latitude"][0]
            longitude = df["longitude"][0]
            dist = ((x-latitude)*(x-latitude)) + ((y-longitude)*(y-longitude))
            distance[math.sqrt(dist)] = i

        min_value = min(distance.keys())
        #print(distance[min_value])
        station="The point is closer to "+ dir_data[distance[min_value]]

        #Load the test file
        df= pd.read_csv(data_path + "/" + dir_data[distance[min_value]])
        x_test =  df.drop(["soil","pe","evap","smd_md","smd_wd","smd_pd","igmin","gmin"],axis=1)

        #x_test = x_test.drop(["date"],axis=1)
        x_test = x_test.replace(r'^\s*$', np.nan, regex=True).dropna()
        y_test = x_test["wdsp"]
        x_test = x_test.drop(["wdsp"],axis =1)
        with open("final_model.pkl", "rb") as fin:
            clf= pickle.load(fin)

        y_pred = clf.predict(x_test.drop(["date"],axis=1))

        y_test_labels = pd.cut(y_test,bins=[0,6,10,17,25,100],labels=['Calm','Gentle Breeze','Moderate','Strong','Very Strong'])
        #Converting array into Series
        y_pred_labels = pd.cut(pd.Series(y_pred),bins=[0,6,10,17,25,100],labels=['Calm','Gentle Breeze','Moderate','Strong','Very Strong'])

        output=pd.DataFrame(columns=['Date','Wind Speed','Class'])
        for date,pred,labels in zip (x_test["date"][-10:],y_pred[-10:],y_pred_labels[-10:]):
            #print(date,round(pred,3),labels)
            output=output.append({'Date':date,'Wind Speed':round(pred,3),'Class':labels},ignore_index=True)


        return render_template('resultsform.html', location=loc,station=station,tables=[output.to_html(classes='data')], titles=output.columns.values)

if __name__ == '__main__':
    app.run("localhost", "9999", debug=False)
