from flask import Flask, render_template, redirect, request
import pandas as pd
import joblib

pred_model = joblib.load('knn_1hr_models.pk1')
features = joblib.load('df_cols.pkl')

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('predictor.html')

@app.route('/about.html')
def about():
    return render_template('about.html')

@app.route('/credits.html')
def credits():
    return render_template('credits.html')

@app.route('/predictor.html')
def predictor_index():
    return redirect('/')

@app.route('/predict', methods=['POST'])
def predict():
    data = [float(x) for x in request.form.values()]
    df = pd.DataFrame([data], columns = features)

    skin_temp_nxt = pred_model[0].predict(df)
    skin_temps = [skin_temp_nxt[0], ]
    for i in range(1, 5):
        df_next = df.copy(deep = True)
        df_next.loc[:, 'SKIN_TEMP_AVG'] = skin_temp_nxt[0]
        skin_temp_nxt[0] = pred_model[i].predict(df_next)
        skin_temps.append(skin_temp_nxt[0])

    return render_template('results.html', skin_temp_curr = skin_temps[0], skin_temp_1 = skin_temps[1], skin_temp_2 = skin_temps[2], skin_temp_3 = skin_temps[3], skin_temp_4 = skin_temps[4])

if __name__ == '__main__':
    app.run()