from flask import Flask,request,render_template
from src.pipeline.prediction_pipeline import CustomData,PredictPipeline


application=Flask(__name__)

app=application



@app.route('/')
def home_page():
    return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])

def predict_datapoint():
    if request.method=='GET':
        return render_template('form.html')
    
    else:
        data=CustomData(
            
            Age = float(request.form.get('Age')) if request.form.get('Age') is not None else 0.0,
            Height = float(request.form.get('Height')),
            Weight = float(request.form.get('Weight')),
            FCVC = float(request.form.get('FCVC')),
            NCP = float(request.form.get('NCP')),
            CH2O = float(request.form.get('CH2O')),
            FAF = float(request.form.get('FAF')),
            TUE = float(request.form.get('TUE')),
            Gender = request.form.get('Gender'),
            family_history_with_overweight= request.form.get('family_history_with_overweight'),
            FAVC = request.form.get('FAVC'),
            CAEC = request.form.get('CAEC'),
            SMOKE = request.form.get('SMOKE'),
            SCC = request.form.get('SCC'),
            CALC = request.form.get('CALC'),
            MTRANS = request.form.get('MTRANS')
        )
        final_new_data=data.get_data_as_dataframe()
        predict_pipeline=PredictPipeline()
        pred=predict_pipeline.predict(final_new_data)

        results=pred[0]

        return render_template('result.html',final_result=results)






if __name__=="__main__":
    app.run(host='0.0.0.0',debug=True)