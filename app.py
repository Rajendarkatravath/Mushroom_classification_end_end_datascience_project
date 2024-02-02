from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)

application = Flask(__name__)
app = application

@app.route('/')
def index():
    logging.debug("Rendering index page.")
    return render_template('Index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        logging.debug("GET request received at /predictdata.")
        return render_template('home.html')
    else:
        logging.debug("POST request received at /predictdata.")
        # Creating a CustomData object with the user inputs
        data = CustomData(
            cap_shape=request.form.get('cap_shape'),
            cap_surface=request.form.get('cap_surface'),
            cap_color=request.form.get('cap_color'),
            bruises=request.form.get('bruises'),
            odor=request.form.get('odor'),
            gill_attachment=request.form.get('gill_attachment'),
            gill_spacing=request.form.get('gill_spacing'),
            gill_size=request.form.get('gill_size'),
            gill_color=request.form.get('gill_color'),
            stalk_shape=request.form.get('stalk_shape'),
            stalk_root=request.form.get('stalk_root'),
            stalk_surface_above_ring=request.form.get('stalk_surface_above_ring'),
            stalk_surface_below_ring=request.form.get('stalk_surface_below_ring'),
            stalk_color_above_ring=request.form.get('stalk_color_above_ring'),
            stalk_color_below_ring=request.form.get('stalk_color_below_ring'),
            veil_type=request.form.get('veil_type'),
            veil_color=request.form.get('veil_color'),
            ring_number=request.form.get('ring_number'),
            ring_type=request.form.get('ring_type'),
            spore_print_color=request.form.get('spore_print_color'),
            population=request.form.get('population'),
            habitat=request.form.get('habitat')
            
        )

        # Converting the CustomData object to a DataFrame
        pred_df = data.get_data_as_dataframe()

        # Logging the DataFrame (for debugging purposes)
        logging.debug(f"DataFrame for prediction: {pred_df}")

        # Initialize the prediction pipeline
        predict_pipeline = PredictPipeline()
        logging.debug("Prediction pipeline initialized.")

        # Perform the prediction
        results = predict_pipeline.predict(pred_df)
        logging.debug(f"Prediction results: {results}")

        # Interpret the prediction result
        prediction = results[0]
        logging.debug(f"Prediction result: {prediction}")
        if prediction == 0:
            result_message = 'edible'
        else:
            result_message = 'poisonous'

        # Return the results
        return render_template('home.html', result_message=result_message)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
