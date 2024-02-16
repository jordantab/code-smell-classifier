from flask import Flask, request, jsonify
from flask_cors import CORS
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from scripts.load_data import load_arff_to_dataframe
from scripts.preprocessing import preprocess_dataset
from scripts.train_model import train_random_forest, train_decision_tree


app = Flask(__name__)
CORS(app)

# MongoDB setup
uri = "mongodb+srv://jordantab20:hi@cluster0.n5curom.mongodb.net/?retryWrites=true&w=majority"

client = MongoClient(uri, server_api=ServerApi('1'))

# Send a ping to confirm a successful connectio
try:
    client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print(e)


@app.route('/results', methods=['POST'])
def get_results():
    request_data = request.get_json()
    model_type = request_data['model_type']
    code_smell = request_data['code_smell']

    results = results.find_one({"model": model_type, "code_smell": code_smell}, {"_id": 0})
    if results:
        return jsonify(results)
    else:
        return jsonify({"error": "Model or code smell not found"}), 404


@app.route('/train', methods=['POST'])
def train_models():
    request_data = request.get_json()
    file_name = request_data['file_name']
    model_type = request_data['model_type']

    df = load_arff_to_dataframe(f'data/{file_name}.arff')
    X_train, X_test, y_train, y_test = preprocess_dataset(df)

    if model_type == 'random_forest':
        _, best_rf, initial_accuracy, initial_f1, rf_accuracy, rf_f1, rf_best_params_ = train_random_forest(X_train, X_test, y_train, y_test)
        model_results = {
            "model": "Random Forest",
            "accuracy": rf_accuracy,
            "f1_score": rf_f1,
            "parameters": rf_best_params_
        }
    elif model_type == 'decision_tree':
        _, best_dt, dt_initial_accuracy, dt_initial_f1, dt_accuracy, dt_f1, dt_best_params_ = train_decision_tree(X_train, X_test, y_train, y_test)
        model_results = {
            "model": "Decision Tree",
            "accuracy": dt_accuracy,
            "f1_score": dt_f1,
            "parameters": dt_best_params_
        }

    # Store results in MongoDB
    # results_collection.insert_one(model_results)
    print(jsonify(model_results))

    return jsonify(model_results), 200

if __name__ == '__main__':
    app.run(debug=True)
