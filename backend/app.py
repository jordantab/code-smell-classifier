from flask import Flask, request, jsonify
from flask_cors import CORS
from pymongo import MongoClient
from scripts.load_data import load_arff_to_dataframe
from scripts.preprocessing import preprocess_dataset
from scripts.train_model import train_random_forest, train_decision_tree

app = Flask(__name__)
CORS(app)

# MongoDB setup
# client = MongoClient('mongodb://localhost:27017/')
# db = client.code_smell_classifier
# results_collection = db.results

if __name__ == '__main__':
    app.run(debug=True)
