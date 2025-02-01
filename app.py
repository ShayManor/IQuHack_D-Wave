import csv

from flask import *
from flask_cors import *

app = Flask('Scheduler')
CORS(app)

@app.route("/students", methods=['GET'])
def students():
    students = []
    with open('students.csv', 'r') as f:
        reader = csv.reader(f)
        for s in reader:
            students.append({"name" : s[0], "schedule" : s[1]})
    return jsonify(students)

@app.route("/make-schedule", methods=['POST'])
def make_schedule(weights):

