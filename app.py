import csv

from flask import *
from flask_cors import *

from solve0 import get_data

app = Flask('Scheduler')
CORS(app)


@app.route("/students", methods=['GET'])
def students():
    students = []
    with open('students.csv', 'r') as f:
        reader = csv.reader(f)
        for s in reader:
            students.append({"name": s[0], "schedule": s[1]})
    return jsonify(students)


@app.route("/make-schedule", methods=['POST'])
def make_schedule():
    weights = request.json
    return jsonify(get_data(weights))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
