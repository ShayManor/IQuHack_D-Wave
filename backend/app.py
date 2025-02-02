import csv

from flask import *
from flask_cors import *

from solve import get_data

app = Flask('Scheduler')
CORS(app)


@app.route("/students", methods=['GET'])
def students():
    students = []
    with open('backend/students.csv', 'r') as f:
        reader = csv.reader(f)
        for s in reader:
            students.append({"name": s[0], "schedule": s[1]})
    return jsonify(students)


@app.route("/make-schedule", methods=['POST'])
def make_schedule():
    weights = request.json
    return stream_with_context(get_data(weights))


@app.route("/get-classrooms", methods=['GET'])
def get_classrooms():
    data = []
    with open("backend/classrooms.csv", "r") as f:
        next(f)
        reader = csv.reader(f)
        for row in reader:
            data.append({"id": int(row[0]),
                         "capacity": row[1],
                         "name": row[2],
                         "selected": row[3] == "False",
                         })
        print(data)
    return jsonify(data)


@app.route("/update-classrooms", methods=['POST'])
def update_classrooms():
    classrooms = request.json["classrooms"]
    with open("backend/classrooms.csv", "w") as w:
        writer = csv.writer(w)
        writer.writerow(["id", "capacity"])
        for classroom in classrooms:
            writer.writerow([classroom["id"], classroom["capacity"], classroom["name"], classroom["selected"]])
    return classrooms


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
