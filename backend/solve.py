import dimod
from dwave.system import LeapHybridCQMSampler
import numpy as np

import csv

import json
import check_solution


#------------------------------------------------------------------------------#
OBJECTIVE_WEIGHT = 1

PENALTY_OVERLAPPING = 50
PENALTY_CAPACITY = 2

STUDENT_DATA_FILE = 'backend/students.csv'
ROOM_DATA_FILE = 'backend/classrooms.csv'

TIME_SLOTS_PER_DAY = 6


#------------------------------------------------------------------------------#
def create_exam_scheduling_cqm(
        num_students,
        num_classes,
        num_rooms,
        time_slots_per_day,
        days,
        room_capacity,
        student_classes,
        weights
):
    time_slots = time_slots_per_day * days

    cqm = dimod.ConstrainedQuadraticModel()

    # Precompute the number of students in each class
    class_student_counts = np.array([len([s for s in range(num_students) if b in student_classes[s]]) for b in range(num_classes)])

    # Decision Variables: x[b, t, d] -> 1 if class b is scheduled at time t in room d
    x = {(b, t, d): dimod.Binary(f'x_{b}_{t}_{d}')
         for b in range(num_classes)
         for t in range(time_slots)
         for d in range(num_rooms)
         }

    # Constraint 1: Each class must be assigned exactly once
    for b in range(num_classes):
        cqm.add_constraint(
            sum(x[b, t, d] for t in range(time_slots) for d in range(num_rooms)) == 1,
            label=f'class_{b}_assigned_once'
        )

    # Constraint 2: Room capacity must not be exceeded
    for t in range(time_slots):
        for d in range(num_rooms):
            cqm.add_constraint(
                sum(x[b, t, d] * class_student_counts[b] for b in range(num_classes)) <= room_capacity[d],
                label=f'room_{d}_capacity_t{t}'
            )

    # Constraint 3: Each room is used once at one time
    for t in range(time_slots):
        for d in range(num_rooms):
            cqm.add_constraint(
                sum(x[b, t, d] for b in range(num_classes)) <= 1,
                label=f'room_{d}_time_{t}_once'
            )

    # Objective 1: Minimize exams closer to noon
    objective = sum(
        x[b, t, d] * -weights[t % time_slots_per_day] * class_student_counts[b]
        for b in range(num_classes)
        for t in range(time_slots)
        for d in range(num_rooms)
    )

    # Penalty 1: Overlapping student exams
    penalty_overlapping = sum(
        (sum(x[b, t, d] for b in student_classes[s] for d in range(num_rooms)) *
         (sum(x[b, t, d] for b in student_classes[s] for d in range(num_rooms)) - 1)) / 2
        for s in range(num_students)
        for t in range(time_slots)
    )

    # Penalty 2: Excess room capacity
    penalty_capacity = sum(
        x[b, t, d] * abs(class_student_counts[b] - room_capacity[d]) ** 1.2
        for b in range(num_classes)
        for t in range(time_slots)
        for d in range(num_rooms)
    )

    cqm.set_objective(
        OBJECTIVE_WEIGHT * objective +
        PENALTY_OVERLAPPING * penalty_overlapping +
        PENALTY_CAPACITY * penalty_capacity
    )

    return cqm


#------------------------------------------------------------------------------#
class Student:
    def __init__(self, id, classes):
        self.id = id
        self.classes = classes  # list of class ids


class Room:
    def __init__(self, id, capacity):
        self.id = id
        self.capacity = capacity


def read_student_data(filename):
    all_classes = []  # list[str]
    all_students = []  # list[Student]
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            student_id = int(row[0])

            student_classes = row[1].split(',')
            for c in student_classes:
                if c not in all_classes:
                    all_classes.append(c)

            student_classes = [all_classes.index(c) for c in student_classes]
            all_students.append(Student(student_id, student_classes))

    return all_students, all_classes


def read_room_data(filename):
    all_rooms = []  # list[Room]
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        next(reader, None)
        i = 0
        for row in reader:
            if row[3] == "True":
                room_id = i
                room_capacity = int(row[1])
                all_rooms.append(Room(room_id, room_capacity))
                i += 1

    return all_rooms


#------------------------------------------------------------------------------#
def get_data(weights):
    yield json.dumps({"status": "Starting scheduling"}) + "\n"
    print("\nSTARTING SCHEDULING...")

    #------------------------------------------------------------------------------#
    yield json.dumps({"status": "Loading user data"}) + "\n"
    print("LOADING DATA...")
    days = weights['days']
    students, classes = read_student_data(STUDENT_DATA_FILE)
    rooms = read_room_data(ROOM_DATA_FILE)
    num_students = len(students)
    num_classes = len(classes)
    num_rooms = len(rooms)

    # print(f"Size: {num_classes * num_rooms * TIME_SLOTS_PER_DAY * days}")

    room_capacities = {r.id: r.capacity for r in rooms}
    student_classes = {s.id: s.classes for s in students}

    yield json.dumps({"status": "Data loaded"}) + "\n"
    print("DATA LOADED...")
    #------------------------------------------------------------------------------#

    #------------------------------------------------------------------------------#
    yield json.dumps({"status": "Creating the Constrained Quadratic Model"}) + "\n"
    print("CREATING CQM...")
    cqm = create_exam_scheduling_cqm(
        num_students,
        num_classes,
        num_rooms,
        TIME_SLOTS_PER_DAY,
        days,
        room_capacities,
        student_classes,
        weights["weights"]
    )
    yield json.dumps({"status": "Constrained Quadratic Model created"}) + "\n"
    print("CQM CREATED...")
    #------------------------------------------------------------------------------#

    #------------------------------------------------------------------------------#
    yield json.dumps({"status": "Submitting to solver"}) + "\n"
    print("SUBMITTING TO SOLVER...")
    # Solve with D-Wave's hybrid CQM solver
    sampler = LeapHybridCQMSampler()
    solutions = sampler.sample_cqm(cqm, time_limit=5)
    print("SOLVED...")

    yield json.dumps({"status": "Filtering solutions"}) + "\n"
    print("FILTERING SOLUTIONS...")
    # Filter to feasible solutions
    feasaible = solutions.filter(lambda row: row.is_feasible)
    print("FEASIBLE SOLUTIONS...")

    if len(feasaible) == 0:
        yield json.dumps({"status": "No feasible solutions found."}) + "\n"
        print("No feasible solutions found.")
        return


    best_sample = feasaible.first.sample
    schedule = [k for k, val in best_sample.items() if val == 1]
    print("Optimized Exam Schedule:")
    print(schedule)

    data = []  # list[(class, time, room)]
    all_rooms = read_room_data(ROOM_DATA_FILE)

    for row in schedule:
        # x_b_t_d
        b, t, d = row.split('_')[1:]
        clas = classes[int(b)]
        time = int(t)

        room_index = int(d)
        room = all_rooms[room_index].id

        data.append((clas, time, room))

    
    yield json.dumps({"status": "Assembling final schedule"}) + "\n"
    yield json.dumps(check_solution.check_solution(data, days)) + "\n"
    #------------------------------------------------------------------------------#