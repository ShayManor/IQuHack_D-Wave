print("STARTING...")

import dimod
from dwave.system import LeapHybridCQMSampler
# import pprint
import csv
import check_solution

#------------------------------------------------------------------------------#

#------------------------------------------------------------------------------#
OBJECTIVE_WEIGHT = 1

PENALTY_OVERLAPPING = 50
PENALTY_CAPACITY = 2

STUDENT_DATA_FILE = 'students.csv'
ROOM_DATA_FILE = 'classrooms.csv'

TIME_SLOTS_PER_DAY = 6


#------------------------------------------------------------------------------#


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
                sum(x[b, t, d] * len([s for s in range(num_students) if b in student_classes[s]]) for b in
                    range(num_classes)) <= room_capacity[d],
                label=f'room_{d}_capacity_t{t}'
            )

    # Contraint 3: Each room is used once at one time
    for t in range(time_slots):
        for d in range(num_rooms):
            cqm.add_constraint(
                sum(x[b, t, d] for b in range(num_classes)) <= 1,
                label=f'room_{d}_time_{t}_once'
            )

    # Objective 1: Minimize exams closer to noon
    objective = sum(
        x[b, t, d] * weights[t % time_slots_per_day]
        * len([s for s in range(num_students) if b in student_classes[s]])
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
        x[b, t, d] * abs(len([s for s in range(num_students) if b in student_classes[s]]) - room_capacity[d]) ** 1.2
        for b in range(num_classes)
        for t in range(time_slots)
        for d in range(num_rooms)
    )

    # TODO: optimize to not waste excess room space
    cqm.set_objective(
        OBJECTIVE_WEIGHT * objective +
        PENALTY_OVERLAPPING * penalty_overlapping +
        PENALTY_CAPACITY * penalty_capacity
    )

    return cqm


#------------------------------------------------------------------------------#


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
        for row in reader:
            room_id = int(row[0])
            room_capacity = int(row[1])
            all_rooms.append(Room(room_id, room_capacity))

    return all_rooms


#------------------------------------------------------------------------------#


#------------------------------------------------------------------------------#
def get_data(weights):
    days = weights['days']
    students, classes = read_student_data(STUDENT_DATA_FILE)
    rooms = read_room_data(ROOM_DATA_FILE)
    num_students = len(students)
    num_classes = len(classes)
    num_rooms = len(rooms)

    print(f"Size: {num_classes * num_rooms * TIME_SLOTS_PER_DAY * days}")

    room_capacities = {r.id: r.capacity for r in rooms}
    student_classes = {s.id: s.classes for s in students}

    print("DATA LOADED...")
    #------------------------------------------------------------------------------#

    #------------------------------------------------------------------------------#
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
    print("CQM CREATED...")
    #------------------------------------------------------------------------------#

    #------------------------------------------------------------------------------#
    # Solve with D-Wave's hybrid CQM solver
    sampler = LeapHybridCQMSampler()
    solutions = sampler.sample_cqm(cqm, time_limit=5)
    print("SOLVED...")

    # Filter to feasible solutions
    feasaible = solutions.filter(lambda row: row.is_feasible)
    print("FEASIBLE SOLUTIONS...")

    # Extract results
    if len(feasaible) == 0:
        print("No feasible solutions found.")
    else:
        # for datum in feasaible.data(fields=['sample', 'energy']):
        #     pprint.pprint(datum)

        best_sample = feasaible.first.sample
        schedule = [k for k, val in best_sample.items() if val == 1]
        print("Optimized Exam Schedule:")
        print(schedule)

        data = []  # list[(class, time, room)]

        for row in schedule:
            # x_b_t_d
            b, t, d = row.split('_')[1:]
            clas = classes[int(b)]
            time = int(t)
            room = int(d)
            data.append((clas, time, room))

        check_solution.check_solution(data)
        return data
    #------------------------------------------------------------------------------#

    print("DONE.")
