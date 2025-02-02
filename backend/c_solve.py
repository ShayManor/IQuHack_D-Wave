import csv
import json
import math
import random
import time

# Global parameters (same as in your original code)
OBJECTIVE_WEIGHT = 1
PENALTY_OVERLAPPING = 50
PENALTY_CAPACITY = 2
# A “big‐M” penalty for hard constraints violations (room capacity and room/time conflicts)
M = 1e6

STUDENT_DATA_FILE = 'backend/students.csv'
ROOM_DATA_FILE = 'backend/classrooms.csv'
TIME_SLOTS_PER_DAY = 6


# ------------------------------------------------------------------------------
# Data Classes

class Student:
    def __init__(self, id, classes):
        self.id = id
        self.classes = classes  # list of class indices


class Room:
    def __init__(self, id, index, capacity):
        self.id = id
        self.index = index  # internal index (0,1,...)
        self.capacity = capacity


# ------------------------------------------------------------------------------
# Data reading functions

def read_student_data(filename):
    """
    Reads student data from CSV.
    Expected CSV format (with header):
        student_id, classes (comma-separated)
    Returns a tuple (students, all_classes) where:
      - students is a list of Student objects;
      - all_classes is a list of unique class names.
    """
    all_classes = []  # list of unique class names
    all_students = []  # list of Student objects
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        next(reader, None)  # skip header
        for row in reader:
            student_id = int(row[0])
            # row[1] is assumed to be comma-separated list of class names (strings)
            student_class_names = row[1].split(',')
            # record new class names as they appear
            for cname in student_class_names:
                if cname not in all_classes:
                    all_classes.append(cname)
            # convert names to indices
            student_class_indices = [all_classes.index(cname) for cname in student_class_names]
            all_students.append(Student(student_id, student_class_indices))
    return all_students, all_classes


def read_room_data(filename):
    """
    Reads room data from CSV.
    Expected CSV format (with header) where each row has:
      room_id, capacity, ... , active_flag (a string "True" if the room is active)
    Only active rooms are returned.
    """
    all_rooms = []  # list of Room objects
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        next(reader, None)  # skip header
        i = 0
        for row in reader:
            # Check the active flag (assumed to be in column 3)
            if row[3].strip() == "True":
                room_id = int(row[0])
                room_capacity = int(row[1])
                all_rooms.append(Room(room_id, i, room_capacity))
                i += 1
    return all_rooms


# ------------------------------------------------------------------------------
# Cost (objective+penalty) function

def compute_cost(schedule, students, class_student_counts, room_capacities,
                 time_weights, time_slots_per_day, total_time_slots):
    """
    Given a schedule (a dict mapping each class index b to (t, r)) compute
    the overall cost as the sum of:
      - the linear objective (weighted by time slot importance and room capacity penalty)
      - a penalty for each student having overlapping exams
      - heavy penalties for any schedule that violates hard constraints:
          * a room assigned more than once at a given time slot
          * a class scheduled in a room with insufficient capacity.
    """
    cost = 0.0
    num_classes = len(class_student_counts)

    # (1) For each class assignment, add the objective and room capacity penalty.
    for b in range(num_classes):
        t, r = schedule[b]
        # Objective: prefer time slots with higher weight (note the minus sign, as in your code)
        cost += -time_weights[t % time_slots_per_day] * class_student_counts[b]
        # Add the (nonlinear) penalty on room capacity mismatch.
        cost += PENALTY_CAPACITY * (abs(class_student_counts[b] - room_capacities[r]) ** 1.2)
        # Hard penalty if the class does not fit the room.
        if class_student_counts[b] > room_capacities[r]:
            cost += M

    # (2) Add heavy penalty if more than one exam is scheduled in the same room at the same time.
    assignments = {}
    for b, (t, r) in schedule.items():
        assignments.setdefault((t, r), []).append(b)
    for key, clist in assignments.items():
        if len(clist) > 1:
            # For each extra exam, add a big penalty.
            conflict_count = len(clist) - 1
            cost += M * conflict_count

    # (3) Add overlapping exam penalty for each student.
    # For each student and each time slot, count the number of exams they have.
    for student in students:
        time_count = {}
        for b in student.classes:
            # Every class should have an assignment.
            t, _ = schedule[b]
            time_count[t] = time_count.get(t, 0) + 1
        # If a student has more than one exam at a time, add a penalty.
        for cnt in time_count.values():
            if cnt > 1:
                cost += PENALTY_OVERLAPPING * (cnt * (cnt - 1) / 2)
    return cost


# ------------------------------------------------------------------------------
# Classical solver using simulated annealing

def classical_exam_scheduling(weights, time_limit=5):
    """
    This generator yields JSON messages (status updates and final schedule)
    while it performs simulated annealing to optimize the exam schedule.

    The `weights` dictionary is expected to have two keys:
      - "days": number of days,
      - "weights": a list of length TIME_SLOTS_PER_DAY containing the weight for each time slot.
    """
    yield json.dumps({"status": "Starting scheduling"}) + "\n"
    print("\nSTARTING SCHEDULING...")

    # --- Data loading ---
    yield json.dumps({"status": "Loading user data"}) + "\n"
    print("LOADING DATA...")
    students, classes = read_student_data(STUDENT_DATA_FILE)
    rooms = read_room_data(ROOM_DATA_FILE)
    num_students = len(students)
    num_classes = len(classes)
    num_rooms = len(rooms)

    # Build a mapping from room index (internal) to capacity.
    room_capacities = {r.index: r.capacity for r in rooms}

    # Compute the number of students in each class.
    class_student_counts = [0] * num_classes
    for student in students:
        for b in student.classes:
            class_student_counts[b] += 1

    yield json.dumps({"status": "Data loaded"}) + "\n"
    print("DATA LOADED...")

    # --- Domain definitions ---
    days = weights['days']
    total_time_slots = TIME_SLOTS_PER_DAY * days
    time_weights = weights["weights"]  # list of length TIME_SLOTS_PER_DAY

    # --- Initial solution ---
    # Represent schedule as a dict: for each class b, assign (time_slot, room)
    schedule = {}
    for b in range(num_classes):
        # Try to choose a room that is large enough; if none, choose any room.
        possible_rooms = [r for r in range(num_rooms) if room_capacities[r] >= class_student_counts[b]]
        if not possible_rooms:
            possible_rooms = list(range(num_rooms))
        t = random.randint(0, total_time_slots - 1)
        r = random.choice(possible_rooms)
        schedule[b] = (t, r)

    # Evaluate the initial cost.
    current_cost = compute_cost(schedule, students, class_student_counts,
                                room_capacities, time_weights, TIME_SLOTS_PER_DAY, total_time_slots)
    best_schedule = schedule.copy()
    best_cost = current_cost

    # --- Simulated annealing parameters ---
    T = 100.0  # initial temperature
    T_min = 1e-3  # minimum temperature
    alpha = 0.995  # cooling rate
    max_iterations = 10000
    iteration = 0
    start_time = time.time()

    yield json.dumps({"status": "Starting optimization"}) + "\n"
    print("STARTING OPTIMIZATION...")

    while T > T_min and iteration < max_iterations and (time.time() - start_time) < time_limit:
        iteration += 1
        # Pick a random class to modify.
        b = random.randint(0, num_classes - 1)
        current_t, current_r = schedule[b]
        # Propose a new assignment: change time, room, or both.
        new_t = current_t
        new_r = current_r
        if random.random() < 0.5:
            new_t = random.randint(0, total_time_slots - 1)
        if random.random() < 0.5:
            new_r = random.randint(0, num_rooms - 1)
        new_schedule = schedule.copy()
        new_schedule[b] = (new_t, new_r)
        new_cost = compute_cost(new_schedule, students, class_student_counts,
                                room_capacities, time_weights, TIME_SLOTS_PER_DAY, total_time_slots)
        delta = new_cost - current_cost

        # Accept the move if it improves cost or with a Boltzmann probability.
        if delta < 0 or random.random() < math.exp(-delta / T):
            schedule = new_schedule
            current_cost = new_cost
            if new_cost < best_cost:
                best_schedule = new_schedule.copy()
                best_cost = new_cost

        T *= alpha

    print("OPTIMIZATION COMPLETE")
    yield json.dumps({"status": "Optimization complete"}) + "\n"

    # --- Assemble final schedule ---
    # Produce a list of tuples (class, time_slot, room_id).
    final_schedule = []
    for b, (t, r) in best_schedule.items():
        # Get the class name.
        clas = classes[b]
        # Map the internal room index to the room's ID.
        room_id = next((room.id for room in rooms if room.index == r), r)
        final_schedule.append((clas, t, room_id))

    print("Optimized Exam Schedule:")
    print(final_schedule)

    yield json.dumps({"status": "Assembling final schedule"}) + "\n"
    # If you have a check_solution function (as in your original code) call it:
    try:
        import check_solution
        result = check_solution.check_solution(final_schedule, days)
        yield json.dumps(result) + "\n"
    except ImportError:
        # Otherwise, just yield the schedule.
        yield json.dumps({"schedule": final_schedule}) + "\n"


# ------------------------------------------------------------------------------
# For testing purposes, you might run the classical solver as a script.
if __name__ == "__main__":
    # Example weights dictionary.
    # (Here we assume there are 6 time slots per day; you can adjust the numbers.)
    example_weights = {
        "days": 3,  # for example, schedule exams over 3 days (i.e. 18 time slots)
        "weights": [1.0, 1.5, 2.0, 1.5, 1.0, 0.5]
    }
    # Run the solver and print all JSON status messages.
    for msg in classical_exam_scheduling(example_weights, time_limit=5):
        print(msg.strip())
