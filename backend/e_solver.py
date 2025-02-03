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
# Exact solver using exhaustive search with backtracking

def exhaustive_exam_scheduling(weights, time_limit=600):
    """
    This generator yields JSON messages (status updates and final schedule)
    while it performs an exhaustive search with backtracking to find the optimal exam schedule.

    The `weights` dictionary is expected to have two keys:
      - "days": number of days,
      - "weights": a list of length TIME_SLOTS_PER_DAY containing the weight for each time slot.

    NOTE: This method is exact and guarantees the optimal solution, but its runtime grows exponentially
          with the number of classes.
    """
    start_time = time.time()
    yield json.dumps({"status": "Starting exhaustive scheduling"}) + "\n"
    print("\nSTARTING EXHAUSTIVE SCHEDULING...")

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

    # --- Domain reduction: Precompute feasible assignments for each class ---
    # For this exact search, we only consider (time, room) pairs where the room can fit the class.
    domain = {}
    for b in range(num_classes):
        domain[b] = []
        for t in range(total_time_slots):
            for r in range(num_rooms):
                if room_capacities[r] >= class_student_counts[b]:
                    domain[b].append((t, r))
        if not domain[b]:
            # If no room is big enough, allow all assignments (they will incur a heavy penalty)
            for t in range(total_time_slots):
                for r in range(num_rooms):
                    domain[b].append((t, r))

    # --- Backtracking setup ---
    best_schedule = None
    best_cost = float('inf')

    # We'll keep track of room usage (t, r) to enforce that no two classes share the same room at the same time.
    room_used = {(t, r): False for t in range(total_time_slots) for r in range(num_rooms)}

    # Define the recursive backtracking function.
    def backtrack(b, current_schedule):
        nonlocal best_schedule, best_cost
        # If all classes have been assigned, evaluate the complete schedule.
        if b == num_classes:
            cost = compute_cost(current_schedule, students, class_student_counts,
                                room_capacities, time_weights, TIME_SLOTS_PER_DAY, total_time_slots)
            if cost < best_cost:
                best_cost = cost
                best_schedule = current_schedule.copy()
            return

        # For class b, iterate over all possible (time, room) assignments in its domain.
        for (t, r) in domain[b]:
            # Enforce the hard constraint that a room cannot be used by more than one exam at the same time.
            if room_used[(t, r)]:
                continue
            # Assign class b to (t, r).
            current_schedule[b] = (t, r)
            room_used[(t, r)] = True
            backtrack(b + 1, current_schedule)
            room_used[(t, r)] = False  # Backtrack (undo the assignment)

    # --- Start the exhaustive search ---
    start_time = time.time()
    backtrack(0, {})
    elapsed = time.time() - start_time
    yield json.dumps({"status": "Exhaustive search complete", "time": elapsed}) + "\n"
    print("EXHAUSTIVE SEARCH COMPLETE in {:.2f} seconds".format(elapsed))

    # --- Assemble final schedule ---
    final_schedule = []
    for b, (t, r) in best_schedule.items():
        # Get the class name.
        clas = classes[b]
        # Map the internal room index to the room's ID.
        room_id = next((room.id for room in rooms if room.index == r), r)
        final_schedule.append((clas, t, room_id))

    print("Optimized Exam Schedule:")
    print(final_schedule)
    yield json.dumps({"status": "Assembling final schedule", "schedule": final_schedule}) + "\n"

    # Optionally, if you have a check_solution function, call it:
    try:
        import check_solution
        result = check_solution.check_solution(final_schedule, days)
        yield json.dumps(result) + "\n"
    except ImportError:
        # Otherwise, just yield the schedule.
        yield json.dumps({"schedule": final_schedule}) + "\n"
    print(f"Real final time: {time.time() - start_time}")


# ------------------------------------------------------------------------------
# For testing purposes, you might run the exact solver as a script.
if __name__ == "__main__":
    # Example weights dictionary.
    # (Here we assume there are 6 time slots per day; you can adjust the numbers.)
    example_weights = {
        "days": 3,  # for example, schedule exams over 3 days (i.e. 18 time slots)
        "weights": [1.0, 1.5, 2.0, 1.5, 1.0, 0.5]
    }
    # Run the solver and print all JSON status messages.
    for msg in exhaustive_exam_scheduling(example_weights, time_limit=600):
        print(msg.strip())
