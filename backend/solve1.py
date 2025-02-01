import pandas as pd
import numpy as np
import itertools
from dwave.system import DWaveSampler, EmbeddingComposite

###############################################################################
# PARAMETERS & PENALTY WEIGHTS
###############################################################################

# Number of discrete time slots available.
T = 5  # For example, 5 time slots

# Number of available rooms (D) and room capacity (E seats).
D = 3  # Number of rooms (adjust as needed)
E = 30  # Seats per room

# Penalty weights (tune these as necessary)
P_assign = 10.0  # Penalty for a class not getting exactly one exam slot.
P_capacity = 20.0  # Penalty for exceeding a room's capacity.
P_overlap = 30.0  # Penalty for a student having more than one exam at a time.
P_time = 1.0  # Weight on the time cost

# Define a cost for each time slot based on its distance from noon.
# Here we assume time slot 2 (0-indexed) is noon.
time_cost = [abs(t - 2) for t in range(T)]  # e.g., [2, 1, 0, 1, 2]


###############################################################################
# HELPER FUNCTION TO ADD TERMS TO THE QUBO DICTIONARY
###############################################################################
def add_to_qubo(key, value, Q):
    """
    Add the value to QUBO dictionary Q at key.
    The key is a tuple (i, j) representing a quadratic (or linear if i == j) term.
    """
    if key in Q:
        Q[key] += value
    else:
        Q[key] = value


###############################################################################
# DATA LOADING: READ THE STUDENTS CSV
###############################################################################
def load_student_data(filename='backend/students.csv'):
    """
    Load student data from a CSV file.
    Expected CSV format:
        student,classes
    where "classes" is a comma‑separated list of class names.

    Returns:
      - student_classes: dict mapping student id to list of classes.
      - class_students: dict mapping each class to a list of student ids.
    """
    df = pd.read_csv(filename)
    student_classes = {}
    class_students = {}

    for idx, row in df.iterrows():
        student = row['student']
        # Split the "classes" string on commas and strip extra spaces.
        classes_list = [c.strip() for c in row['classes'].split(',')]
        student_classes[student] = classes_list
        for c in classes_list:
            if c not in class_students:
                class_students[c] = []
            class_students[c].append(student)
    return student_classes, class_students


###############################################################################
# BUILD THE QUBO FOR THE FINAL SCHEDULING PROBLEM
###############################################################################
def build_qubo(student_classes, class_students):
    """
    Build a QUBO dictionary for the exam scheduling problem.

    Decision variables:
      For each class c, time slot t, and room r,
         x_{c,t,r} = 1 if class c is scheduled at time t in room r, else 0.

    Constraints and objectives:
      1. Assignment: Each class is scheduled exactly once.
         Penalty: P_assign * (sum_{t,r} x_{c,t,r} - 1)^2.
      2. Room capacity: For each time slot and room, the total number of students
         taking exams there must not exceed E.
         Penalty: P_capacity * (sum_{c} (n_c * x_{c,t,r}) - E)^2.
      3. No overlapping exams: A student cannot have two exams at the same time.
         For each student and time slot, add penalty for every pair of classes scheduled.
      4. Time cost: Each assignment has an additional cost depending on the time slot.
         (Time slots farther from noon incur a higher cost.)

    Returns:
      - Q: A dictionary representing the QUBO.
      - variables: A list of all variable keys (each is a tuple (c, t, r)).
    """
    Q = {}
    # List of all classes
    classes = list(class_students.keys())
    # Number of students registered in each class
    class_size = {c: len(class_students[c]) for c in classes}

    # Create decision variables: for each class c, time slot t, room r.
    variables = []
    for c in classes:
        for t in range(T):
            for r in range(D):
                variables.append((c, t, r))

    # -------------------------------
    # Constraint 1: Each class gets exactly one exam slot.
    # For each class c, let S = {(c,t,r) for all t, r}.
    # We add a penalty: P_assign * (sum_{i in S} x_i - 1)^2.
    # Expanding the square gives:
    #   2*P_assign * sum_{i<j in S} x_i x_j - P_assign * sum_{i in S} x_i  + constant.
    # (The constant is ignored.)
    for c in classes:
        S = [(c, t, r) for t in range(T) for r in range(D)]
        for i in range(len(S)):
            var_i = S[i]
            # Linear term: -P_assign * x_i.
            add_to_qubo((var_i, var_i), -P_assign, Q)
            for j in range(i + 1, len(S)):
                var_j = S[j]
                key = tuple(sorted((var_i, var_j)))
                add_to_qubo(key, 2 * P_assign, Q)

    # -------------------------------
    # Constraint 2: Room capacity constraint.
    # For each time slot t and room r, let Y_{t,r} = sum_{c in classes} n_c * x_{c,t,r}.
    # We penalize deviations from capacity E via:
    #   P_capacity * (Y_{t,r} - E)^2.
    # Expanding gives linear and quadratic terms.
    for t in range(T):
        for r in range(D):
            S = [(c, t, r) for c in classes]
            for i in range(len(S)):
                var_i = S[i]
                n_i = class_size[var_i[0]]
                linear_coeff = P_capacity * (n_i ** 2 - 2 * E * n_i)
                add_to_qubo((var_i, var_i), linear_coeff, Q)
                for j in range(i + 1, len(S)):
                    var_j = S[j]
                    n_j = class_size[var_j[0]]
                    key = tuple(sorted((var_i, var_j)))
                    add_to_qubo(key, 2 * P_capacity * n_i * n_j, Q)

    # -------------------------------
    # Constraint 3: No overlapping exams for students.
    # For each student s and time slot t, if s is registered in more than one class,
    # we add a penalty P_overlap for each pair of exams scheduled simultaneously.
    for s, classes_s in student_classes.items():
        for t in range(T):
            # Build list S of all variables for student s at time t.
            S = []
            for c in classes_s:
                # Only add if c is a recognized class.
                if c in classes:
                    for r in range(D):
                        S.append((c, t, r))
            # For every pair in S, add a penalty.
            for i in range(len(S)):
                var_i = S[i]
                for j in range(i + 1, len(S)):
                    var_j = S[j]
                    key = tuple(sorted((var_i, var_j)))
                    add_to_qubo(key, P_overlap, Q)

    # -------------------------------
    # Objective: Time cost.
    # For each variable (c,t,r), add a cost proportional to how far the time slot is
    # from noon (time_cost[t]). Lower cost is better.
    for c in classes:
        for t in range(T):
            for r in range(D):
                var = (c, t, r)
                add_to_qubo((var, var), P_time * time_cost[t], Q)

    return Q, variables


###############################################################################
# MAIN FUNCTION: BUILD, SOLVE, AND INTERPRET THE QUBO
###############################################################################
def main():
    # Load the student schedule data.
    student_classes, class_students = load_student_data('backend/students.csv')

    # Build the QUBO formulation.
    Q, variables = build_qubo(student_classes, class_students)

    # Create a sampler using D-Wave's EmbeddingComposite.
    sampler = EmbeddingComposite(DWaveSampler())

    # Solve the QUBO. (num_reads may be adjusted.)
    print("Submitting QUBO to D-Wave...")
    sampleset = sampler.sample_qubo(Q, num_reads=50)

    # Retrieve the best solution.
    best_sample = sampleset.first.sample

    # Interpret the solution:
    # For each class, choose the (time, room) assignment where the corresponding
    # variable is 1.
    schedule = {}
    for var in variables:
        if best_sample.get(var, 0) == 1:
            c, t, r = var
            if c in schedule:
                # In a valid solution each class should be assigned only once.
                # If multiple assignments appear, we simply keep the first one.
                pass
            else:
                schedule[c] = (t, r)

    # Print the final exam schedule.
    print("\nFinal Exam Schedule:")
    for c, (t, r) in schedule.items():
        print(f"Class {c}: Time slot {t} (cost {time_cost[t]}), Room {r}")


if __name__ == '__main__':
    main()
