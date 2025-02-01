import pandas as pd
import numpy as np
from dimod import ConstrainedQuadraticModel, Binary, quicksum
from dwave.system import LeapHybridCQMSampler

# If you wish to simulate locally, you can import:
# from dimod import ExactCQMSolver

###############################################################################
# PARAMETERS
###############################################################################

# Number of available time slots (e.g. 5 slots, indexed 0 to 4).
# We assume slot 2 is noon.
T = 5
# Cost for each time slot based on its distance from noon.
time_cost = {t: abs(t - 2) for t in range(T)}

# Penalty weight for overlapping exams (soft constraint)
P_overlap = 50.0  # Adjust this weight as needed


###############################################################################
# DATA LOADING FUNCTIONS
###############################################################################
def load_student_data(filename='backend/students.csv'):
    """
    Loads student exam data.

    Expected CSV format:
       student,classes
    where the 'classes' column is a comma-separated list of classes.

    Returns:
      - student_classes: dict mapping student id to list of classes.
      - class_students: dict mapping class to list of student ids.
    """
    df = pd.read_csv(filename)
    student_classes = {}
    class_students = {}

    for idx, row in df.iterrows():
        student = row['id']
        # Split the classes string and strip whitespace.
        classes_list = [c.strip() for c in row['classes'].split(',')]
        student_classes[student] = classes_list
        for c in classes_list:
            class_students.setdefault(c, []).append(student)

    return student_classes, class_students


def load_classroom_data(filename='backend/classrooms.csv'):
    """
    Loads classroom data.

    Expected CSV format:
       room,capacity
    where each row gives the room index (or name) and its capacity (number of seats).

    Returns:
      - A dictionary mapping room id to capacity.
    """
    df = pd.read_csv(filename)
    room_capacities = {}
    for idx, row in df.iterrows():
        room = row['id']
        capacity = int(row['capacity'])
        room_capacities[room] = capacity
    return room_capacities


###############################################################################
# BUILD THE CONSTRAINED QUADRATIC MODEL (CQM)
###############################################################################
def build_cqm(student_classes, class_students, room_capacities):
    """
    Constructs a constrained quadratic model for exam scheduling.

    Decision variables:
      For each class c, each time slot t, and each room r that can seat all of c’s students,
      we define a binary variable:
          x[c, t, r] = 1  if class c is scheduled at time slot t in room r,
                       0  otherwise.

    Constraints:
      1. (Assignment) Each class must be assigned exactly one (time, room) pair.
         For each class c:
             sum_{t in T, r in allowed(c)} x[c,t,r] == 1.
      2. (Room capacity) Implicitly enforced by only creating variables for assignments
         where room r’s capacity is at least the enrollment of class c.

    Soft Objective:
      The objective minimizes the total time cost plus a high penalty for overlapping
      exams for the same student at the same time. For every student s and time slot t,
      for each pair of exams scheduled simultaneously (i.e. both assignments active),
      we add a penalty P_overlap.

    Returns:
      - cqm: the ConstrainedQuadraticModel.
      - x: a dictionary mapping keys (c, t, r) to Binary decision variables.
    """

    cqm = ConstrainedQuadraticModel()

    # List of classes and compute enrollment (number of students) for each class.
    classes = list(class_students.keys())
    enrollment = {c: len(class_students[c]) for c in classes}

    # Create decision variables only for assignments where the room can accommodate the class.
    # x[(c, t, r)] is defined only if room r's capacity >= enrollment[c].
    x = {}
    for c in classes:
        for t in range(T):
            for r, capacity in room_capacities.items():
                if capacity >= enrollment[c]:
                    # Create a Binary variable with a unique name.
                    x[(c, t, r)] = Binary(f"x_{c}_{t}_{r}")

    # Constraint 1: Each class must be assigned exactly one (time, room) pair.
    for c in classes:
        valid_assignments = [x[(c, t, r)]
                             for t in range(T)
                             for r in room_capacities
                             if (c, t, r) in x]
        cqm.add_constraint(quicksum(valid_assignments) == 1, label=f"assign_{c}")

    # NOTE: We remove the hard "no overlap" constraint.
    # Instead, we add a high penalty in the objective for each pair of overlapping exams.

    # Objective: Time cost + penalty for overlapping exams.
    # Time cost: For each assignment, cost = time_cost[t].
    time_objective = quicksum(time_cost[t] * x[(c, t, r)]
                              for (c, t, r) in x)

    # Overlap penalty: For each student s and time slot t, for every pair of exams
    # that s has at the same time, add penalty P_overlap.
    overlap_penalty = 0
    for s, classes_s in student_classes.items():
        for t in range(T):
            # Build list of all assignment variables for student s at time t.
            exam_vars = []
            for c in classes_s:
                for r in room_capacities:
                    if (c, t, r) in x:
                        exam_vars.append(x[(c, t, r)])
            # For every pair among these variables, add a quadratic penalty.
            for i in range(len(exam_vars)):
                for j in range(i + 1, len(exam_vars)):
                    overlap_penalty += exam_vars[i] * exam_vars[j]

    # Total objective: minimize time cost plus the overlap penalty.
    total_objective = time_objective + P_overlap * overlap_penalty
    cqm.set_objective(total_objective)

    return cqm, x


###############################################################################
# MAIN FUNCTION: LOAD DATA, BUILD CQM, SOLVE, AND INTERPRET SOLUTION
###############################################################################
def main():
    # Load data from CSV files.
    student_classes, class_students = load_student_data('backend/students.csv')
    room_capacities = load_classroom_data('backend/classrooms.csv')

    # Build the CQM.
    cqm, x = build_cqm(student_classes, class_students, room_capacities)

    print("Number of variables:", len(x))
    print("Number of constraints:", len(cqm.constraints))

    # Choose the solver.
    # For real runs, use:
    sampler = LeapHybridCQMSampler()
    # For local simulation, you could use:
    # from dimod import ExactCQMSolver
    # sampler = ExactCQMSolver()

    print("Submitting the CQM to the solver...")
    solution = sampler.sample_cqm(cqm, time_limit=10)

    # Filter to the feasible solutions.
    feasible_sampleset = solution.filter(lambda d: d.is_feasible)
    if len(feasible_sampleset) == 0:
        print("No feasible solution found!")
        return

    best_sample = feasible_sampleset.first.sample

    # Interpret the solution.
    # For each class, pick the (time, room) assignment where x[c,t,r] is "active."
    schedule = {}
    for (c, t, r), var in x.items():
        # Here we check if the value is at least 0.5.
        if best_sample.get(str(var), 0) >= 0.5:
            if c not in schedule:
                schedule[c] = (t, r)

    # Print the final exam schedule.
    print("\nFinal Exam Schedule:")
    for c, (t, r) in schedule.items():
        print(f"Class '{c}': Time slot {t} (cost {time_cost[t]}), Room '{r}' (Capacity: {room_capacities[r]})")


if __name__ == '__main__':
    main()
