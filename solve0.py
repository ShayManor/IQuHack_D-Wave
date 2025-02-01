print("STARTING...")

import dimod
from dwave.system import LeapHybridCQMSampler
# import pprint

PENALTY_WEIGHT = 50.0

def create_exam_scheduling_cqm(
    num_students,
    num_classes, 
    classes_per_student, 
    num_rooms, 
    room_capacity, 
    time_slots, 
    student_classes
):
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
        
    # Constraint 3: Room capacity must not be exceeded
    for t in range(time_slots):
        for d in range(num_rooms):
            cqm.add_constraint(
                sum(x[b, t, d] * len([s for s in range(num_students) if b in student_classes[s]]) for b in range(num_classes)) <= room_capacity,
                label=f'room_{d}_capacity_t{t}'
            )
    
    # Objective: Minimize exams closer to noon
    noon_slot = time_slots // 2
    objective = sum(
        x[b, t, d] * abs(t - noon_slot)
        for b in range(num_classes) for t in range(time_slots) for d in range(num_rooms)
    )

    # Penalty: Overlapping student exams
    penalty_weight = 10  # Adjust as needed
    penalty = sum(
        sum(x[b, t, d] for b in student_classes[s] for d in range(num_rooms)) - 1
        for s in range(num_students) for t in range(time_slots)
    )
    
    cqm.set_objective(objective + penalty_weight * penalty)
    
    return cqm

# Example problem parameters
NUM_STUDENTS = 10       # A
NUM_CLASSES = 5         # B
CLASSES_PER_STUDENT = 2 # C
NUM_ROOMS = 2           # D
ROOM_CAPACITY = 10      # E
TIME_SLOTS = 10         # T

# Randomly assigning students to classes
import random
student_classes = {s: random.sample(range(NUM_CLASSES), CLASSES_PER_STUDENT) for s in range(NUM_STUDENTS)}

# Create CQM model
cqm = create_exam_scheduling_cqm(
    NUM_STUDENTS,
    NUM_CLASSES,
    CLASSES_PER_STUDENT,
    NUM_ROOMS,
    ROOM_CAPACITY,
    TIME_SLOTS,
    student_classes,
)
print("CQM CREATED...")

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



print("DONE.")