import dimod
from dwave.system import LeapHybridCQMSampler

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
    
    # Constraint 2: No student should have overlapping exams
    for s in range(num_students):
        for t in range(time_slots):
            cqm.add_constraint(
                sum(x[b, t, d] for b in student_classes[s] for d in range(num_rooms)) <= 1, 
                label=f'student_{s}_no_overlap_t{t}'
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
        x[b, t, d] * abs(t - noon_slot) for b in range(num_classes) for t in range(time_slots) for d in range(num_rooms)
    )
    cqm.set_objective(objective)
    
    return cqm

# Example problem parameters
NUM_STUDENTS = 10       # A
NUM_CLASSES = 5         # B
CLASSES_PER_STUDENT = 2 # C
NUM_ROOMS = 2           # D
ROOM_CAPACITY = 5       # E
TIME_SLOTS = 6          # T

# Randomly assigning students to classes
import random
student_classes = {s: random.sample(range(B), C) for s in range(A)}

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

# Solve with D-Wave's hybrid CQM solver
sampler = LeapHybridCQMSampler()
solution = sampler.sample_cqm(cqm, time_limit=10)

# Extract results
best_sample = solution.first.sample
schedule = [(b, t, d) for (b, t, d), val in best_sample.items() if val == 1]
print("Optimized Exam Schedule:")
print(schedule)
