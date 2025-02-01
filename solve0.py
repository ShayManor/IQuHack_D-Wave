import dimod
from dwave.system import LeapHybridCQMSampler

def create_exam_scheduling_cqm(A, B, C, D, E, time_slots, student_classes):
    cqm = dimod.ConstrainedQuadraticModel()
    
    # Decision Variables: x[b, t, d] -> 1 if class b is scheduled at time t in room d
    x = {(b, t, d): dimod.Binary(f'x_{b}_{t}_{d}') for b in range(B) for t in range(time_slots) for d in range(D)}
    
    # Constraint 1: Each class must be assigned exactly once
    for b in range(B):
        cqm.add_constraint(sum(x[b, t, d] for t in range(time_slots) for d in range(D)) == 1, label=f'class_{b}_assigned_once')
    
    # Constraint 2: No student should have overlapping exams
    for s in range(A):
        for t in range(time_slots):
            cqm.add_constraint(sum(x[b, t, d] for b in student_classes[s] for d in range(D)) <= 1, 
                               label=f'student_{s}_no_overlap_t{t}')
    
    # Constraint 3: Room capacity must not be exceeded
    for t in range(time_slots):
        for d in range(D):
            cqm.add_constraint(sum(x[b, t, d] * len([s for s in range(A) if b in student_classes[s]]) 
                                   for b in range(B)) <= E, label=f'room_{d}_capacity_t{t}')
    
    # Objective: Minimize exams closer to noon
    noon_slot = time_slots // 2
    objective = sum(x[b, t, d] * abs(t - noon_slot) for b in range(B) for t in range(time_slots) for d in range(D))
    cqm.set_objective(objective)
    
    return cqm

# Example problem parameters
A = 10   # Number of students
B = 5    # Number of classes
C = 2    # Classes per student
D = 2    # Number of rooms
E = 5    # Room capacity
T = 6    # Number of time slots

# Randomly assigning students to classes
import random
student_classes = {s: random.sample(range(B), C) for s in range(A)}

# Create CQM model
cqm = create_exam_scheduling_cqm(A, B, C, D, E, T, student_classes)

# Solve with D-Wave's hybrid CQM solver
sampler = LeapHybridCQMSampler()
solution = sampler.sample_cqm(cqm, time_limit=10)

# Extract results
best_sample = solution.first.sample
schedule = [(b, t, d) for (b, t, d), val in best_sample.items() if val == 1]
print("Optimized Exam Schedule:")
print(schedule)
