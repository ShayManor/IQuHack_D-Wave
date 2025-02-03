# iQuHACK 2025 In-Person D-WAVE Challenge

Submission for the MIT iQuHACK 2025 in-person D-WAVE challenge by Shay Manor and Millan Kumar.
Uses D-WAVE's CQM (Constrained Quadratic Model) quantum annealer to build the perfect finals schedule.

### Links

- In-person D-WAVE challenge prompt: https://github.com/iQuHACK/2025-D-Wave
- MIT iQuHACK 2025 home page: https://www.iquise.mit.edu/iQuHACK/2025-01-31

## Code Overview

- `frontend/`:
	- Contains an HTML file that displays the admin panel.
	- From this panel, you can easily change:
		- The amount of days testing is spread over.
		- Which classrooms are allowed to be use.
		- During what part of the day you would prefer to have finals.
- `backend/`:
	- `app.py`: runs a Flask server that handles handles requests from the frontend:
		- Handle checkbox (classrooms allowed) changes (and save them to a file).
		- Handle the main request to generate the finals schedule
			- Both using a classical algorithm and the D-WAVE quantum annealer.
	- `c_solver.py`: contains the classical algorithm to solve the finals scheduling problem.
	- `solve.py`: contains the D-WAVE quantum annealer algorithm to solve the finals scheduling problem.
	- `check_solution.py`: verfies that the solution is indeed valid (sanity check that our contraints were correctly implemented) and is responsible for converting the output into JSON to be sent to the frontend.
	- `generate_student_data.py`: generates random student data to be used in the finals scheduling problem.

## CQM (Constrained Quadratic Model)

- Unlike other teams which used matrices, we used Python dictionaries to represent the CQM. \
- Decision Variables: `x[b, t, d]` -> `1` if class `b` is scheduled at time `t` in room `d`:
```python
x = {(b, t, d): dimod.Binary(f'x_{b}_{t}_{d}')
		for b in range(num_classes)
		for t in range(time_slots)
		for d in range(num_rooms)
	}
```

### Contraints

Because we used a CQM instead of a BQM, we could directly use contraints (things that are guaranteed to be true, no compromises) instead of implementing them as penalties.
```python
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
```

### Objectives/Penalties

- The objective is to minimize the weighted sum of the classes scheduled at each time slot (weighted more = time slot is more desirable).
- Penalties are used to discourage the solver from generating certain solutions, but are not guaranteed to be enforced. For example, it might be possible that it is impossible to create a schedule were no students have overlapping exams, but we still want to try to find the solution with the least overlapping exams.
- The objectives and penalties are combined using weights.
	- For example: the weight for overlapping exams is significantly higher than the other weights so if there is a solution with no overlapping exams, it will be chosen.
```python
# Objective 1: Priorize classes during weighted times
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
# Combine objective and penalties using weights
cqm.set_objective(
	OBJECTIVE_WEIGHT * objective +
	PENALTY_OVERLAPPING * penalty_overlapping +
	PENALTY_CAPACITY * penalty_capacity
)
```