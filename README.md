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