import csv


def get_students_in_class(students: list, cl: str):
    count = 0
    for student in students:
        if cl in student[1]:
            count += 1
    return count


# check no classrooms are double booked, number of students that overlap,
def check_solution(data: list[tuple[str, int, int]]) -> bool:  # (class: str, time: int, room: int)
    classrooms = {}
    with open('classrooms.csv', 'r') as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            classrooms[int(row[0])] = int(row[1])

    with open('students.csv', 'r') as f:
        reader = csv.reader(f)
        next(reader, None)
        students = []
        for row in reader:
            students.append([row[0], row[1:][0].split(',')])
        times_cl = []
        times_classroom = []
        total_times = []
        for cl, time, room in data:
            if (time, room) in times_classroom:
                print(f"Room {room} is occupied by two classes at once")
                return False
            times_cl.append((time, cl))
            if time not in total_times:
                total_times.append(time)
        sorted(total_times)

        # check each time has no student overlaps
        overlap_count = 0
        for t in total_times:
            for c in times_cl:
                if c[0] == t:
                    for s in students:
                        if c[1] in s[1]:
                            overlap_count += 1
        print(f"Number of overlapping students: {overlap_count}")

    #   Check that room is never too full
        for cl, time, room in data:
            num_students = get_students_in_class(students, cl)
            if num_students > classrooms[room]:
                print(f"Classroom {cl} is overbooked! Maximum {classrooms[room]} but has {num_students}")

        for time in total_times:
            print(f"At time {time}:")
            for cl, t, room in data:
                if t == time:
                    print(f"{cl} at room {room} has {get_students_in_class(students, cl)} students")
            print()

        return True


if __name__ == '__main__':
    check_solution([("history", 0, 0), ("pe", 1, 0), ("cs", 1, 1), ("physics", 2, 2), ("photography", 2, 2)])
