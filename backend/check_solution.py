import csv


def get_students_in_class(students: list, cl: str):
    count = 0
    for student in students:
        if cl in student[1]:
            count += 1
    return count


def get_overlap(students: list, cl1: str, cl2: str):
    count = 0
    for s in students:
        if cl1 in s[1] and cl2 in s[1]:
            count += 1
    return count


# check no classrooms are double booked, number of students that overlap,
def check_solution(data: list[tuple[str, int, int]], days):  # (class: str, time: int, room: int)
    classrooms = {}
    room_id_to_name = {}
    with open('backend/classrooms.csv', 'r') as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if row[3] == "True":
                room_id_to_name[int(row[0])] = row[2]
                classrooms[int(row[0])] = int(row[1])

    with open('backend/students.csv', 'r') as f:
        reader = csv.reader(f)
        next(reader, None)
        students = []
        for row in reader:
            students.append([row[0], row[1:][0].split(',')])
        times_cl = []

        total_times = []
        total_classes = []
        for cl, time, room in data:
            for cl2, time2, room2 in data:
                if time == time2 and room == room2 and cl != cl2:
                    print(f"Room {room} is occupied by two classes at once")
                    return None

                total_classes.append(cl)
            times_cl.append((time, cl))
            if time not in total_times:
                total_times.append(time)
        total_times.sort()

        # Check each time has no student overlaps
        overlap_count = 0
        for cl1 in times_cl:
            for cl2 in times_cl:
                if cl1[0] == cl2[0] and cl1[1] != cl2[1]:
                    overlap_count += get_overlap(students, cl1[1], cl2[1])
        overlap_count /= 2

        # Check that room is never too full
        schedule = []
        for t in total_times:
            for cl, time, room in data:
                if time == t:
                    schedule.append({
                        "subject": cl,
                        "room": room_id_to_name[room],
                        "students": get_students_in_class(students, cl),
                        "time": time % 6,
                        "day": int(time / 6) + 1
                    })
        return {"schedule": schedule, "overlaps": overlap_count}