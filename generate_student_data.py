import csv
import random

# multiple of the same class so more students have that class
classes = ["physics", "physics", "math", "cs", "history", "history", "history", "biology", "chemistry",
           "english", "pe", "pe", "photography", "health", "health", "economics", "art", "japanese", "french",
           "spanish", "italian"]

num_students = 500
num_classrooms = 3
num_classes_per_student = 7
with open('students.csv', 'w') as f:
    for i in range(num_students):
        student_classes = []
        while len(student_classes) < num_classes_per_student:
            rand_index = random.randint(0, len(classes) - 1)
            print(rand_index)
            if classes[rand_index] not in student_classes:
                student_classes.append(classes[rand_index])
        writer = csv.writer(f)
        writer.writerow([i, ','.join(student_classes)])
