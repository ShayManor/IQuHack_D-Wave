import os
import csv
import random

NUM_STUDENTS = 400
TARGET_CLASSES_PER_STUDENT = 4
CORE_CHANCE = 0.7

STUDENT_DATA_FILE = 'students.csv'

classes = { # classes[subject] = [class1, class2, ...]
    "science": ["Physics", "Computer Science"],
    "math": ["Algebra", "Calculus", "Computer Science"],
    "history": ["World History", "US History", "European History"],
    "language": ["Spanish", "Chinese"],
    "arts": ["Art", "Photography"],
    "finance": ["Economics", "Accounting"],
}

def generate_student_data():
    if os.path.exists(STUDENT_DATA_FILE):
        os.remove(STUDENT_DATA_FILE)

    with open(STUDENT_DATA_FILE, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['student', 'classes'])
        
        for _ in range(NUM_STUDENTS):
            core_subject = random.choice(list(classes.keys()))
            num_classes = random.randint(TARGET_CLASSES_PER_STUDENT - 2, TARGET_CLASSES_PER_STUDENT + 1)
            
            student_classes = []
            while len(student_classes) < num_classes:
                if random.random() < CORE_CHANCE:
                    subject = core_subject
                else:
                    subject = random.choice(list(classes.keys()))
                
                class_name = random.choice(classes[subject])
                if class_name not in student_classes:
                    student_classes.append(class_name)

            writer.writerow([_, ','.join(student_classes)])

    

if __name__ == '__main__':
    generate_student_data()