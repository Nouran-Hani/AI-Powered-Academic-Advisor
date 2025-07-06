import random
import json
import pickle
import networkx as nx

# Global lists
interests_list = ["AI", "Security", "Data Science", "Elective"]
grades_list = ["A", "B", "C", "D", "F"]
grade_points = {"A": 4.0, "B": 3.0, "C": 2.0, "D": 1.0, "F": 0.0}

# Load the curriculum graph
with open("AI-Powered-Academic-Advisor/curriculum.pkl", "rb") as f:
    curi = pickle.load(f)

sorted_courses = list(nx.topological_sort(curi))

# Student class
class Student:
    def __init__(self, student_id, interests, courses=None):
        self.student_id = student_id
        self.interests = interests
        self.courses = courses or {}
        self.GPA = self.calculate_gpa()

    def calculate_gpa(self):
        passed = [grade_points[g] for g in self.courses.values() if g != "F"]
        return round(sum(passed) / len(passed), 2) if passed else 0.0

    def get_student(self):
        return {
            "ID": self.student_id,
            "Interests": self.interests,
            "Courses": self.courses,
            "Failed_Courses": [c for c, g in self.courses.items() if g == "F"],
            "GPA": self.GPA
        }

# Helper: Generate realistic course record based on prerequisites
def generate_first_term_courses(graph, sorted_courses, min_courses=3, max_courses=5):
    completed = {}
    available_courses = []

    for course in sorted_courses:
        prereqs = list(graph.predecessors(course))
        if all(pr in completed and completed[pr] != "F" for pr in prereqs):
            available_courses.append(course)
            if len(available_courses) >= max_courses:
                break

    selected_courses = random.sample(available_courses, k=random.randint(min_courses, min(len(available_courses), max_courses)))
    
    for course in selected_courses:
        grade = random.choices(grades_list, weights=[0.3, 0.3, 0.2, 0.1, 0.1])[0]
        completed[course] = grade

    return completed


# Generate 100 students
all_students = []

for i in range(100):
    student_interests = random.sample(interests_list, random.randint(1, 2))
    course_history = generate_first_term_courses(curi, sorted_courses)
    student = Student(i + 1, student_interests, course_history)
    all_students.append(student.get_student())


# Save to JSON
with open("AI-Powered-Academic-Advisor/student_profiles.json", "w") as f:
    json.dump(all_students, f, indent=2)
