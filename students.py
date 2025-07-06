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

class Student:
    def __init__(self, student_id, interests, completed_courses=None, current_courses=None, current_term=1):
        self.student_id = student_id
        self.interests = interests
        self.completed_courses = completed_courses or {}  # Previous terms' courses with grades
        self.current_courses = current_courses or []      # Current term's enrolled courses
        self.current_term = current_term
        self.failed_courses = [c for c, g in self.completed_courses.items() if g == "F"]
        self.GPA = self.calculate_gpa()

    def calculate_gpa(self):
        passed = [grade_points[g] for g in self.completed_courses.values() if g != "F"]
        return round(sum(passed) / len(passed), 2) if passed else 0.0

    def can_take_course(self, course, graph):
        """Check if student can take a course (prerequisites met, not already passed)"""
        # Already passed this course
        if course in self.completed_courses and self.completed_courses[course] != "F":
            return False
        
        # Already enrolled in current term
        if course in self.current_courses:
            return False
        
        # Check prerequisites
        prereqs = list(graph.predecessors(course))
        for prereq in prereqs:
            if prereq not in self.completed_courses or self.completed_courses[prereq] == "F":
                return False
        
        return True

    def get_available_courses(self, graph):
        """Get all courses student can currently take"""
        available = []
        for course in graph.nodes():
            if self.can_take_course(course, graph):
                available.append(course)
        return available

    def get_student(self):
        return {
            "ID": self.student_id,
            "Interests": self.interests,
            "Completed_Courses": self.completed_courses,
            "Current_Courses": self.current_courses,
            "Failed_Courses": self.failed_courses,
            "Current_Term": self.current_term,
            "GPA": self.GPA,
        }

def simulate_student_progression(student, graph, terms_to_simulate):
    """Simulate a student's academic progression over multiple terms"""
    for term in range(terms_to_simulate):
        available_courses = student.get_available_courses(graph)
        
        if not available_courses:
            break
            
        # Choose 3-5 courses based on interests and availability
        course_load = random.randint(3, min(5, len(available_courses)))
        
        # Prioritize courses matching student interests
        prioritized_courses = []
        other_courses = []
        
        for course in available_courses:
            course_category = graph.nodes[course]['category']
            if course_category in student.interests:
                prioritized_courses.append(course)
            else:
                other_courses.append(course)
        
        # Select courses (prioritize interest-aligned courses)
        selected_courses = []
        if prioritized_courses:
            selected_courses.extend(random.sample(
                prioritized_courses, 
                min(len(prioritized_courses), course_load)
            ))
        
        remaining_slots = course_load - len(selected_courses)
        if remaining_slots > 0 and other_courses:
            selected_courses.extend(random.sample(
                other_courses, 
                min(len(other_courses), remaining_slots)
            ))
        
        # Assign grades (better students get better grades)
        for course in selected_courses:
            # Students with higher GPA are more likely to get good grades
            if student.GPA >= 3.5:
                grade = random.choices(grades_list, weights=[0.5, 0.3, 0.15, 0.04, 0.01])[0]
            elif student.GPA >= 3.0:
                grade = random.choices(grades_list, weights=[0.3, 0.4, 0.2, 0.08, 0.02])[0]
            elif student.GPA >= 2.5:
                grade = random.choices(grades_list, weights=[0.2, 0.3, 0.3, 0.15, 0.05])[0]
            else:
                grade = random.choices(grades_list, weights=[0.1, 0.2, 0.3, 0.25, 0.15])[0]
            
            student.completed_courses[course] = grade
        
        # Update student state
        student.failed_courses = [c for c, g in student.completed_courses.items() if g == "F"]
        student.GPA = student.calculate_gpa()

def enroll_current_term_courses(student, graph):
    """Enroll student in current term courses"""
    available_courses = student.get_available_courses(graph)
    
    if not available_courses:
        return
        
    # Choose 3-5 courses based on interests and availability
    course_load = random.randint(3, min(5, len(available_courses)))
    
    # Prioritize courses matching student interests
    prioritized_courses = []
    other_courses = []
    
    for course in available_courses:
        course_category = graph.nodes[course]['category']
        if course_category in student.interests:
            prioritized_courses.append(course)
        else:
            other_courses.append(course)
    
    # Select courses (prioritize interest-aligned courses)
    selected_courses = []
    if prioritized_courses:
        selected_courses.extend(random.sample(
            prioritized_courses, 
            min(len(prioritized_courses), course_load)
        ))
    
    remaining_slots = course_load - len(selected_courses)
    if remaining_slots > 0 and other_courses:
        selected_courses.extend(random.sample(
            other_courses, 
            min(len(other_courses), remaining_slots)
        ))
    
    student.current_courses = selected_courses

# Generate 100 students with varied academic progression
all_students = []

for i in range(100):
    student_interests = random.sample(interests_list, random.randint(1, 2))
    
    # Randomly select current term (1-2)
    current_term = random.randint(1, 2)
    
    # Create student and simulate progression up to their current term
    student = Student(i + 1, student_interests, {}, [], current_term)
    
    # Simulate academic progression up to current term (complete previous terms)
    terms_to_simulate = current_term - 1  # -1 because we start at term 1
    if terms_to_simulate > 0:
        simulate_student_progression(student, curi, terms_to_simulate)
    
    # Enroll in current term courses
    enroll_current_term_courses(student, curi)
    
    all_students.append(student.get_student())

# Save to JSON
with open("AI-Powered-Academic-Advisor/enhanced_student_profiles.json", "w") as f:
    json.dump(all_students, f, indent=2)

print(f"Generated {len(all_students)} students")
print(f"Sample student progression:")
for i in range(5):
    student = all_students[i]
    print(f"Student {student['ID']}: Term {student['Current_Term']}")
    print(f"  Completed: {len(student['Completed_Courses'])} courses, GPA: {student['GPA']}")
    print(f"  Current enrollment: {len(student['Current_Courses'])} courses")
    print(f"  Failed courses: {len(student['Failed_Courses'])}")
    print()

# Show distribution by term
term_distribution = {}
for student in all_students:
    term = student['Current_Term']
    term_distribution[term] = term_distribution.get(term, 0) + 1

print(f"Student distribution by term:")
for term in sorted(term_distribution.keys()):
    print(f"Term {term}: {term_distribution[term]} students")
    
# Show average courses completed by term
term_stats = {1: [], 2: [], 3: []}
for student in all_students:
    term = student['Current_Term']
    completed_count = len(student['Completed_Courses'])
    term_stats[term].append(completed_count)

print(f"\nAverage completed courses by term:")
for term in sorted(term_stats.keys()):
    if term_stats[term]:
        avg = sum(term_stats[term]) / len(term_stats[term])
        print(f"Term {term}: {avg:.1f} courses completed on average")