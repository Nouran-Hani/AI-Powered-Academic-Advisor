import random
import json
import pickle
import networkx as nx
import numpy as np
from datetime import datetime

# Global lists
interests_list = ["AI", "Security", "Data Science", "Elective"]
grades_list = ["A", "B", "C", "D", "F"]
grade_points = {"A": 4.0, "B": 3.0, "C": 2.0, "D": 1.0, "F": 0.0}

# Load the curriculum graph
try:
    with open("AI-Powered-Academic-Advisor/data/curriculum.pkl", "rb") as f:
        curi = pickle.load(f)
    print("Curriculum loaded successfully")
except FileNotFoundError:
    print("Curriculum file not found. Please run curriculum.py first.")
    exit(1)

sorted_courses = list(nx.topological_sort(curi))

class Student:
    def __init__(self, student_id, interests, completed_courses=None, current_term=1):
        self.student_id = student_id
        self.interests = interests
        self.completed_courses = completed_courses or {}  # All courses with grades (only completed)
        self.current_term = current_term
        self.failed_courses = [c for c, g in self.completed_courses.items() if g == "F"]
        self.retaken_courses = []
        self.GPA = self.calculate_gpa()
        self.academic_standing = self.determine_academic_standing()

    def calculate_gpa(self):
        """Calculate GPA based on completed courses (excluding failed courses that weren't retaken)"""
        # For GPA calculation, use the best grade for each course (in case of retakes)
        course_grades = {}
        for course, grade in self.completed_courses.items():
            if course not in course_grades:
                course_grades[course] = grade
            else:
                # If course was retaken, use the better grade
                if grade_points[grade] > grade_points[course_grades[course]]:
                    course_grades[course] = grade
        
        # Calculate GPA from passing grades only
        passing_grades = [grade_points[g] for g in course_grades.values() if g != "F"]
        return round(sum(passing_grades) / len(passing_grades), 2) if passing_grades else 0.0

    def determine_academic_standing(self):
        """Determine academic standing based on GPA"""
        if self.GPA >= 3.5:
            return "Excellent"
        elif self.GPA >= 3.0:
            return "Good"
        elif self.GPA >= 2.5:
            return "Satisfactory"
        elif self.GPA >= 2.0:
            return "Probation"
        else:
            return "Critical"

    def can_take_course(self, course, graph):
        """Check if student can take a course (prerequisites met, not already passed)"""
        # Already passed this course (and didn't fail it most recently)
        if course in self.completed_courses:
            # If the course was failed, can retake it
            if self.completed_courses[course] == "F":
                return True
            else:
                return False  # Already passed, cannot retake
        
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

    def get_graduation_progress(self):
        """Calculate graduation progress as percentage"""
        total_courses = len(curi.nodes)
        # Count unique courses that were passed (not failed)
        passed_courses = set()
        for course, grade in self.completed_courses.items():
            if grade != "F":
                passed_courses.add(course)
        
        return round((len(passed_courses) / total_courses) * 100, 1)

    def get_passed_courses(self):
        """Get list of courses that were passed (not failed)"""
        return [course for course, grade in self.completed_courses.items() if grade != "F"]

    def get_student_dict(self):
        """Get student data as dictionary"""
        passed_courses = self.get_passed_courses()
        return {
            "ID": self.student_id,
            "Interests": self.interests,
            "Completed_Courses": self.completed_courses,
            "Passed_Courses": passed_courses,  # Only courses that were passed
            "Failed_Courses": self.failed_courses,
            "Retaken_Courses": self.retaken_courses,
            "Current_Term": self.current_term,
            "GPA": self.GPA,
            "Academic_Standing": self.academic_standing,
            "Graduation_Progress": self.get_graduation_progress(),
            "Total_Courses_Passed": len(passed_courses),
            "Total_Courses_Failed": len(self.failed_courses),
            "Available_Next_Courses": self.get_available_courses(curi)
        }

def simulate_realistic_student_progression(student, graph, terms_to_simulate):
    """Simulate realistic student progression with varied academic performance"""
    base_success_rate = 0.85  # Base success rate
    
    for term in range(terms_to_simulate):
        available_courses = student.get_available_courses(graph)
        
        if not available_courses:
            break
        
        # Determine course load based on student's academic standing and term
        # Following constraint: max 3-5 courses per term
        if student.academic_standing == "Critical":
            course_load = random.randint(3, 4)  # Lighter load for struggling students
        elif student.academic_standing == "Probation":
            course_load = random.randint(3, 4)
        else:
            course_load = random.randint(3, 5)  # Normal load
        
        course_load = min(course_load, len(available_courses))
        
        # Prioritize courses based on interests and requirements
        selected_courses = select_courses_intelligently(student, available_courses, course_load, graph)
        
        # Simulate grades with realistic distribution
        for course in selected_courses:
            # Adjust success rate based on student's academic history
            success_rate = base_success_rate
            
            if student.GPA >= 3.5:
                success_rate = 0.95
            elif student.GPA >= 3.0:
                success_rate = 0.90
            elif student.GPA >= 2.5:
                success_rate = 0.85
            elif student.GPA >= 2.0:
                success_rate = 0.75
            else:
                success_rate = 0.65
            
            # Check if this is a retake
            if course in student.failed_courses:
                success_rate += 0.15  # Better chance on retake
                student.retaken_courses.append(course)
            
            # Generate grade
            if random.random() < success_rate:
                # Passed - distribute among A, B, C, D
                grade = random.choices(["A", "B", "C", "D"], weights=[0.3, 0.4, 0.2, 0.1])[0]
            else:
                grade = "F"
            
            student.completed_courses[course] = grade
        
        # Update student state
        student.failed_courses = [c for c, g in student.completed_courses.items() if g == "F"]
        student.GPA = student.calculate_gpa()
        student.academic_standing = student.determine_academic_standing()
        student.current_term += 1

def select_courses_intelligently(student, available_courses, course_load, graph):
    """Intelligently select courses based on interests, prerequisites, and requirements"""
    # Categorize available courses
    interest_courses = []
    core_courses = []
    other_courses = []
    
    for course in available_courses:
        category = graph.nodes[course]['category']
        if category in student.interests:
            interest_courses.append(course)
        elif category == "Core":
            core_courses.append(course)
        else:
            other_courses.append(course)
    
    # Selection strategy: prioritize core courses, then interests, then others
    selected = []
    
    # First, add core courses (important for graduation)
    core_needed = min(2, len(core_courses), course_load)
    if core_courses:
        selected.extend(random.sample(core_courses, core_needed))
    
    # Then add interest-aligned courses
    remaining_slots = course_load - len(selected)
    if remaining_slots > 0 and interest_courses:
        interest_to_add = min(remaining_slots, len(interest_courses))
        selected.extend(random.sample(interest_courses, interest_to_add))
    
    # Fill remaining slots with other courses
    remaining_slots = course_load - len(selected)
    if remaining_slots > 0 and other_courses:
        other_to_add = min(remaining_slots, len(other_courses))
        selected.extend(random.sample(other_courses, other_to_add))
    
    return selected

def generate_100_students():
    """Generate exactly 100 students with varied backgrounds"""
    print("Generating 100 students with diverse academic backgrounds...")
    
    all_students = []
    
    # Create students with varied starting points
    for i in range(100):
        student_id = f"STU{i+1:03d}"  # STU001, STU002, etc.
        
        # Generate interests (1-3 interests per student)
        num_interests = random.randint(1, 3)
        student_interests = random.sample(interests_list, num_interests)
        
        # Create student starting from term 1
        student = Student(student_id, student_interests, {}, 1)
        
        # Determine how many terms to simulate (1-8 terms for variety)
        # This creates students at different stages of their academic journey
        terms_completed = random.choices(
            [1, 2, 3, 4, 5, 6, 7, 8], 
            weights=[0.15, 0.15, 0.15, 0.15, 0.15, 0.10, 0.10, 0.05]
        )[0]
        
        # Simulate progression
        if terms_completed > 1:
            simulate_realistic_student_progression(student, curi, terms_completed - 1)
        
        all_students.append(student.get_student_dict())
        
        # Print progress every 20 students
        if (i + 1) % 20 == 0:
            print(f"Generated {i + 1}/100 students...")
    
    return all_students

def analyze_student_cohort(students):
    """Analyze the generated student cohort"""
    print("\n" + "="*60)
    print("STUDENT COHORT ANALYSIS")
    print("="*60)
    
    # Basic statistics
    print(f"Total Students: {len(students)}")
    
    # Term distribution
    term_dist = {}
    for student in students:
        term = student['Current_Term']
        term_dist[term] = term_dist.get(term, 0) + 1
    
    print("\nTerm Distribution:")
    for term in sorted(term_dist.keys()):
        print(f"  Term {term}: {term_dist[term]} students ({term_dist[term]/len(students)*100:.1f}%)")
    
    # GPA distribution
    gpas = [s['GPA'] for s in students if s['GPA'] > 0]
    if gpas:
        print(f"\nGPA Statistics:")
        print(f"  Average GPA: {np.mean(gpas):.2f}")
        print(f"  Median GPA: {np.median(gpas):.2f}")
        print(f"  Min GPA: {np.min(gpas):.2f}")
        print(f"  Max GPA: {np.max(gpas):.2f}")
    
    # Academic standing
    standings = {}
    for student in students:
        standing = student['Academic_Standing']
        standings[standing] = standings.get(standing, 0) + 1
    
    print("\nAcademic Standing Distribution:")
    for standing, count in standings.items():
        print(f"  {standing}: {count} students ({count/len(students)*100:.1f}%)")
    
    # Interest distribution
    interest_counts = {}
    for student in students:
        for interest in student['Interests']:
            interest_counts[interest] = interest_counts.get(interest, 0) + 1
    
    print("\nInterest Distribution:")
    for interest, count in interest_counts.items():
        print(f"  {interest}: {count} students")
    
    # Graduation progress
    progress_values = [s['Graduation_Progress'] for s in students]
    print(f"\nGraduation Progress:")
    print(f"  Average Progress: {np.mean(progress_values):.1f}%")
    print(f"  Students > 25% complete: {sum(1 for p in progress_values if p > 25)}")
    print(f"  Students > 50% complete: {sum(1 for p in progress_values if p > 50)}")
    print(f"  Students > 75% complete: {sum(1 for p in progress_values if p > 75)}")
    
    # Available courses analysis
    available_counts = [len(s['Available_Next_Courses']) for s in students]
    print(f"\nAvailable Next Courses:")
    print(f"  Average: {np.mean(available_counts):.1f} courses available")
    print(f"  Range: {np.min(available_counts)}-{np.max(available_counts)} courses")
    
    # Course completion analysis
    completed_counts = [s['Total_Courses_Passed'] for s in students]
    print(f"\nCourse Completion:")
    print(f"  Average Courses Passed: {np.mean(completed_counts):.1f}")
    print(f"  Range: {np.min(completed_counts)}-{np.max(completed_counts)} courses")
    
    # Sample student profiles
    print("\nSample Student Profiles:")
    for i in range(min(5, len(students))):
        student = students[i]
        print(f"\nStudent {student['ID']}:")
        print(f"  Interests: {student['Interests']}")
        print(f"  Current Term: {student['Current_Term']}, GPA: {student['GPA']}")
        print(f"  Courses Passed: {student['Total_Courses_Passed']}, Failed: {student['Total_Courses_Failed']}")
        print(f"  Available Next: {len(student['Available_Next_Courses'])} courses")
        print(f"  Progress: {student['Graduation_Progress']}%")
        print(f"  Academic Standing: {student['Academic_Standing']}")

def save_students_to_file(students, filename="AI-Powered-Academic-Advisor/data/student_profiles.json"):
    """Save students to JSON file"""
    # Convert numpy arrays to lists for JSON serialization
    serializable_students = []
    for student in students:
        serializable_student = student.copy()
        # Convert any numpy arrays to lists
        for key, value in serializable_student.items():
            if isinstance(value, np.ndarray):
                serializable_student[key] = value.tolist()
        serializable_students.append(serializable_student)
    
    with open(filename, 'w') as f:
        json.dump(serializable_students, f, indent=2)
    print(f"\nStudent profiles saved to {filename}")

def create_test_subset(students, filename="AI-Powered-Academic-Advisor/data/test_student_profiles.json"):
    """Create a subset of students for testing"""
    # Take 10 students for testing - select diverse students
    # Sort by graduation progress to get variety
    sorted_students = sorted(students, key=lambda x: x['Graduation_Progress'])
    
    # Select every 10th student to get variety
    test_students = []
    for i in range(0, len(sorted_students), len(sorted_students)//10):
        if len(test_students) < 10:
            test_students.append(sorted_students[i].copy())
    
    # Reassign test IDs
    for i, student in enumerate(test_students):
        student['ID'] = f"TEST{i+1:03d}"
    
    # Convert numpy arrays to lists for JSON serialization
    serializable_test_students = []
    for student in test_students:
        serializable_student = student.copy()
        for key, value in serializable_student.items():
            if isinstance(value, np.ndarray):
                serializable_student[key] = value.tolist()
        serializable_test_students.append(serializable_student)
    
    with open(filename, 'w') as f:
        json.dump(serializable_test_students, f, indent=2)
    
    print(f"Test subset ({len(test_students)} students) saved to {filename}")

def main():
    """Main function to generate student cohort"""
    print("="*60)
    print("STUDENT COHORT GENERATION - 48 Hours Challenge")
    print("="*60)
    
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Generate 100 students
    students = generate_100_students()
    
    # Analyze the cohort
    analyze_student_cohort(students)
    
    # Save to files
    save_students_to_file(students)
    create_test_subset(students)
    
    print("\n" + "="*60)
    print("STUDENT GENERATION COMPLETE")
    print("="*60)
    print("Files created:")
    print("- student_profiles.json (100 students)")
    print("- test_student_profiles.json (10 students)")
    print("\nStudent cohort ready for AI-powered academic advising!")
    print("\nKey Features:")
    print("✓ No current courses - all courses are completed")
    print("✓ Course load constraint: 3-5 courses per term")
    print("✓ Prerequisite enforcement")
    print("✓ Retake policy for failed courses")
    print("✓ Diverse student interests and academic standings")
    print("✓ Available next courses calculated for each student")

if __name__ == "__main__":
    main()