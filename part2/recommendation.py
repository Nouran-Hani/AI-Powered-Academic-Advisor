import pickle
import gymnasium as gym
from gymnasium import spaces
import random
import numpy as np
from copy import deepcopy
import networkx as nx

# Define interest categories to match curriculum
INTEREST_CATEGORIES = ["AI", "Security", "Data Science", "Elective", "Core"]
GRADE_POINTS = {"A": 4.0, "B": 3.0, "C": 2.0, "D": 1.0, "F": 0.0}

class CurriculumEnv(gym.Env):
    """Enhanced Curriculum environment with better action handling"""

    def __init__(self, student_profile, curriculum_graph):
        super().__init__()
        self.student = deepcopy(student_profile)
        self.graph = curriculum_graph
        self.courses = list(curriculum_graph.nodes())
        self.max_terms = 8
        self.courses_per_term = (3, 5)  # min, max
        
        # Action space: binary vector for each course
        self.action_space = spaces.MultiBinary(len(self.courses))
        
        # Fixed observation space - flattened for compatibility
        obs_size = (
            len(self.courses) * 4 +  # completed, passed, failed, available
            len(INTEREST_CATEGORIES) +  # interests
            3  # term, gpa, graduation_progress
        )
        
        self.observation_space = gym.spaces.Box(
            low=0.0, 
            high=10.0, 
            shape=(140,),  # Must match model's expected shape
            dtype=np.float32
        )
        
        self.reset()

    def reset(self, seed=None, options=None):
        # Initialize student profile with defaults if missing
        self.student = deepcopy(self.student)
        self.student.setdefault("Completed_Courses", {})
        self.student.setdefault("Passed_Courses", [])
        self.student.setdefault("Failed_Courses", [])
        self.student.setdefault("Retaken_Courses", [])
        self.student.setdefault("Current_Term", 1)
        self.student.setdefault("GPA", 0.0)
        self.student.setdefault("Academic_Standing", "Critical")
        self.student.setdefault("Graduation_Progress", 0.0)
        self.student.setdefault("Total_Courses_Passed", 0)
        self.student.setdefault("Total_Courses_Failed", 0)
        self.student.setdefault("Available_Next_Courses", [])
        
        # Ensure derived fields are consistent
        self._update_derived_fields()
        
        return self._get_obs(), {}

    def _update_derived_fields(self):
        """Ensure all derived fields are consistent with completed courses"""
        # Update passed/failed lists
        self.student["Passed_Courses"] = [
            c for c, g in self.student["Completed_Courses"].items() if g != "F"
        ]
        self.student["Failed_Courses"] = [
            c for c, g in self.student["Completed_Courses"].items() if g == "F"
        ]
        
        # Update counts
        self.student["Total_Courses_Passed"] = len(self.student["Passed_Courses"])
        self.student["Total_Courses_Failed"] = len(self.student["Failed_Courses"])
        
        # Update available courses
        self.student["Available_Next_Courses"] = self._get_eligible_courses()
        
        # Update GPA if not set
        if "GPA" not in self.student or self.student["GPA"] == 0.0:
            self.student["GPA"] = self._calculate_gpa()
        
        # Update graduation progress
        total_courses = len(self.courses)
        completed = self.student["Total_Courses_Passed"]
        self.student["Graduation_Progress"] = round((completed / total_courses) * 100, 1)
        
        # Update academic standing
        self.student["Academic_Standing"] = self._determine_academic_standing()

    def _calculate_gpa(self):
        """Calculate GPA based on passed courses"""
        passing_grades = [
            GRADE_POINTS[g] 
            for c, g in self.student["Completed_Courses"].items() 
            if g != "F"
        ]
        return round(sum(passing_grades) / len(passing_grades), 2) if passing_grades else 0.0

    def _determine_academic_standing(self):
        """Determine academic standing based on GPA"""
        gpa = self.student["GPA"]
        if gpa >= 3.5:
            return "Excellent"
        elif gpa >= 3.0:
            return "Good"
        elif gpa >= 2.5:
            return "Satisfactory"
        elif gpa >= 2.0:
            return "Probation"
        else:
            return "Critical"

    def _get_obs(self):
        """Convert all observations into a flat vector"""
        # Example implementation - adjust to match your model's expected format
        obs = np.zeros(140, dtype=np.float32)
        
        # Populate the observation vector
        # [0-33]: Course completion status (1=completed)
        for i, course in enumerate(self.courses):
            if course in self.student["Completed_Courses"]:
                obs[i] = 1.0
        
        # [34-67]: Course availability (1=available)
        available = self._get_eligible_courses()
        for i, course in enumerate(self.courses):
            if course in available:
                obs[i+34] = 1.0
        
        # [68]: GPA (normalized to 0-1 range)
        obs[68] = self.student["GPA"] / 4.0
        
        # [69]: Current term (normalized)
        obs[69] = self.student["Current_Term"] / self.max_terms
        
        # [70-139]: Other features (interests, etc.)
        for i, interest in enumerate(INTEREST_CATEGORIES):
            obs[70+i] = 1.0 if interest in self.student["Interests"] else 0.0
        
        return obs

    def _get_eligible_courses(self):
        """Get courses that can be taken next"""
        eligible = []
        completed = set(self.student["Passed_Courses"])
        
        for course in self.courses:
            # Skip if already passed
            if course in completed:
                continue
                
            # Check prerequisites
            prereqs = list(self.graph.predecessors(course))
            if all(p in completed for p in prereqs):
                eligible.append(course)
        
        return eligible

    def step(self, action):
        """Execute one term"""
        selected = [c for c, a in zip(self.courses, action) if a == 1]
        eligible = self.student["Available_Next_Courses"]
        
        # Validate action
        if len(selected) < self.courses_per_term[0]:
            return self._get_obs(), -10, True, False, {"error": "Too few courses selected"}
        if len(selected) > self.courses_per_term[1]:
            return self._get_obs(), -10, True, False, {"error": "Too many courses selected"}
        if any(c not in eligible for c in selected):
            return self._get_obs(), -10, True, False, {"error": "Ineligible course selected"}

        # Simulate grades for selected courses
        for course in selected:
            grade = self._simulate_grade(course)
            self.student["Completed_Courses"][course] = grade
            
            # Track retakes
            if course in self.student["Failed_Courses"]:
                self.student["Retaken_Courses"].append(course)
        
        # Update all derived fields
        self._update_derived_fields()
        
        # Advance term
        self.student["Current_Term"] += 1
        
        # Calculate reward
        reward = self._calculate_reward(selected)
        
        # Check termination
        done = (
            self.student["Graduation_Progress"] >= 100 or
            self.student["Current_Term"] > self.max_terms
        )
        
        info = {
        'selected_courses': selected,  # List of course names
        'current_term': self.student["Current_Term"],
        'gpa': self.student["GPA"],
        "graduation_progress": self.student["Graduation_Progress"],
        'passed': len(self.student["Passed_Courses"]),
        'failed': len(self.student["Failed_Courses"])
    }

        return self._get_obs(), reward, done, False, info

    def _simulate_grade(self, course):
        """Simulate grade for a course based on student performance"""
        base_success = 0.7 + 0.1 * self.student["GPA"]  # Base 70% + GPA bonus
        
        # Retake bonus
        if course in self.student["Failed_Courses"]:
            base_success += 0.15
            
        # Interest bonus
        if self.graph.nodes[course]['category'] in self.student["Interests"]:
            base_success += 0.1
            
        base_success = min(base_success, 0.95)  # Cap at 95%
        
        if random.random() < base_success:
            # Passed - generate grade based on GPA
            if self.student["GPA"] >= 3.5:
                return random.choices(["A", "B", "C"], [0.6, 0.3, 0.1])[0]
            elif self.student["GPA"] >= 3.0:
                return random.choices(["A", "B", "C"], [0.4, 0.4, 0.2])[0]
            elif self.student["GPA"] >= 2.5:
                return random.choices(["A", "B", "C", "D"], [0.2, 0.4, 0.3, 0.1])[0]
            else:
                return random.choices(["B", "C", "D"], [0.2, 0.5, 0.3])[0]
        else:
            return "F"

    def _calculate_reward(self, selected):
        """Calculate reward based on student performance"""
        reward = 0
        
        # GPA reward (normalized)
        reward += (self.student["GPA"] - 2.0) / 2.0
        
        # Progress reward
        reward += len(self.student["Passed_Courses"]) / len(self.courses)
        
        # Penalty for failing courses
        failed_this_term = sum(1 for c in selected if self.student["Completed_Courses"][c] == "F")
        reward -= failed_this_term * 0.5
        
        # Graduation bonus
        if self.student["Graduation_Progress"] >= 100:
            reward += 10
            
        return reward

    def get_valid_action(self, num_courses=4):
        """Generate a valid action for testing"""
        eligible = self.student["Available_Next_Courses"]
        
        if len(eligible) < num_courses:
            num_courses = len(eligible)
        
        # Select random courses from eligible ones
        selected = random.sample(eligible, min(num_courses, len(eligible)))
        
        # Create action vector
        action = np.zeros(len(self.courses), dtype=np.int8)
        for course in selected:
            if course in self.courses:
                action[self.courses.index(course)] = 1
        
        return action

class CurriculumRecommendationSystem:
    """Main recommendation system with enhanced features"""
    
    def __init__(self, curriculum_graph):
        self.graph = curriculum_graph
        self.courses = list(curriculum_graph.nodes())
        
    def recommend_courses(self, student_profile, num_courses=4):
        """Recommend courses based on student profile"""
        env = CurriculumEnv(student_profile, self.graph)
        obs, _ = env.reset()
        
        # Get eligible courses
        eligible = env.student["Available_Next_Courses"]
        
        if len(eligible) < num_courses:
            num_courses = len(eligible)
        
        # Score courses based on various factors
        scored_courses = []
        
        for course in eligible:
            score = self._calculate_course_score(course, student_profile)
            scored_courses.append((course, score))
        
        # Sort by score and select top courses
        scored_courses.sort(key=lambda x: x[1], reverse=True)
        recommendations = [course for course, score in scored_courses[:num_courses]]
        
        return recommendations, scored_courses
    
    def _calculate_course_score(self, course, student_profile):
        """Calculate score for a course based on student profile"""
        score = 0
        
        # Interest alignment
        if self.graph.nodes[course]['category'] in student_profile["Interests"]:
            score += 10
        
        # Retake priority
        if course in student_profile.get("Failed_Courses", []):
            score += 15
        
        # GPA consideration - easier courses for struggling students
        if student_profile.get("GPA", 0) < 2.5:
            # Prioritize core courses
            if self.graph.nodes[course]['category'] == "Core":
                score += 5
        
        # Random factor to avoid deterministic recommendations
        score += random.uniform(0, 3)
        
        return score

def test_environment_comprehensive():
    """Comprehensive test of the environment"""
    print("=== Comprehensive Environment Test ===")
    
    # Create sample student profile
    sample_student = {
        "ID": "STU001",
        "Interests": ["AI", "Elective", "Security"],
        "Completed_Courses": {"CS101": "F", "CS105": "B"},
        "Passed_Courses": ["CS105"],
        "Failed_Courses": ["CS101"],
        "Retaken_Courses": [],
        "Current_Term": 2,
        "GPA": 3.0,
        "Academic_Standing": "Good",
        "Graduation_Progress": 3.0,
        "Total_Courses_Passed": 1,
        "Total_Courses_Failed": 1,
        "Available_Next_Courses": ["CS101", "CS102", "CS107"]
    }
    
    # Create a simple curriculum graph for testing
    curriculum = nx.DiGraph()
    courses = [
        ("CS101", {"category": "Core"}),
        ("CS102", {"category": "Core"}),
        ("CS103", {"category": "Core"}),
        ("CS105", {"category": "Core"}),
        ("CS107", {"category": "Elective"}),
        ("CS201", {"category": "AI"}),
        ("CS202", {"category": "Security"}),
        ("CS203", {"category": "Data Science"}),
    ]
    
    curriculum.add_nodes_from(courses)
    curriculum.add_edges_from([
        ("CS101", "CS201"),
        ("CS102", "CS202"),
        ("CS103", "CS203"),
        ("CS105", "CS107"),
    ])
    
    # Test environment
    env = CurriculumEnv(sample_student, curriculum)
    obs, _ = env.reset()
    
    print(f"Initial state:")
    print(f"  Available courses: {env.student['Available_Next_Courses']}")
    print(f"  Current term: {env.student['Current_Term']}")
    print(f"  GPA: {env.student['GPA']}")
    print(f"  Progress: {env.student['Graduation_Progress']}%")
    
    # Test valid action
    print("\n=== Testing Valid Action ===")
    valid_action = env.get_valid_action(4)
    selected_courses = [c for c, a in zip(env.courses, valid_action) if a == 1]
    print(f"Selected courses: {selected_courses}")
    
    obs, reward, done, truncated, info = env.step(valid_action)
    print(f"Reward: {reward}")
    print(f"Info: {info}")
    print(f"Done: {done}")
    print(f"New GPA: {env.student['GPA']}")
    print(f"New Progress: {env.student['Graduation_Progress']}%")
    
    # Test recommendation system
    print("\n=== Testing Recommendation System ===")
    rec_system = CurriculumRecommendationSystem(curriculum)
    recommendations, scored_courses = rec_system.recommend_courses(sample_student, 3)
    
    print(f"Recommendations: {recommendations}")
    print("All scored courses:")
    for course, score in scored_courses:
        print(f"  {course}: {score:.2f}")
    
    return True


if __name__ == "__main__":
    # Run comprehensive test
    test_environment_comprehensive()
    
    # Additional test with mock curriculum
    print("\n" + "="*50)
    print("=== Testing with Mock Curriculum ===")
    
    try:
        with open("AI-Powered-Academic-Advisor/data/curriculum.pkl", "rb") as f:
            curriculum = pickle.load(f)    
        
        # Test student with more diverse background
        advanced_student = {
            "ID": "STU001",
            "Interests": ["AI", "Elective", "Security"],
            "completed_courses": {
                "CS101": "F",  # Failed, can retake
                "CS105": "B"   # Passed
            },
            "current_term": 2,
            "Passed_Courses": [
                "CS105"
            ],
            "Failed_Courses": [
                "CS101"
            ],
            "Retaken_Courses": [],
            "GPA": 3.0,
            "Academic_Standing": "Good",
            "Graduation_Progress": 3.0,
            "Total_Courses_Passed": 1,
            "Total_Courses_Failed": 1,
            "Available_Next_Courses": [
                "CS101",
                "CS102",
                "CS107"
            ]
        }
        
        rec_system = CurriculumRecommendationSystem(curriculum)
        recommendations, scored_courses = rec_system.recommend_courses(advanced_student, 4)
        
        print(f"Advanced student recommendations: {recommendations}")
        print("Course scores:")
        for course, score in scored_courses[:10]:  # Top 10
            course_name = curriculum.nodes[course]['label']
            category = curriculum.nodes[course]['category']
            print(f"  {course} ({course_name}) [{category}]: {score:.2f}")
    
    except FileNotFoundError:
        print("Curriculum file not found. Skipping advanced test.")
    except Exception as e:
        print(f"Error in advanced test: {e}")