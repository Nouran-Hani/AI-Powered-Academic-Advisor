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
    """
    Reinforcement Learning Environment for Course Recommendation
    
    State: current completed courses, GPA, term number, interests, failed courses
    Action: selecting a set of next-term eligible courses
    Reward: GPA improvement, interest alignment, progress toward graduation
    """
    
    def __init__(self, student_profile, curriculum_graph):
        super(CurriculumEnv, self).__init__()
        self.original_profile = deepcopy(student_profile)
        self.graph = curriculum_graph
        self.course_list = list(curriculum_graph.nodes())
        self.max_terms = 12  # Allow up to 12 terms for graduation
        self.min_courses_per_term = 3
        self.max_courses_per_term = 5
        self.total_courses_needed = len(curriculum_graph.nodes)
        
        # Define action and observation spaces
        self.action_space = spaces.MultiBinary(len(self.course_list))
        self.observation_space = spaces.Dict({
            'completed_courses': spaces.MultiBinary(len(self.course_list)),
            'failed_courses': spaces.MultiBinary(len(self.course_list)),
            'current_gpa': spaces.Box(low=0.0, high=4.0, shape=(1,), dtype=np.float32),
            'current_term': spaces.Box(low=1, high=self.max_terms, shape=(1,), dtype=np.int32),
            'interests': spaces.MultiBinary(len(INTEREST_CATEGORIES)),
            'graduation_progress': spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            'eligible_courses': spaces.MultiBinary(len(self.course_list))
        })
        
        self.reset()

    def reset(self, seed=None, options=None):
        """Reset environment to initial state"""
        if seed is not None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)
            random.seed(seed)
        
        # Reset student profile
        self.profile = deepcopy(self.original_profile)
        self.profile["Completed_Courses"] = self.profile.get("Completed_Courses", {})
        self.profile["Failed_Courses"] = self.profile.get("Failed_Courses", [])
        self.profile["Current_Term"] = self.profile.get("Current_Term", 1)
        self.profile["GPA"] = self.profile.get("GPA", 0.0)
        self.profile["Interests"] = self.profile.get("Interests", [])
        
        self.done = False
        self.terms_completed = 0
        
        return self._get_observation(), {}

    def _get_observation(self):
        """Get current state observation"""
        # Completed courses binary vector
        completed_vec = np.zeros(len(self.course_list), dtype=np.int8)
        for i, course in enumerate(self.course_list):
            if course in self.profile["Completed_Courses"] and self.profile["Completed_Courses"][course] != "F":
                completed_vec[i] = 1
        
        # Failed courses binary vector
        failed_vec = np.zeros(len(self.course_list), dtype=np.int8)
        for i, course in enumerate(self.course_list):
            if course in self.profile["Failed_Courses"]:
                failed_vec[i] = 1
        
        # Interest vector
        interest_vec = np.zeros(len(INTEREST_CATEGORIES), dtype=np.int8)
        for i, category in enumerate(INTEREST_CATEGORIES):
            if category in self.profile["Interests"]:
                interest_vec[i] = 1
        
        # Eligible courses
        eligible_courses = self._get_eligible_courses()
        eligible_vec = np.zeros(len(self.course_list), dtype=np.int8)
        for i, course in enumerate(self.course_list):
            if course in eligible_courses:
                eligible_vec[i] = 1
        
        # Graduation progress
        completed_count = len([c for c, g in self.profile["Completed_Courses"].items() if g != "F"])
        progress = completed_count / self.total_courses_needed
        
        return {
            'completed_courses': completed_vec,
            'failed_courses': failed_vec,
            'current_gpa': np.array([self.profile["GPA"]], dtype=np.float32),
            'current_term': np.array([self.profile["Current_Term"]], dtype=np.int32),
            'interests': interest_vec,
            'graduation_progress': np.array([progress], dtype=np.float32),
            'eligible_courses': eligible_vec
        }

    def _get_eligible_courses(self):
        """Get courses that can be taken this term"""
        eligible = []
        completed_courses = set([c for c, g in self.profile["Completed_Courses"].items() if g != "F"])
        
        for course in self.course_list:
            # Skip if already completed with passing grade
            if course in completed_courses:
                continue
                
            # Check prerequisites
            prereqs = list(self.graph.predecessors(course))
            prereqs_satisfied = all(p in completed_courses for p in prereqs)
            
            if prereqs_satisfied:
                eligible.append(course)
        
        return eligible

    def _validate_action(self, action):
        """Validate the selected course action"""
        selected_courses = [self.course_list[i] for i in range(len(action)) if action[i] == 1]
        
        # Check course load constraints
        if len(selected_courses) < self.min_courses_per_term:
            return False, "Too few courses selected"
        if len(selected_courses) > self.max_courses_per_term:
            return False, "Too many courses selected"
        
        # Check if all selected courses are eligible
        eligible_courses = self._get_eligible_courses()
        for course in selected_courses:
            if course not in eligible_courses:
                return False, f"Course {course} is not eligible"
        
        return True, "Valid action"

    def _simulate_course_outcomes(self, selected_courses):
        """Simulate outcomes for selected courses"""
        grades = []
        passed_courses = []
        failed_courses = []
        
        for course in selected_courses:
            # Determine success probability based on student performance and course difficulty
            base_success_rate = 0.8
            
            # Adjust based on GPA
            if self.profile["GPA"] >= 3.5:
                success_rate = 0.95
            elif self.profile["GPA"] >= 3.0:
                success_rate = 0.90
            elif self.profile["GPA"] >= 2.5:
                success_rate = 0.85
            elif self.profile["GPA"] >= 2.0:
                success_rate = 0.75
            else:
                success_rate = 0.65
            
            # Bonus for retaking failed courses
            if course in self.profile["Failed_Courses"]:
                success_rate += 0.15
            
            # Bonus for interest alignment
            course_category = self.graph.nodes[course]['category']
            if course_category in self.profile["Interests"]:
                success_rate += 0.1
            
            success_rate = min(success_rate, 0.98)  # Cap at 98%
            
            # Generate grade
            if random.random() < success_rate:
                # Passed - generate grade based on performance level
                if self.profile["GPA"] >= 3.5:
                    grade = random.choices(["A", "B", "C"], weights=[0.6, 0.3, 0.1])[0]
                elif self.profile["GPA"] >= 3.0:
                    grade = random.choices(["A", "B", "C"], weights=[0.4, 0.4, 0.2])[0]
                elif self.profile["GPA"] >= 2.5:
                    grade = random.choices(["A", "B", "C", "D"], weights=[0.2, 0.4, 0.3, 0.1])[0]
                else:
                    grade = random.choices(["B", "C", "D"], weights=[0.2, 0.5, 0.3])[0]
                
                passed_courses.append(course)
            else:
                grade = "F"
                failed_courses.append(course)
            
            grades.append(grade)
            self.profile["Completed_Courses"][course] = grade
        
        return grades, passed_courses, failed_courses

    def _calculate_reward(self, selected_courses, grades, passed_courses, failed_courses):
        """Calculate reward based on outcomes"""
        reward = 0.0
        
        # GPA Component
        term_gpa = np.mean([GRADE_POINTS[g] for g in grades])
        reward += term_gpa * 5  # Scale GPA contribution
        
        # Interest Alignment Bonus
        for course in selected_courses:
            course_category = self.graph.nodes[course]['category']
            if course_category in self.profile["Interests"]:
                reward += 3  # Bonus for taking courses aligned with interests
        
        # Progress toward graduation
        completed_count = len([c for c, g in self.profile["Completed_Courses"].items() if g != "F"])
        progress = completed_count / self.total_courses_needed
        reward += progress * 10  # Progress bonus
        
        # Penalty for failing courses
        reward -= len(failed_courses) * 5
        
        # Bonus for retaking and passing previously failed courses
        for course in passed_courses:
            if course in self.profile["Failed_Courses"]:
                reward += 5  # Redemption bonus
                self.profile["Failed_Courses"].remove(course)
        
        # Add new failed courses to the list
        for course in failed_courses:
            if course not in self.profile["Failed_Courses"]:
                self.profile["Failed_Courses"].append(course)
        
        # Graduation bonus
        if progress >= 1.0:
            reward += 50  # Large bonus for graduating
        
        return reward

    def _update_student_profile(self, grades):
        """Update student profile after completing a term"""
        # Update GPA
        all_passing_grades = [g for g in self.profile["Completed_Courses"].values() if g != "F"]
        if all_passing_grades:
            total_points = sum(GRADE_POINTS[g] for g in all_passing_grades)
            self.profile["GPA"] = round(total_points / len(all_passing_grades), 2)
        
        # Update term
        self.profile["Current_Term"] += 1
        self.terms_completed += 1

    def step(self, action):
        """Execute one step in the environment"""
        if self.done:
            return self._get_observation(), 0, True, False, {}
        
        # Validate action
        valid, message = self._validate_action(action)
        if not valid:
            return self._get_observation(), -20, True, False, {"error": message}
        
        # Get selected courses
        selected_courses = [self.course_list[i] for i in range(len(action)) if action[i] == 1]
        
        # Simulate course outcomes
        grades, passed_courses, failed_courses = self._simulate_course_outcomes(selected_courses)
        
        # Calculate reward
        reward = self._calculate_reward(selected_courses, grades, passed_courses, failed_courses)
        
        # Update student profile
        self._update_student_profile(grades)
        
        # Check termination conditions
        completed_count = len([c for c, g in self.profile["Completed_Courses"].items() if g != "F"])
        progress = completed_count / self.total_courses_needed
        
        if progress >= 1.0:
            self.done = True
            reward += 100  # Large graduation bonus
        elif self.profile["Current_Term"] > self.max_terms:
            self.done = True
            reward -= 50  # Penalty for not graduating in time
        
        # Prepare info
        info = {
            "selected_courses": selected_courses,
            "grades": grades,
            "passed": len(passed_courses),
            "failed": len(failed_courses),
            "graduation_progress": progress,
            "graduated": progress >= 1.0,
            "current_gpa": self.profile["GPA"],
            "current_term": self.profile["Current_Term"]
        }
        
        return self._get_observation(), reward, self.done, False, info


class CurriculumMultiStudentEnv(gym.Env):
    """Multi-student environment for training on multiple student profiles"""
    
    def __init__(self, students, curriculum_graph):
        super(CurriculumMultiStudentEnv, self).__init__()
        self.students = students
        self.graph = curriculum_graph
        self.current_student_idx = 0
        
        # Create a single student environment for reference
        sample_env = CurriculumEnv(students[0], curriculum_graph)
        self.action_space = sample_env.action_space
        self.observation_space = sample_env.observation_space
        
        self.current_env = None
        self.reset()

    def reset(self, seed=None, options=None):
        """Reset to next student or cycle back to first student"""
        if seed is not None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)
            random.seed(seed)
        
        # Move to next student
        self.current_student_idx = (self.current_student_idx + 1) % len(self.students)
        student = self.students[self.current_student_idx]
        
        # Create new environment for this student
        self.current_env = CurriculumEnv(student, self.graph)
        
        return self.current_env.reset(seed=seed)

    def step(self, action):
        """Execute step in current student environment"""
        return self.current_env.step(action)

    def get_current_student_info(self):
        """Get information about current student"""
        if self.current_env:
            return {
                "student_id": self.students[self.current_student_idx].get("ID", f"Student_{self.current_student_idx}"),
                "student_index": self.current_student_idx,
                "student_profile": self.students[self.current_student_idx]
            }
        return None


def create_heuristic_recommender(curriculum_graph):
    """
    Create a heuristic-based course recommender as baseline
    """
    
    def recommend_courses(student_profile, max_courses=4):
        """
        Heuristic recommendation algorithm
        """
        # Get eligible courses
        completed_courses = set([c for c, g in student_profile.get("Completed_Courses", {}).items() if g != "F"])
        eligible_courses = []
        
        for course in curriculum_graph.nodes():
            if course in completed_courses:
                continue
            
            prereqs = list(curriculum_graph.predecessors(course))
            if all(p in completed_courses for p in prereqs):
                eligible_courses.append(course)
        
        if not eligible_courses:
            return []
        
        # Score courses based on multiple criteria
        course_scores = []
        student_interests = student_profile.get("Interests", [])
        
        for course in eligible_courses:
            score = 0
            course_category = curriculum_graph.nodes[course]['category']
            
            # Interest alignment (highest priority)
            if course_category in student_interests:
                score += 10
            
            # Core course priority (essential for graduation)
            if course_category == "Core":
                score += 8
            
            # Prerequisite for many courses (enables future options)
            successors = list(curriculum_graph.successors(course))
            score += len(successors) * 2
            
            # Failed course redemption
            if course in student_profile.get("Failed_Courses", []):
                score += 5
            
            course_scores.append((course, score))
        
        # Sort by score and return top courses
        course_scores.sort(key=lambda x: x[1], reverse=True)
        recommended = [course for course, score in course_scores[:max_courses]]
        
        return recommended
    
    return recommend_courses


# Example usage and testing functions
def test_environment_setup():
    """Test the environment setup with sample data"""
    print("Testing environment setup...")
    
    # Create sample student profile
    sample_student = {
        "ID": "TEST001",
        "Interests": ["AI", "Data Science"],
        "Completed_Courses": {"CS101": "A", "CS102": "B"},
        "Failed_Courses": [],
        "Current_Term": 2,
        "GPA": 3.5
    }
    
    # Create simple curriculum graph
    sample_graph = nx.DiGraph()
    sample_graph.add_node("CS101", category="Core")
    sample_graph.add_node("CS102", category="Core")
    sample_graph.add_node("CS103", category="Core")
    sample_graph.add_node("AI201", category="AI")
    sample_graph.add_edge("CS102", "CS103")
    sample_graph.add_edge("CS103", "AI201")
    
    # Test single student environment
    env = CurriculumEnv(sample_student, sample_graph)
    obs, _ = env.reset()
    
    print("Initial observation keys:", obs.keys())
    print("Action space:", env.action_space)
    print("Observation space:", env.observation_space)
    
    # Test action
    action = np.array([0, 0, 1, 0])  # Select CS103
    obs, reward, done, truncated, info = env.step(action)
    
    print("After step - Reward:", reward)
    print("Info:", info)
    print("Done:", done)
    
    return True

if __name__ == "__main__":
    test_environment_setup()