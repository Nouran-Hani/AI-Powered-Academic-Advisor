import gymnasium as gym
from gymnasium import spaces
import random
import numpy as np
from copy import deepcopy

interests_list = ["AI", "Security", "Data Science", "Elective"]

class CurriculumEnv(gym.Env):
    def __init__(self, student_profile, curriculum_graph):
        super(CurriculumEnv, self).__init__()
        self.original_profile = deepcopy(student_profile)
        self.graph = curriculum_graph
        self.max_terms = 8  # Extended to allow more realistic graduation timeline
        self.min_courses_per_term = 3
        self.max_courses_per_term = 5
        self.total_courses_needed = len(curriculum_graph.nodes)
        self.reset()

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)
            random.seed(seed)
        
        self.profile = deepcopy(self.original_profile)
        self.profile["Completed_Courses"] = self.profile.get("Completed_Courses", {})
        self.profile["Current_Term"] = self.profile.get("Current_Term", 1)
        self.profile["Failed_Courses"] = self.profile.get("Failed_Courses", [])
        self.profile["Retaken_Courses"] = self.profile.get("Retaken_Courses", [])
        self.done = False
        return self._get_state(), {}

    def _get_state(self):
        state = {
            "Completed": set(self.profile["Completed_Courses"].keys()),
            "GPA": self.profile["GPA"],
            "Term": self.profile["Current_Term"],
            "Interests": self.profile["Interests"],
            "Failed": set(self.profile["Failed_Courses"]),
            "Progress": len(self.profile["Completed_Courses"]) / self.total_courses_needed
        }
        return state

    def _eligible_courses(self):
        """Get courses that can be taken this term"""
        eligible = []
        completed_courses = set(self.profile["Completed_Courses"].keys())
        
        for course in self.graph.nodes:
            # Skip if already completed with passing grade
            if course in completed_courses and self.profile["Completed_Courses"][course] != "F":
                continue
                
            # Check prerequisites
            prereqs = list(self.graph.predecessors(course))
            prereqs_satisfied = all(
                p in self.profile["Completed_Courses"] and 
                self.profile["Completed_Courses"][p] != "F" 
                for p in prereqs
            )
            
            if prereqs_satisfied:
                eligible.append(course)
        
        return eligible

    def _validate_course_load(self, action):
        """Validate course load constraints"""
        if len(action) < self.min_courses_per_term:
            return False, "Too few courses selected"
        if len(action) > self.max_courses_per_term:
            return False, "Too many courses selected"
        return True, "Valid course load"

    def _calculate_graduation_progress(self):
        """Calculate progress toward graduation"""
        completed_passing = sum(1 for grade in self.profile["Completed_Courses"].values() if grade != "F")
        return completed_passing / self.total_courses_needed

    def step(self, action):
        if self.done:
            return self._get_state(), 0, True, False, {}

        # Validate course load
        valid, message = self._validate_course_load(action)
        if not valid:
            # Penalty for invalid course load
            return self._get_state(), -10, True, False, {"error": message}

        # Check if all courses are eligible
        eligible_courses = self._eligible_courses()
        invalid_courses = [c for c in action if c not in eligible_courses]
        if invalid_courses:
            # Penalty for taking ineligible courses
            return self._get_state(), -5, True, False, {"error": f"Ineligible courses: {invalid_courses}"}

        # Simulate course outcomes
        term_gpa = 0
        passed = 0
        failed = 0
        retakes = 0
        
        for course in action:
            # Check if this is a retake
            if course in self.profile["Failed_Courses"]:
                retakes += 1
                # Slightly better odds for retakes (assuming student learned from failure)
                grade = random.choices(["A", "B", "C", "D", "F"], weights=[0.2, 0.35, 0.25, 0.15, 0.05])[0]
                if grade != "F":
                    # Remove from failed courses list
                    self.profile["Failed_Courses"].remove(course)
                    self.profile["Retaken_Courses"].append(course)
            else:
                # Regular course attempt
                grade = random.choices(["A", "B", "C", "D", "F"], weights=[0.3, 0.3, 0.2, 0.1, 0.1])[0]
            
            # Update course record
            self.profile["Completed_Courses"][course] = grade
            
            if grade != "F":
                term_gpa += {"A": 4, "B": 3, "C": 2, "D": 1}[grade]
                passed += 1
            else:
                if course not in self.profile["Failed_Courses"]:
                    self.profile["Failed_Courses"].append(course)
                failed += 1

        # Calculate overall GPA
        all_grades = [g for g in self.profile["Completed_Courses"].values() if g != "F"]
        if all_grades:
            grade_points = [{"A": 4, "B": 3, "C": 2, "D": 1}[g] for g in all_grades]
            self.profile["GPA"] = round(sum(grade_points) / len(grade_points), 2)

        # Increment term
        self.profile["Current_Term"] += 1
        
        # Check for graduation or term limit
        graduation_progress = self._calculate_graduation_progress()
        if graduation_progress >= 1.0:
            self.done = True
            graduation_bonus = 50  # Big bonus for graduating
        elif self.profile["Current_Term"] > self.max_terms:
            self.done = True
            graduation_bonus = -20  # Penalty for not graduating in time
        else:
            graduation_bonus = 0

        # Calculate reward components
        if len(action) > 0:
            # GPA component
            gpa_reward = term_gpa / len(action)
            
            # Interest alignment bonus
            interest_bonus = sum(2 for c in action if self.graph.nodes[c]['category'] in self.profile["Interests"])
            
            # Progress toward graduation
            progress_bonus = graduation_progress * 10
            
            # Retake penalty (encourage not failing)
            retake_penalty = retakes * -1
            
            # Failed course penalty
            fail_penalty = failed * -3
            
            # Total reward
            reward = gpa_reward + interest_bonus + progress_bonus + retake_penalty + fail_penalty + graduation_bonus
        else:
            reward = -10  # Penalty for no courses

        return self._get_state(), reward, self.done, False, {
            "term_gpa": term_gpa / len(action) if len(action) > 0 else 0,
            "passed": passed,
            "failed": failed,
            "retakes": retakes,
            "graduation_progress": graduation_progress,
            "graduated": graduation_progress >= 1.0
        }

class CurriculumMultiStudentEnv(gym.Env):
    def __init__(self, students, curriculum_graph):
        self.students = students
        self.graph = curriculum_graph
        self.max_terms = 8
        self.current_env = None
        self.current_index = -1

        # Set up observation and action spaces
        self.observation_space = spaces.Dict({
            "Completed": spaces.MultiBinary(len(self.graph.nodes)),
            "GPA": spaces.Box(0, 4.0, shape=(1,), dtype=np.float32),
            "Term": spaces.Discrete(9),  # 0-8 terms
            "Interests": spaces.MultiBinary(len(interests_list)),
            "Failed": spaces.MultiBinary(len(self.graph.nodes)),
            "Progress": spaces.Box(0, 1.0, shape=(1,), dtype=np.float32)
        })
        self.action_space = spaces.MultiBinary(len(self.graph.nodes))

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)
            random.seed(seed)
        
        self.current_index = (self.current_index + 1) % len(self.students)
        student = self.students[self.current_index]
        self.current_env = CurriculumEnv(student, self.graph)
        state, _ = self.current_env.reset(seed=seed)
        return self._encode_state(state), {}

    def step(self, action_vector):
        action = self._decode_action(action_vector)
        state, reward, done, truncated, info = self.current_env.step(action)
        return self._encode_state(state), reward, done, truncated, info

    def _encode_state(self, state):
        """Encode state dictionary to gym observation space format"""
        completed_vec = [1 if c in state["Completed"] else 0 for c in self.graph.nodes]
        failed_vec = [1 if c in state["Failed"] else 0 for c in self.graph.nodes]
        interest_vec = [1 if cat in state["Interests"] else 0 for cat in interests_list]
        
        return {
            "Completed": np.array(completed_vec, dtype=np.int8),
            "GPA": np.array([state["GPA"]], dtype=np.float32),
            "Term": state["Term"],
            "Interests": np.array(interest_vec, dtype=np.int8),
            "Failed": np.array(failed_vec, dtype=np.int8),
            "Progress": np.array([state["Progress"]], dtype=np.float32)
        }

    def _decode_action(self, action_vec):
        """Decode binary action vector to list of selected courses"""
        return [c for i, c in enumerate(self.graph.nodes) if action_vec[i] == 1]

    def get_current_student_id(self):
        """Get the ID of the current student being processed"""
        if self.current_index >= 0:
            return self.students[self.current_index].get("ID", self.current_index)
        return None