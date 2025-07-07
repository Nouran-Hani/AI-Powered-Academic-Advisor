
# Curriculum-Based Academic Advisor using Reinforcement Learning

This project simulates an AI-powered academic advising system that guides students through a structured university curriculum using **Reinforcement Learning (RL)**. The system models academic progress and course selection using a graph-based curriculum and a custom Gymnasium environment.

---

## Project Overview

- **Curriculum Graph**: Modeled as a Directed Acyclic Graph (DAG) using `networkx`, where each node is a course and each edge is a prerequisite.
- **Course Metadata**: Each course has an `id`, `label`, and belongs to one of the following categories: `Core`, `AI`, `Data Science`, `Security`, or `Elective`.
- **Synthetic Student Profiles**:
  - 100 students with varying GPAs, interests, and academic progress.
  - Attributes include interests, completed courses, GPA, academic standing, and available next courses.
- **Academic Simulation**:
  - Simulates student journeys over 1–8 academic terms.
  - Course selection each term is based on performance, interests, and prerequisite constraints.
  - Grade assignment is probabilistic; failures may lead to course retakes.
- **RL-based Advising**:
  - A custom `CurriculumEnv` built with Gymnasium.
  - Uses `PPO` (Proximal Policy Optimization) from `stable-baselines3`.
  - Rewards are based on GPA improvements, graduation progress, and penalty for failures.

---

## Project Features

- Prerequisite-aware course planning
- Personalized, interest-driven recommendations
- Dynamic simulation of academic progression
- Retake support for failed courses
- Reinforcement Learning-driven policy training (PPO)

---

## Project Structure

```bash
.
├── graph_schema.py         # Curriculum DAG creation and visualization
├── student_generator.py    # Generates synthetic student profiles
├── curriculum_env.py       # Custom Gym environment for simulation
├── train_agent.py          # Trains the RL agent (PPO)
├── evaluate.py             # Evaluates the agent's performance
├── requirements.txt        # Project dependencies
└── README.md               # Project documentation
```
---

## Setup & Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name
   ```

2. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

---

## Run the Project

1. **Train the RL agent**:

   ```bash
   python train_agent.py
   ```

2. **Evaluate the agent**:

   ```bash
   python evaluate.py
   ```

---

## requirements.txt
   ```txt
matplotlib
networkx
numpy
gymnasium
stable-baselines3
```
