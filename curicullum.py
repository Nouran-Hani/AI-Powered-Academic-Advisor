import networkx as nx
import matplotlib.pyplot as plt
import pickle
from matplotlib.lines import Line2D
import pydot

curi = nx.DiGraph()

def add_course(ID, course_name, cat, prerequisites=None):
    curi.add_node(ID, label=course_name, category=cat)
    if prerequisites:
        for prereq in prerequisites:
            curi.add_edge(prereq, ID)

courses = [
    # Core Courses (7 courses, no prerequisites)
    ("CS101", "Intro to CS", "Core", []),
    ("CS102", "Programming I", "Core", []),
    ("CS103", "Programming II", "Core", ["CS102"]),
    ("CS104", "Data Structures", "Core", ["CS103"]),
    ("CS105", "Algorithms", "Core", []),
    ("CS106", "Computer Architecture", "Core", ["CS101"]),
    ("CS107", "Operating Systems", "Core", []),
    # AI Track
    ("AI201", "Intro to AI", "AI", ["CS104", "CS105"]),
    ("AI202", "Machine Learning", "AI", ["AI201"]),
    ("AI203", "Neural Networks", "AI", ["AI202"]),

    # Data Science Track
    ("DS201", "Intro to Data Science", "Data Science", ["CS104"]),
    ("DS202", "Statistics for DS", "Data Science", ["DS201"]),
    ("DS203", "Data Mining", "Data Science", ["DS202"]),

    # Security Track
    ("SEC201", "Intro to Cybersecurity", "Security", ["CS104", "CS106"]),
    ("SEC202", "Network Security", "Security", ["SEC201"]),
    ("SEC203", "Cryptography", "Security", ["SEC202"]),

    # Electives
    ("EL301", "Mobile App Development", "Elective", ["CS103"]),
    ("EL302", "Web Development", "Elective", ["CS103"]),
    ("EL303", "Cloud Computing", "Elective", ["CS104", "CS107"]),
    ("EL304", "Computer Graphics", "Elective", ["CS104"]),
    ("EL305", "Parallel Computing", "Elective", ["CS105", "CS107"]),
]


# Build the curriculum graph
# for course in courses:
#     add_course(*course)

def visualize_curriculum(graph):
    category_colors = {
        "Core": "#87CEEB",
        "AI": "#FFB347",
        "Data Science": "#90EE90",
        "Security": "#FF9999",
        "Elective": "#DDA0DD",
    }

    # Compute levels (prerequisite depth)
    levels = {}
    def dfs(node, level=0):
        if node not in levels or level > levels[node]:
            levels[node] = level
        for succ in graph.successors(node):
            dfs(succ, level + 1)

    # Start DFS from root nodes (no prerequisites)
    roots = [n for n in graph.nodes if graph.in_degree(n) == 0]
    for root in roots:
        dfs(root)

    # Group nodes by level
    level_nodes = {}
    for node, lvl in levels.items():
        level_nodes.setdefault(lvl, []).append(node)

    # Assign positions
    pos = {}
    for lvl, nodes in level_nodes.items():
        n = len(nodes)
        for i, node in enumerate(sorted(nodes)):
            pos[node] = (i - n / 2, -lvl)  # x = horizontal spread, y = level

    # Draw
    plt.figure(figsize=(12, 6))
    node_colors = [category_colors.get(graph.nodes[n]['category'], 'gray') for n in graph.nodes()]
    nx.draw_networkx_nodes(graph, pos, node_color=node_colors, node_size=1000)
    nx.draw_networkx_edges(graph, pos, arrows=True, arrowsize=25)

    # Labels (only IDs)
    nx.draw_networkx_labels(graph, pos, {n: n for n in graph.nodes()}, font_size=7)

    # Legend
    legend = [Line2D([0], [0], marker='o', color='w', label=cat,
                     markerfacecolor=color, markersize=10)
              for cat, color in category_colors.items()]
    plt.legend(handles=legend, loc='lower left')

    plt.title("Curriculum Prerequisite Graph (Hierarchical Layout)", fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig("AI-Powered-Academic-Advisor/curriculum_graph.png", dpi=300)
    plt.show()



# visualize_curriculum(curi)


def save_curriculum(filename='AI-Powered-Academic-Advisor/curriculum.pkl'):
    with open(filename, "wb") as f:
        pickle.dump(curi, f)


# Save the curriculum graph to a file
# save_curriculum()