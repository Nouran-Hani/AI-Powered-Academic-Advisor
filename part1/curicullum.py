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
    ("AI204", "Deep Learning", "AI", ["AI203"]),
    ("AI205", "Computer Vision", "AI", ["AI202"]),
    ("AI206", "Natural Language Processing", "AI", ["AI202"]),

    # Data Science Track
    ("DS201", "Intro to Data Science", "Data Science", ["CS104"]),
    ("DS202", "Statistics for DS", "Data Science", ["DS201"]),
    ("DS203", "Data Mining", "Data Science", ["DS202"]),
    ("DS204", "Big Data Analytics", "Data Science", ["DS203"]),
    ("DS205", "Data Visualization", "Data Science", ["DS201"]),
    ("DS206", "Database Systems", "Data Science", ["CS104"]),

    # Security Track
    ("SEC201", "Intro to Cybersecurity", "Security", ["CS104", "CS106"]),
    ("SEC202", "Network Security", "Security", ["SEC201"]),
    ("SEC203", "Cryptography", "Security", ["SEC202"]),
    ("SEC204", "Ethical Hacking", "Security", ["SEC201"]),
    ("SEC205", "Digital Forensics", "Security", ["SEC202"]),
    ("SEC206", "Security Policy", "Security", ["SEC201"]),

    # Electives
    ("EL301", "Mobile App Development", "Elective", ["CS103"]),
    ("EL302", "Web Development", "Elective", ["CS103"]),
    ("EL303", "Cloud Computing", "Elective", ["CS104", "CS107"]),
    ("EL304", "Computer Graphics", "Elective", ["CS104"]),
    ("EL305", "Parallel Computing", "Elective", ["CS105", "CS107"]),
    ("EL306", "Software Engineering", "Elective", ["CS104"]),
    ("EL307", "Human-Computer Interaction", "Elective", ["CS101"]),
    ("EL308", "Game Development", "Elective", ["CS104"]),
]

# Build the curriculum graph
for course in courses:
    add_course(*course)

def get_graph_schema():
    """Generate graph schema information"""
    schema = {
        "total_courses": len(curi.nodes),
        "total_prerequisites": len(curi.edges),
        "categories": {},
        "course_levels": {},
        "sample_cypher_queries": []
    }
    
    # Analyze categories
    for node in curi.nodes:
        category = curi.nodes[node]['category']
        if category not in schema["categories"]:
            schema["categories"][category] = []
        schema["categories"][category].append(node)
    
    # Calculate course levels (prerequisite depth)
    levels = {}
    def calculate_depth(node, visited=None):
        if visited is None:
            visited = set()
        if node in visited:
            return 0  # Cycle detection
        visited.add(node)
        
        predecessors = list(curi.predecessors(node))
        if not predecessors:
            return 0
        return 1 + max(calculate_depth(pred, visited.copy()) for pred in predecessors)
    
    for node in curi.nodes:
        levels[node] = calculate_depth(node)
    
    schema["course_levels"] = levels
    
    # Sample Cypher queries (for Neo4j compatibility)
    schema["sample_cypher_queries"] = [
        "MATCH (c:Course) RETURN c.id, c.name, c.category",
        "MATCH (c1:Course)-[:PREREQUISITE]->(c2:Course) RETURN c1.id, c2.id",
        "MATCH (c:Course {category: 'AI'}) RETURN c.id, c.name",
        "MATCH (c:Course) WHERE c.category IN ['AI', 'Data Science'] RETURN c.id, c.name",
        "MATCH path = (start:Course)-[:PREREQUISITE*]->(end:Course) WHERE start.id = 'CS101' RETURN path"
    ]
    
    return schema

def visualize_curriculum(graph, save_path="AI-Powered-Academic-Advisor/part1/data/curriculum_graph.png"):
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
    plt.figure(figsize=(16, 10))
    node_colors = [category_colors.get(graph.nodes[n]['category'], 'gray') for n in graph.nodes()]
    nx.draw_networkx_nodes(graph, pos, node_color=node_colors, node_size=1200)
    nx.draw_networkx_edges(graph, pos, arrows=True, arrowsize=20, alpha=0.6)

    # Labels (only IDs)
    nx.draw_networkx_labels(graph, pos, {n: n for n in graph.nodes()}, font_size=8)

    # Legend
    legend = [Line2D([0], [0], marker='o', color='w', label=cat,
                     markerfacecolor=color, markersize=10)
              for cat, color in category_colors.items()]
    plt.legend(handles=legend, loc='upper right')

    plt.title("Curriculum Prerequisite Graph (Hierarchical Layout)", fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def print_graph_statistics():
    """Print detailed graph statistics"""
    print("="*60)
    print("CURRICULUM GRAPH STATISTICS")
    print("="*60)
    
    schema = get_graph_schema()
    
    print(f"Total Courses: {schema['total_courses']}")
    print(f"Total Prerequisites: {schema['total_prerequisites']}")
    print()
    
    print("Courses by Category:")
    for category, courses in schema['categories'].items():
        print(f"  {category}: {len(courses)} courses")
        for course in courses[:3]:  # Show first 3
            print(f"    - {course}")
        if len(courses) > 3:
            print(f"    ... and {len(courses) - 3} more")
    print()
    
    print("Course Levels (Prerequisite Depth):")
    level_counts = {}
    for course, level in schema['course_levels'].items():
        level_counts[level] = level_counts.get(level, 0) + 1
    
    for level in sorted(level_counts.keys()):
        print(f"  Level {level}: {level_counts[level]} courses")
    print()
    
    print("Sample Neo4j Cypher Queries:")
    for i, query in enumerate(schema['sample_cypher_queries'], 1):
        print(f"  {i}. {query}")
    print()

def save_curriculum(filename='AI-Powered-Academic-Advisor/part1/data/curriculum.pkl'):
    """Save curriculum graph to file"""
    with open(filename, "wb") as f:
        pickle.dump(curi, f)
    print(f"Curriculum graph saved to {filename}")

def export_to_neo4j_format(filename='AI-Powered-Academic-Advisor/part1/data/curriculum_neo4j.txt'):
    """Export curriculum in Neo4j import format"""
    with open(filename, 'w') as f:
        f.write("// Neo4j Cypher commands to create curriculum graph\n\n")
        
        # Create nodes
        f.write("// Create Course nodes\n")
        for node in curi.nodes:
            data = curi.nodes[node]
            f.write(f"CREATE (c_{node}:Course {{id: '{node}', name: '{data['label']}', category: '{data['category']}'}})\n")
        
        f.write("\n// Create prerequisite relationships\n")
        for edge in curi.edges:
            f.write(f"CREATE (c_{edge[0]})-[:PREREQUISITE]->(c_{edge[1]})\n")
    
    print(f"Neo4j format exported to {filename}")

if __name__ == "__main__":
    # Print statistics
    print_graph_statistics()
    
    # Visualize curriculum
    visualize_curriculum(curi)
    
    # Save curriculum
    save_curriculum()
    
    # Export to Neo4j format
    export_to_neo4j_format()
    
    print("\nCurriculum setup complete!")