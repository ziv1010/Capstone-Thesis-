import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def draw_pipeline():
    plt.figure(figsize=(14, 8))
    G = nx.DiGraph()
    
    nodes = {
        "Raw Case JSON files": (0, 4),
        "Leakage Filter & Audit": (2, 4),
        "Extract Texts & Entities": (4, 4),
        "Build Local Case Star Graphs": (6, 4),
        "Merge to Global Authority Graph": (8, 4),
        "Compute Features": (10, 4),
        "HeteroData Object": (12, 4),
        "HGT Layers": (12, 2),
        "Case Node Representations": (10, 2),
        "MLP Classifier": (8, 2),
        "Predictions & Metrics": (6, 2)
    }
    
    for node, pos in nodes.items():
        G.add_node(node, pos=pos)
        
    edges = [
        ("Raw Case JSON files", "Leakage Filter & Audit"),
        ("Leakage Filter & Audit", "Extract Texts & Entities"),
        ("Extract Texts & Entities", "Build Local Case Star Graphs"),
        ("Build Local Case Star Graphs", "Merge to Global Authority Graph"),
        ("Merge to Global Authority Graph", "Compute Features"),
        ("Compute Features", "HeteroData Object"),
        ("HeteroData Object", "HGT Layers"),
        ("HGT Layers", "Case Node Representations"),
        ("Case Node Representations", "MLP Classifier"),
        ("MLP Classifier", "Predictions & Metrics")
    ]
    G.add_edges_from(edges)
    
    pos = nx.get_node_attributes(G, 'pos')
    
    # Draw nodes and edges
    nx.draw(G, pos, with_labels=False, node_size=5000, node_color='lightblue', edge_color='gray', 
            arrowsize=20, arrowstyle='-|>', width=2)
            
    # Draw custom labels
    for node, (x, y) in pos.items():
        plt.text(x, y, node, fontsize=10, ha='center', va='center', fontweight='bold',
                 bbox=dict(facecolor='lightblue', boxstyle='round,pad=0.5', edgecolor='gray'))
                 
    plt.title("Pre-Judgment Outcome Prediction GNN Pipeline", fontsize=16, fontweight='bold', pad=20)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('/scratch/ziv_baretto/Thesis_Ziv/Capstone-Thesis-/GNN/document/pipeline_flow.png', dpi=300, bbox_inches='tight')
    plt.close()

def draw_graph_schema():
    plt.figure(figsize=(16, 12))
    G = nx.DiGraph()
    
    # Root Case
    G.add_node("case", color='#ff9999', node_type='Root Case')
    
    # Texts
    texts = ["preamble", "facts", "arguments"]
    for t in texts:
        G.add_node(t, color='#99ccff', node_type='Text Node')
        
    # Entities
    entities = ["petitioner", "respondent", "court", "judge", "lawyer", 
                "petitioner_lawyer", "defence_lawyer", "statute", "provision", "precedent"]
    for e in entities:
        G.add_node(e, color='#99ff99', node_type='Entity/Authority Node')
        
    # Context
    context = ["org", "gpe", "date", "case_number"]
    for c in context:
        G.add_node(c, color='#ffcc99', node_type='Optional Context')

    # Add edges
    edges = [
        ("case", "preamble"), ("case", "facts"), ("case", "arguments"),
        ("case", "petitioner"), ("case", "respondent"), ("case", "court"),
        ("case", "judge"), ("case", "lawyer"), ("case", "petitioner_lawyer"), 
        ("case", "defence_lawyer"),
        ("arguments", "statute"), ("arguments", "provision"), ("arguments", "precedent"),
        ("provision", "statute"),
        ("petitioner_lawyer", "arguments"), ("defence_lawyer", "arguments"),
        ("provision", "arguments"), ("statute", "arguments"),
        ("petitioner", "arguments"), ("respondent", "arguments"),
        ("judge", "arguments")
    ]
    for c in context:
        edges.append(("case", c))
        
    G.add_edges_from(edges)
    
    pos = nx.spring_layout(G, k=0.9, iterations=50, seed=42)
    
    # Slightly tweak 'case' position to center
    pos["case"] = [0, 0]
    
    # Separate colors
    colors = [nx.get_node_attributes(G, 'color').get(n) for n in G.nodes()]
    
    nx.draw(G, pos, with_labels=False, node_size=3500, node_color=colors,
            edge_color='gray', arrowsize=15, arrowstyle='-|>', width=1.5, alpha=0.9)
            
    for node, (x, y) in pos.items():
        plt.text(x, y, node, fontsize=10, ha='center', va='center', fontweight='bold')
        
    # Legend
    legend_elements = [
        mpatches.Patch(color='#ff9999', label='Root Case'),
        mpatches.Patch(color='#99ccff', label='Text Nodes'),
        mpatches.Patch(color='#99ff99', label='Entity/Authority Nodes'),
        mpatches.Patch(color='#ffcc99', label='Optional Context')
    ]
    plt.legend(handles=legend_elements, loc='upper left', fontsize=12)
    
    plt.title("Graph Schema: Node Connections", fontsize=16, fontweight='bold', pad=20)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('/scratch/ziv_baretto/Thesis_Ziv/Capstone-Thesis-/GNN/document/graph_schema.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    print("Generating Pipeline Flow PNG...")
    draw_pipeline()
    print("Generating Graph Schema PNG...")
    draw_graph_schema()
    print("Done!")
