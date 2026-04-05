import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def draw_message_passing_hops():
    plt.figure(figsize=(16, 12))
    G = nx.DiGraph()

    # Target Node (Layer 2 Output)
    G.add_node("case (Target)", layer=0, color='#ff9999')

    # Layer 1 Nodes (1-hop neighbors from case)
    l1_nodes = ["preamble", "facts", "arguments", "petitioner", "respondent", 
                "court", "judge", "lawyer", "petitioner_lawyer", "defence_lawyer",
                "org", "gpe", "date", "case_number"]
    
    for i, n in enumerate(l1_nodes):
        node_name = f"{n}\n(1-hop)"
        G.add_node(node_name, layer=1, color='#99ccff' if n in ["preamble", "facts", "arguments"] else '#99ff99')
        # Edges directed towards case to represent message passing aggregation
        G.add_edge(node_name, "case (Target)")

    # Layer 2 Nodes (2-hop neighbors through arguments/lawyers)
    # These pass messages to 'arguments'
    l2_to_args = ["statute", "provision", "precedent", "petitioner", "respondent", "judge"]
    for i, n in enumerate(l2_to_args):
        node_name = f"{n}\n(2-hop to args)"
        G.add_node(node_name, layer=2, color='#99ff99')
        G.add_edge(node_name, "arguments\n(1-hop)")

    # 2-hop through provision -> statute
    G.add_node("statute\n(3-hop via prov)", layer=3, color='#99ff99')
    G.add_edge("statute\n(3-hop via prov)", "provision\n(2-hop to args)")

    # Lawyer citations
    G.add_edge("petitioner_lawyer\n(1-hop)", "arguments\n(1-hop)")
    G.add_edge("defence_lawyer\n(1-hop)", "arguments\n(1-hop)")

    pos = nx.multipartite_layout(G, subset_key="layer", align="horizontal")
    
    # Custom node colors based on our assignments
    colors = [nx.get_node_attributes(G, 'color').get(n, '#cccccc') for n in G.nodes()]

    nx.draw(G, pos, with_labels=False, node_size=3500, node_color=colors,
            edge_color='gray', arrowsize=20, arrowstyle='-|>', width=2)
            
    for node, (x, y) in pos.items():
        plt.text(x, y, node, fontsize=9, ha='center', va='center', fontweight='bold')

    plt.title("2-Layer GNN Message Passing (Information Flow to 'case')", fontsize=16, fontweight='bold', pad=20)
    plt.text(-0.5, 1.1, "Layer 3\nNeighbors", fontsize=12, fontweight='bold', ha='center', transform=plt.gca().transAxes)
    plt.text(-0.1, 1.1, "Layer 2\nNeighbors", fontsize=12, fontweight='bold', ha='center', transform=plt.gca().transAxes)
    plt.text(0.5, 1.1, "Layer 1\nNeighbors", fontsize=12, fontweight='bold', ha='center', transform=plt.gca().transAxes)
    plt.text(1.0, 1.1, "Root Node", fontsize=12, fontweight='bold', ha='center', transform=plt.gca().transAxes)
    
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('/scratch/ziv_baretto/Thesis_Ziv/Capstone-Thesis-/GNN/document/message_passing_hops.png', dpi=300, bbox_inches='tight')
    plt.close()

def draw_gnn_layer_internals():
    plt.figure(figsize=(12, 14))
    G = nx.DiGraph()

    nodes = {
        "Input HeteroData": (2, 10),
        "Text Embeddings (e.g. 384d)": (1, 9),
        "Scalar Features (e.g. 12d)": (3, 9),
        "Concat Features (Node.x)": (2, 8),
        "Linear Projection (-> 128d)": (2, 7),
        "+ Learnable Type Embeddings": (2, 6),
        "L1: HGTConv Message Passing": (2, 5),
        "L1: Dropout (p=0.2)": (2, 4),
        "L1: LayerNorm & Residual(+)": (2, 3),
        "L1: ReLU Activation": (2, 2),
        "L2: HGTConv Message Passing": (2, 1),
        "L2: Dropout & Norm & ReLU": (2, 0),
        "Extract 'case' node hidden state": (2, -1),
        "MLP Classifier (Linear/Dropout)": (2, -2),
        "Output Logits": (2, -3)
    }

    for node, pos in nodes.items():
        G.add_node(node, pos=pos)

    edges = [
        ("Input HeteroData", "Text Embeddings (e.g. 384d)"),
        ("Input HeteroData", "Scalar Features (e.g. 12d)"),
        ("Text Embeddings (e.g. 384d)", "Concat Features (Node.x)"),
        ("Scalar Features (e.g. 12d)", "Concat Features (Node.x)"),
        ("Concat Features (Node.x)", "Linear Projection (-> 128d)"),
        ("Linear Projection (-> 128d)", "+ Learnable Type Embeddings"),
        ("+ Learnable Type Embeddings", "L1: HGTConv Message Passing"),
        ("L1: HGTConv Message Passing", "L1: Dropout (p=0.2)"),
        ("L1: Dropout (p=0.2)", "L1: LayerNorm & Residual(+)"),
        ("L1: LayerNorm & Residual(+)", "L1: ReLU Activation"),
        ("L1: ReLU Activation", "L2: HGTConv Message Passing"),
        ("L2: HGTConv Message Passing", "L2: Dropout & Norm & ReLU"),
        ("L2: Dropout & Norm & ReLU", "Extract 'case' node hidden state"),
        ("Extract 'case' node hidden state", "MLP Classifier (Linear/Dropout)"),
        ("MLP Classifier (Linear/Dropout)", "Output Logits")
    ]
    
    # Residual jump visualization
    G.add_edge("+ Learnable Type Embeddings", "L1: LayerNorm & Residual(+)", style='dashed')
    G.add_edge("L1: ReLU Activation", "L2: Dropout & Norm & ReLU", style='dashed')

    G.add_edges_from(edges)
    pos = nx.get_node_attributes(G, 'pos')

    nx.draw(G, pos, with_labels=False, node_size=6000, node_color='#d9d2e9', edge_color='gray',
            arrowsize=20, arrowstyle='-|>', width=2)
            
    # Draw dashed edges specifically
    dashed_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('style') == 'dashed']
    nx.draw_networkx_edges(G, pos, edgelist=dashed_edges, style='dashed', alpha=0.5, edge_color='red', width=2, arrowsize=20, connectionstyle='arc3,rad=-0.5')

    for node, (x, y) in pos.items():
        plt.text(x, y, node, fontsize=10, ha='center', va='center', fontweight='bold',
                 bbox=dict(facecolor='#d9d2e9', boxstyle='round,pad=0.5', edgecolor='gray'))

    plt.title("GNN Layer Internals (hetero_gnn.py)", fontsize=16, fontweight='bold', pad=20)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('/scratch/ziv_baretto/Thesis_Ziv/Capstone-Thesis-/GNN/document/gnn_layer_internals.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    print("Generating Detailed Message Passing Flow...")
    draw_message_passing_hops()
    print("Generating Layer Internals...")
    draw_gnn_layer_internals()
    print("Done!")
