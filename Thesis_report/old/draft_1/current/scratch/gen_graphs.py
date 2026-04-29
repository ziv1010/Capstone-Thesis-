import networkx as nx
import matplotlib.pyplot as plt

def draw_single_case():
    plt.figure(figsize=(10, 6))
    G = nx.DiGraph()
    
    pos = {
        'Case': (0, 0),
        'Preamble': (-2, 2),
        'Facts': (-3, 0),
        'Arguments': (-2, -2),
        'Court': (2, 2),
        'Judge': (3, 0),
        'Lawyer': (2, -2),
        'Statute': (-2, -4),
        'Provision': (0, -4),
        'Precedent': (-4, -4)
    }
    
    edges = [
        ('Case', 'Preamble', 'has_preamble'),
        ('Case', 'Facts', 'has_facts'),
        ('Case', 'Arguments', 'has_arguments'),
        ('Case', 'Court', 'heard_in'),
        ('Case', 'Judge', 'decided_by_bench'),
        ('Case', 'Lawyer', 'has_lawyer'),
        ('Arguments', 'Statute', 'cites_statute'),
        ('Arguments', 'Precedent', 'cites_precedent'),
        ('Statute', 'Provision', 'belongs_to')
    ]
    
    for u, v, l in edges:
        G.add_edge(u, v, label=l)
        
    node_colors = []
    for node in G.nodes():
        if node == 'Case': node_colors.append('#cfe2f3') # light blue
        elif node in ['Preamble', 'Facts', 'Arguments']: node_colors.append('#d9ead3') # light green
        else: node_colors.append('#fce5cd') # light orange
        
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_shape='o', node_size=3500, edgecolors='gray')
    nx.draw_networkx_edges(G, pos, width=1.5, alpha=0.8, edge_color='#6fa8dc', arrows=True, arrowsize=20)
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold', font_family='sans-serif')
    
    edge_labels = {(u, v): d['label'] for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, font_color='blue', bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))
    
    plt.title("Sample star-graph structure for a single Case node", fontsize=14, pad=20)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('/scratch/ziv_baretto/Thesis_Ziv/Capstone-Thesis-/Thesis_report/current/figures/single_case_graph.png', dpi=300, bbox_inches='tight')
    plt.close()

def draw_cross_case():
    plt.figure(figsize=(10, 8))
    G = nx.DiGraph()
    
    pos = {
        'Case A\n(2015)': (-2, 2),
        'Case B\n(2022)': (-2, -2),
        'Court A\n(Delhi HC)': (-4, 2),
        'Arguments A': (0, 2),
        'Court B\n(Bombay HC)': (-4, -2),
        'Arguments B': (0, -2),
        'Statute\n(IPC)': (2, 0),
        'Provision\n(Sec 420)': (4, 0),
        'Precedent\n(X vs Y, 2010)': (0, -4)
    }
    
    edges = [
        ('Case A\n(2015)', 'Court A\n(Delhi HC)', 'heard_in'),
        ('Case A\n(2015)', 'Arguments A', 'has_arguments'),
        ('Case B\n(2022)', 'Court B\n(Bombay HC)', 'heard_in'),
        ('Case B\n(2022)', 'Arguments B', 'has_arguments'),
        ('Arguments A', 'Statute\n(IPC)', 'cites_statute'),
        ('Arguments B', 'Statute\n(IPC)', 'cites_statute'),
        ('Statute\n(IPC)', 'Provision\n(Sec 420)', 'belongs_to'),
        ('Arguments A', 'Precedent\n(X vs Y, 2010)', 'cites_precedent'),
        ('Arguments B', 'Precedent\n(X vs Y, 2010)', 'cites_precedent')
    ]
    
    for u, v, l in edges:
        G.add_edge(u, v, label=l)
        
    node_colors = []
    for node in G.nodes():
        if 'Case' in node: node_colors.append('#cfe2f3') # light blue
        elif 'Arguments' in node: node_colors.append('#d9ead3') # light green
        elif 'Court' in node: node_colors.append('#f4cccc') # light red (local)
        else: node_colors.append('#ffe599') # light yellow (shared)
        
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_shape='o', node_size=4500, edgecolors='gray')
    nx.draw_networkx_edges(G, pos, width=1.5, alpha=0.8, edge_color='#6fa8dc', arrows=True, arrowsize=20)
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold', font_family='sans-serif')
    
    edge_labels = {(u, v): d['label'] for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, font_color='blue', bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))
    
    plt.title("Cross-connection of cases via shared nodes", fontsize=14, pad=20)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('/scratch/ziv_baretto/Thesis_Ziv/Capstone-Thesis-/Thesis_report/current/figures/cross_case_graph.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    draw_single_case()
    draw_cross_case()
    print("PNGs generated.")
