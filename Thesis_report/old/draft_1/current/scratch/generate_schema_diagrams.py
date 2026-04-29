import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.patches as mpatches

def draw_single_case():
    G = nx.DiGraph()
    pos = {
        'Case': (0, 0),
        'Preamble': (-1.5, 1),
        'Facts': (-2, 0),
        'Arguments': (-1.5, -1),
        'Court': (1.5, 1),
        'Judge': (2, 0),
        'Lawyer': (1.5, -1),
        'Precedent': (-3, -2),
        'Statute': (-1.5, -2.5),
        'Provision': (0, -2.5)
    }
    
    colors = {
        'Case': '#B3CDE3',
        'Preamble': '#CCEBC5', 'Facts': '#CCEBC5', 'Arguments': '#CCEBC5',
        'Court': '#FED9A6', 'Judge': '#FED9A6', 'Lawyer': '#FED9A6',
        'Precedent': '#FED9A6', 'Statute': '#FED9A6', 'Provision': '#FED9A6'
    }
    
    edges = [
        ('Case', 'Preamble', 'has_preamble'), ('Case', 'Facts', 'has_facts'), ('Case', 'Arguments', 'has_arguments'),
        ('Case', 'Court', 'heard_in'), ('Case', 'Judge', 'decided_by'), ('Case', 'Lawyer', 'has_lawyer'),
        ('Arguments', 'Precedent', 'cites'), ('Arguments', 'Statute', 'cites'), ('Statute', 'Provision', 'belongs_to')
    ]
    
    for n in pos:
        G.add_node(n, color=colors[n])
    for u, v, l in edges:
        G.add_edge(u, v, label=l)
        
    plt.figure(figsize=(9, 6))
    ax = plt.gca()
    nx.draw_networkx_nodes(G, pos, node_color=[G.nodes[n]['color'] for n in G.nodes()], 
                           node_size=3500, node_shape='o', alpha=1.0, edgecolors='black')
    
    # Custom edges
    for u, v, l in edges:
        ax.annotate("", xy=pos[v], xycoords='data', xytext=pos[u], textcoords='data',
                    arrowprops=dict(arrowstyle="->", color="#333333", shrinkA=25, shrinkB=25, lw=1.5))
        mid_x = (pos[u][0] + pos[v][0]) / 2
        mid_y = (pos[u][1] + pos[v][1]) / 2 + 0.1
        plt.text(mid_x, mid_y, l, fontsize=9, ha='center', va='center', rotation=0, color='#333333', 
                 bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=0.5))

    nx.draw_networkx_labels(G, pos, font_size=10, font_family='sans-serif', font_color='black')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('../figures/schema_single_case.png', dpi=300, bbox_inches='tight')
    plt.close()

def draw_cross_case():
    G = nx.DiGraph()
    pos = {
        'Case A\n(2015)': (0, 2),
        'Court A': (-2, 2),
        'Arguments A': (2, 2),
        'Case B\n(2022)': (0, -2),
        'Court B': (-2, -2),
        'Arguments B': (2, -2),
        'Statute\n(Shared)': (4, 0),
        'Provision\n(Shared)': (6, 0),
        'Precedent\n(Shared)': (2, 0)
    }
    
    colors = {
        'Case A\n(2015)': '#B3CDE3', 'Case B\n(2022)': '#B3CDE3',
        'Court A': '#FBB4AE', 'Court B': '#FBB4AE',
        'Arguments A': '#CCEBC5', 'Arguments B': '#CCEBC5',
        'Statute\n(Shared)': '#FED9A6', 'Provision\n(Shared)': '#FED9A6', 'Precedent\n(Shared)': '#FED9A6'
    }
    
    edges = [
        ('Case A\n(2015)', 'Court A', 'heard_in'), ('Case A\n(2015)', 'Arguments A', 'has_args'),
        ('Case B\n(2022)', 'Court B', 'heard_in'), ('Case B\n(2022)', 'Arguments B', 'has_args'),
        ('Arguments A', 'Statute\n(Shared)', 'cites'), ('Arguments B', 'Statute\n(Shared)', 'cites'),
        ('Arguments A', 'Precedent\n(Shared)', 'cites'), ('Arguments B', 'Precedent\n(Shared)', 'cites'),
        ('Statute\n(Shared)', 'Provision\n(Shared)', 'belongs_to')
    ]
    
    for n in pos:
        G.add_node(n, color=colors[n])
        
    plt.figure(figsize=(10, 7))
    ax = plt.gca()
    nx.draw_networkx_nodes(G, pos, node_color=[colors[n] for n in G.nodes()], 
                           node_size=4000, node_shape='o', alpha=1.0, edgecolors='black')
                           
    for u, v, l in edges:
        ax.annotate("", xy=pos[v], xycoords='data', xytext=pos[u], textcoords='data',
                    arrowprops=dict(arrowstyle="->", color="#333333", shrinkA=30, shrinkB=30, lw=1.5))
        mid_x = (pos[u][0] + pos[v][0]) / 2
        mid_y = (pos[u][1] + pos[v][1]) / 2 + 0.15
        plt.text(mid_x, mid_y, l, fontsize=9, ha='center', va='center', color='#333333',
                 bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=0.5))

    nx.draw_networkx_labels(G, pos, font_size=9, font_family='sans-serif', font_color='black')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('../figures/schema_cross_case.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    draw_single_case()
    draw_cross_case()
