import streamlit as st
import os
from Bio.PDB import PDBParser, PDBList
import igraph as ig
import leidenalg
from sklearn.metrics import adjusted_rand_score
import pydssp
from Bio.PDB import PDBIO
from Bio.SeqUtils import seq1
import plotly.colors as pc
import numpy as np
import pandas as pd
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import streamlit.components.v1 as components
import json
import plotly.graph_objects as go
from io import StringIO
import tempfile
from scipy.spatial.distance import cdist
from collections import defaultdict
import base64
from PIL import Image  
import networkx as nx
im = Image.open("icon.png") 

st.set_page_config(
    page_title="Protein Contact Network Explorer", 
    layout="wide",
    page_icon=im  
)
high_res_config = {
    'toImageButtonOptions': {
        'format': 'png', 
        'height': 1200, 
        'width': 1600, 
        'scale': 10
    }
}

st.markdown("""
<style>
    /* Global App Background - Clean White/Light Grey */
    .stApp { background-color: #f8f9fa; font-family: "Segoe UI", "Helvetica Neue", Helvetica, Arial, sans-serif; }
    
    /* Header - NCBI Blue */
    .main-header {
        background-color: #112e51;
        padding: 25px;
        border-radius: 4px;
        border-bottom: 4px solid #f28e2b; /* Subtle structural accent */
        margin-bottom: 20px;
    }
    
    /* Standard Section Box - Flat, clean borders */
    .section-box {
        background: white;
        padding: 20px;
        border-radius: 4px;
        border: 1px solid #d1d5da;
        border-top: 4px solid #112e51;
        margin-bottom: 15px;
        min-height: 220px;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
    }

    /* --- UPLOADER STYLING --- */
    [data-testid="stFileUploader"] { margin: 0 auto; width: 90%; margin-top: 10px; }
    [data-testid="stFileUploader"] section { background-color: transparent; border: none; padding: 10px 0; }

    /* --- UI Buttons --- */
    .stButton>button {
        background-color: #112e51;
        color: white;
        border-radius: 3px;
        border: 1px solid #0b1f38;
        padding: 8px 20px;
        font-weight: 600;
        transition: background-color 0.2s ease;
    }
    .stButton>button:hover { background-color: #1a4478; border-color: #112e51; }
    
    .hub-card {
        background: white; padding: 15px; border-radius: 4px;
        border: 1px solid #e1e4e8; border-left: 4px solid #112e51;
        margin-bottom: 10px;
    }
    .metric-box {
        background: #f1f3f6; padding: 12px; border-radius: 4px;
        text-align: center; margin: 5px 0; border: 1px solid #e1e4e8;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { gap: 4px; }
    .stTabs [data-baseweb="tab"] {
        height: 45px; background-color: #e1e4e8; border-radius: 4px 4px 0 0;
        padding: 0px 20px; border: 1px solid #d1d5da; border-bottom: none;
    }
    .stTabs [aria-selected="true"] { background-color: white; border-top: 3px solid #112e51; color: #112e51; font-weight: bold; }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# HELPER FUNCTIONS & DATA
# -----------------------------------------------------------------------------
aa_1_letter = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V", "UNK": "X"
}

res_type_map = {
    "ALA": "hydrophobic", "VAL": "hydrophobic", "ILE": "hydrophobic",
    "LEU": "hydrophobic", "MET": "hydrophobic", "PHE": "hydrophobic",
    "TYR": "hydrophobic", "TRP": "hydrophobic", "PRO": "hydrophobic",
    "GLY": "polar", "SER": "polar", "THR": "polar", "CYS": "polar",
    "ASN": "polar", "GLN": "polar",
    "ASP": "negative", "GLU": "negative",
    "LYS": "positive", "ARG": "positive", "HIS": "positive"
}
res_type_map["UNK"] = "polar"

residue_type_colors = {
    "hydrophobic": "#D94E1E",
    "polar": "#003B6F",
    "positive": "#007A55",
    "negative": "#B32630",
}
default_color = "#8da0cb"

def generate_ngl_html(pdb_string, color_map, rep_style="cartoon"):
    """Helper to generate HTML for the NGL viewer with dynamic representations and fullscreen."""
    color_map_json = json.dumps(color_map)
    safe_pdb_string = pdb_string.replace('`', '\\`')
    
    extra_params = "radiusScale: 1.5" if rep_style == "backbone" else ""
    
    return f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <script src="https://unpkg.com/ngl@2.0.0-dev.37/dist/ngl.js"></script>
        <style>
            html, body {{ margin: 0; padding: 0; width: 100%; height: 100%; overflow: hidden; background-color: #fafafa; font-family: Arial, sans-serif; }}
            #container {{ width: 100%; height: 500px; position: relative; border-radius: 8px; box-shadow: inset 0 0 10px rgba(0,0,0,0.05); border: 1px solid #ddd; background-color: #fafafa; }}
            #viewport {{ width: 100%; height: 100%; }}
            
            .btn-group {{ position: absolute; top: 15px; right: 15px; z-index: 10; display: flex; align-items: center; gap: 15px; }}
            
            #fullscreen-btn {{
                background: none; color: #1e3c72; border: none; cursor: pointer;
                font-weight: bold; font-size: 13px; padding: 0; text-decoration: underline;
            }}
            #fullscreen-btn:hover {{ color: #2a5298; }}
            
            #download-btn {{
                padding: 6px 12px; background: #1e3c72; color: white;
                border: none; border-radius: 4px; cursor: pointer;
                font-weight: bold; font-size: 13px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.2); transition: background 0.2s;
            }}
            #download-btn:hover {{ background: #2a5298; }}
        </style>
    </head>
    <body>
        <div id="container">
            <div id="viewport"></div>
            <div class="btn-group">
                <button id="fullscreen-btn">Full Screen</button>
                <button id="download-btn">PNG</button>
            </div>
        </div>
        <script>
            document.addEventListener("DOMContentLoaded", function () {{
                var stage = new NGL.Stage("viewport", {{ backgroundColor: "#fafafa" }});
                var colorMap = {color_map_json};
                
                var schemeId = NGL.ColormakerRegistry.addScheme(function (params) {{
                    this.atomColor = function (atom) {{
                        if (colorMap[atom.resno]) {{
                            return parseInt(colorMap[atom.resno].replace("#", "0x"));
                        }}
                        return 0xDDDDDD; 
                    }};
                }});

                var pdbData = `{safe_pdb_string}`;
                var blob = new Blob([pdbData], {{ type: 'text/plain' }});
                
                stage.loadFile(blob, {{ ext: "pdb" }}).then(function (o) {{
                    o.addRepresentation("{rep_style}", {{ 
                        color: schemeId,
                        sele: "protein"  // <--- THIS STRIPS OUT THE WATER AND LIGANDS
                        {',' + extra_params if extra_params else ''}
                    }});
                    o.autoView();
                }});

                window.addEventListener("resize", function(event){{ stage.handleResize(); }}, false);

                document.getElementById('fullscreen-btn').addEventListener('click', function() {{
                    var container = document.getElementById('container');
                    if (!document.fullscreenElement) {{
                        if (container.requestFullscreen) {{ container.requestFullscreen(); }}
                        else if (container.webkitRequestFullscreen) {{ container.webkitRequestFullscreen(); }}
                        else if (container.msRequestFullscreen) {{ container.msRequestFullscreen(); }}
                    }} else {{
                        if (document.exitFullscreen) {{ document.exitFullscreen(); }}
                        else if (document.webkitExitFullscreen) {{ document.webkitExitFullscreen(); }}
                        else if (document.msExitFullscreen) {{ document.msExitFullscreen(); }}
                    }}
                }});

                document.addEventListener('fullscreenchange', function() {{
                    var btn = document.getElementById('fullscreen-btn');
                    if (document.fullscreenElement) {{ btn.textContent = "Exit Full Screen"; }}
                    else {{ btn.textContent = "Full Screen"; }}
                    setTimeout(function(){{ stage.handleResize(); }}, 100);
                }});

                document.getElementById('download-btn').addEventListener('click', function() {{
                    stage.makeImage({{
                        factor: 4, 
                        antialias: true, 
                        trim: false, 
                        transparent: true
                    }}).then(function(blob) {{
                        NGL.download(blob, "Protein_3D_Snapshot.png");
                    }});
                }});
            }});
        </script>
    </body>
    </html>
    """

def compute_enhanced_metrics(adj_np, dist_matrix):
    """
    UPDATED: Uses NetworkX for verified calculations of Centrality measures.
    Ensures Closeness Centrality is normalized (0 to 1) using wf_improved=True.
    """
    N = len(adj_np)
    
    # Create NetworkX graph for accurate topological metrics
    G = nx.from_numpy_array(adj_np)
    
    # 1. Degree
    degrees = np.sum(adj_np, axis=0)
    
    # 2. Betweenness Centrality
    # NetworkX normalizes this by default
    betweenness_dict = nx.betweenness_centrality(G, normalized=True)
    betweenness = np.array([betweenness_dict[i] for i in range(N)])
    
    # 3. Closeness Centrality (UPDATED)
    # wf_improved=True uses the Wasserman and Faust formula.
    # This allows for correct normalization (0 to 1) even if the graph 
    # has multiple disconnected components.
    closeness_dict = nx.closeness_centrality(G, wf_improved=True)
    closeness = np.array([closeness_dict[i] for i in range(N)])
    
    # 4. Clustering Coefficient
    clustering_dict = nx.clustering(G)
    clustering = np.array([clustering_dict[i] for i in range(N)])
    
    return {
        'degree': degrees,
        'betweenness': betweenness,
        'closeness': closeness,
        'clustering': clustering
    }


def identify_hub_communities(adj_np, hub_indices, residues):
    communities = {}
    for hub_idx in hub_indices:
        neighbors = np.where(adj_np[hub_idx] == 1)[0]
        res_types = defaultdict(int)
        for n_idx in neighbors:
            res_name = residues[n_idx].get_resname().strip().upper()
            res_type = res_type_map.get(res_name, "polar")
            res_types[res_type] += 1
        
        communities[hub_idx] = {
            'size': len(neighbors),
            'composition': dict(res_types),
            'neighbors': neighbors.tolist()
        }
    return communities

def get_node_coordinate(res, rep_mode):
    """Extracts the specific coordinate based on the selected network representation."""
    if rep_mode == "C-alpha":
        return res['CA'].get_coord() if 'CA' in res else None
        
    elif rep_mode == "C-beta":
        if 'CB' in res:
            return res['CB'].get_coord()
        elif 'CA' in res:
            return res['CA'].get_coord() # Glycine fallback
        return None
        
    elif rep_mode == "Side-chain Centroid":
        sc_coords = [
            atom.get_coord() for atom in res.get_atoms()
            if atom.element != 'H' and atom.get_name() not in ('N', 'CA', 'C', 'O')
        ]
        if len(sc_coords) > 0:
            return np.mean(sc_coords, axis=0)
        elif 'CA' in res:
            return res['CA'].get_coord() # Glycine / missing side-chain fallback
        return None
        
    return None
def compute_pcn_df(structure, model_id, chain_id, threshold, rep_mode, progress=None, progress_label=None):
    model = structure[model_id - 1]
    chain = model[chain_id]
    
    standard_aa = {
        'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 
        'GLN', 'GLU', 'GLY', 'HIS', 'ILE', 
        'LEU', 'LYS', 'MET', 'PHE', 'PRO', 
        'SER', 'THR', 'TRP', 'TYR', 'VAL'
    }

    residues = []
    labels = []
    vis_labels = []
    coords = []
    
    if progress:
        progress.progress(60)
        progress_label.text(f"Extracting {rep_mode} coordinates...")

    for res in chain:
        res_name = res.get_resname().strip().upper()
        
        if res.id[0] == ' ' and res_name in standard_aa:
           
            coord = get_node_coordinate(res, rep_mode)
            
            if coord is not None:
                residues.append(res)
                labels.append(f"{res_name}-{res.id[1]}")
                vis_labels.append(f"{res_name}\n{res.id[1]}")
                coords.append(coord)

    if not residues:
        return None, None, None, None, None, None, None

    N = len(residues)
    coords = np.array(coords)
    
    if progress:
        progress.progress(70)
        progress_label.text(f"Computing distances for {N} residues...")

    dist_matrix = cdist(coords, coords, metric='euclidean')
    dist_df = pd.DataFrame(dist_matrix, index=labels, columns=labels)
    
    adj_matrix = np.where((dist_matrix < threshold) & (dist_matrix > 0), 1, 0)
    adj_df = pd.DataFrame(adj_matrix, index=labels, columns=labels)

    if progress:
        progress.progress(85)
        progress_label.text("Calculating network metrics...")

    metrics = compute_enhanced_metrics(adj_matrix, dist_matrix)
    
    return adj_df, dist_df, labels, vis_labels, coords, residues, metrics


def render_distance_heatmap(dist_df):
    """
    Renders an interactive, strictly square heatmap for the distance matrix.
    Color Scheme: Fluorescent Red (Close) -> Yellow -> Green -> Cyan -> Blue (Far).
    """
    
    try:
        axis_labels = [int(label.split('-')[-1]) for label in dist_df.index]
    except:
        axis_labels = dist_df.index
    custom_colorscale = [
        [0.00, 'rgb(255, 0, 0)'],    # Red (Close)
        [0.25, 'rgb(255, 255, 0)'],  # Yellow (Bright)
        [0.50, 'rgb(0, 255, 0)'],    # Green (Fluorescent)
        [0.75, 'rgb(0, 255, 255)'],  # Cyan (Bright)
        [1.00, 'rgb(0, 0, 255)']     # Blue (Far)
    ]

    fig = go.Figure(data=go.Heatmap(
        z=dist_df.values,
        x=axis_labels,
        y=axis_labels,
        colorscale=custom_colorscale,
        colorbar=dict(
            title='Distance (Å)',
            tickmode='auto', 
            nticks=6,
            x=0.78
        ),
        hovertemplate=(
            '<b>Residue i:</b> %{y}<br>'
            '<b>Residue j:</b> %{x}<br>'
            '<b>Distance:</b> %{z:.2f} Å<extra></extra>'
        )
    ))

    fig.update_layout(
        title={
            'text': 'Pairwise Distance Matrix Heatmap',
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        xaxis_title="Residue Number",
        yaxis_title="Residue Number",
        width=700,
        height=700,
        autosize=False,
        xaxis=dict(
            scaleanchor='y',
            scaleratio=1,
            constrain='domain',
            showgrid=False
        ),
        yaxis=dict(
            autorange='reversed', 
            scaleanchor='x',      
            scaleratio=1,
            constrain='domain',
            showgrid=False
        ),
        margin=dict(l=50, r=50, t=80, b=50)
    )
    
    st.plotly_chart(fig, use_container_width=True, config=high_res_config)


def render_adjacency_heatmap(adj_df):
    """
    Renders a strictly square binary heatmap for the adjacency matrix.
    Black = Connected, White = Not Connected.
    """
  
    try:
        axis_labels = [int(label.split('-')[-1]) for label in adj_df.index]
    except:
        axis_labels = adj_df.index

    z_vals = adj_df.values
    hover_text = np.where(z_vals == 1, "Connected", "Not Connected")

    fig = go.Figure(data=go.Heatmap(
        z=z_vals,
        x=axis_labels,
        y=axis_labels,
        colorscale=[[0, 'white'], [1, 'black']],
        showscale=False, 
        xgap=0.5,
        ygap=0.5,
        customdata=hover_text,
        hovertemplate=(
            '<b>Residue i:</b> %{y}<br>'
            '<b>Residue j:</b> %{x}<br>'
            '<b>Status:</b> %{customdata}<extra></extra>'
        )
    ))

    fig.update_layout(
        title={
            'text': 'Binary Adjacency Map',
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        xaxis_title="Residue Number",
        yaxis_title="Residue Number",
        width=700,
        height=700,
        autosize=False,
        xaxis=dict(
            scaleanchor='y',
            scaleratio=1,
            constrain='domain',
            showgrid=False
        ),
        yaxis=dict(
            autorange='reversed',
            scaleanchor='x',
            scaleratio=1,
            constrain='domain',
            showgrid=False
        ),
        margin=dict(l=50, r=50, t=80, b=50)
    )
    
    st.plotly_chart(fig, use_container_width=True, config=high_res_config)


def build_3d_figure_enhanced(labels, vis_labels, coords, adj_np, dist_matrix, node_colors, node_sizes, 
                            residues, hub_indices_global, metrics, highlight_communities=False, 
                            view_mode="Show All", path_indices=None, show_legend=True, 
                            focused_hub_coords=None): # Added focused_hub_coords arg
    N = len(labels)
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]

    hover_texts = []
    for i in range(N):
        res = residues[i]
        name = res.get_resname().strip().upper()
        if name not in res_type_map: name = "UNK"
        bc_val = metrics['betweenness'][i] if 'betweenness' in metrics else 0.0
        
        hover_text = (
            f"<b>{labels[i]}</b><br>" 
            f"Type: {res_type_map[name].capitalize()}<br>"
            f"Degree: {int(metrics['degree'][i])}<br>"
            f"Clustering: {metrics['clustering'][i]:.3f}<br>"
            f"Closeness: {metrics['closeness'][i]:.3f}<br>"
            f"Betweenness: {bc_val:.4f}"
        )
        hover_texts.append(hover_text)

    # --- Edge Generation ---
    traces = []
    path_edge_x, path_edge_y, path_edge_z = [], [], []
    matched_edge_x, matched_edge_y, matched_edge_z = [], [], []
    ghost_edge_x, ghost_edge_y, ghost_edge_z = [], [], []
    
    path_set = set(path_indices) if path_indices else set()

    for i in range(N):
        for j in range(i + 1, N):
            if adj_np[i, j] == 1:
                # 1. Check for Path Edge
                is_path_edge = False
                if view_mode == "Shortest Path (Betweenness)" and path_indices:
                    if i in path_set and j in path_set:
                        try:
                            idx_i = path_indices.index(i)
                            idx_j = path_indices.index(j)
                            if abs(idx_i - idx_j) == 1:
                                is_path_edge = True
                        except ValueError:
                            pass
                
                # 2. Check for "Matched" (Visible) Edge
                is_matched_pair = (node_sizes[i] > 5 and node_sizes[j] > 5)

                if is_path_edge:
                    path_edge_x.extend([coords[i][0], coords[j][0], None])
                    path_edge_y.extend([coords[i][1], coords[j][1], None])
                    path_edge_z.extend([coords[i][2], coords[j][2], None])
                elif is_matched_pair:
                    matched_edge_x.extend([coords[i][0], coords[j][0], None])
                    matched_edge_y.extend([coords[i][1], coords[j][1], None])
                    matched_edge_z.extend([coords[i][2], coords[j][2], None])
                else:
                    ghost_edge_x.extend([coords[i][0], coords[j][0], None])
                    ghost_edge_y.extend([coords[i][1], coords[j][1], None])
                    ghost_edge_z.extend([coords[i][2], coords[j][2], None])

    # Trace A: Ghost Edges
    if ghost_edge_x:
        traces.append(go.Scatter3d(
            x=ghost_edge_x, y=ghost_edge_y, z=ghost_edge_z,
            mode="lines", line=dict(color='rgba(80, 80, 80, 0.4)', width=1), 
            hoverinfo="none", showlegend=False
        ))

    # Trace B: Matched Edges
    if matched_edge_x:
        col = 'black'
        wid = 2
        if view_mode == "Show Hubs Only" and highlight_communities:
             col = 'rgba(200, 50, 50, 0.6)'
             wid = 4
        traces.append(go.Scatter3d(
            x=matched_edge_x, y=matched_edge_y, z=matched_edge_z,
            mode="lines", line=dict(color=col, width=wid),
            hoverinfo="none", showlegend=False
        ))

    
    if path_edge_x:
        traces.append(go.Scatter3d(
            x=path_edge_x, y=path_edge_y, z=path_edge_z,
            mode="lines",
           
            line=dict(color='orange', width=4),
            hoverinfo="none", showlegend=False
        ))

    # Trace D: Nodes
    traces.append(
        go.Scatter3d(
            x=x, y=y, z=z,
            mode="markers+text", 
            marker=dict(symbol='circle', size=node_sizes, color=node_colors, line=dict(width=1.5, color="white")),
            text=vis_labels, textposition="middle center",
            textfont=dict(family="Arial", size=10, color="black", weight="bold"), 
            hovertext=hover_texts, hoverinfo="text", showlegend=False,
        )
    )

   
    if show_legend:
        legend_items = [("Hydrophobic", "#D94E1E"), ("Polar", "#003B6F"), ("Positive", "#007A55"), ("Negative", "#B32630")]
        for name, color in legend_items:
            traces.append(go.Scatter3d(
                x=[None], y=[None], z=[None], 
                mode='markers', marker=dict(size=15, color=color), 
                name=name, showlegend=True
            ))
    
    fig = go.Figure(data=traces)

    layout_dict = dict(
        paper_bgcolor="#f8f9fa",
        plot_bgcolor="#f8f9fa",
        showlegend=show_legend,
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99, bgcolor="rgba(255, 255, 255, 0.8)", bordercolor="#e0e0e0", borderwidth=1),
        scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False), aspectmode="data", bgcolor="#f8f9fa", dragmode="orbit"),
        margin=dict(l=0, r=0, t=40, b=0),
        height=900 
    )

    if focused_hub_coords is not None:
        cx, cy, cz = focused_hub_coords
        layout_dict['scene']['camera'] = {
            "center": {"x": 0, "y": 0, "z": 0},
            # Divisor controls zoom level, +0.5 offsets viewing angle slightly
            "eye": {"x": cx/40 + 0.5, "y": cy/40 + 0.5, "z": cz/40 + 0.5}
        }

    fig.update_layout(layout_dict)
    return fig


def render_hub_analysis_panel(labels, residues, metrics, hub_indices, communities, adj_np):
    hub_icon_svg = """<svg width="80" height="80" viewBox="-20 -20 140 140" xmlns="http://www.w3.org/2000/svg" style="display: block;">
    <g stroke="#1e3c72" stroke-width="8" stroke-linecap="round">
        <line x1="50" y1="50" x2="50" y2="10" />
        <line x1="50" y1="50" x2="90" y2="50" />
        <line x1="50" y1="50" x2="50" y2="90" />
        <line x1="50" y1="50" x2="10" y2="50" />
        <line x1="50" y1="50" x2="22" y2="22" />
        <line x1="50" y1="50" x2="78" y2="22" />
        <line x1="50" y1="50" x2="78" y2="78" />
        <line x1="50" y1="50" x2="22" y2="78" />
    </g>
    <circle cx="50" cy="50" r="18" fill="#1e3c72" />
    <circle cx="50" cy="10" r="8" fill="#1e3c72" />
    <circle cx="90" cy="50" r="8" fill="#1e3c72" />
    <circle cx="50" cy="90" r="8" fill="#1e3c72" />
    <circle cx="10" cy="50" r="8" fill="#1e3c72" />
    <circle cx="22" cy="22" r="8" fill="#1e3c72" />
    <circle cx="78" cy="22" r="8" fill="#1e3c72" />
    <circle cx="78" cy="78" r="8" fill="#1e3c72" />
    <circle cx="22" cy="78" r="8" fill="#1e3c72" />
</svg>"""

    header_html = f"""<div style="background: white; padding: 20px; border-radius: 12px; margin: 30px 0 20px 0; box-shadow: 0 4px 6px rgba(0,0,0,0.1); border-left: 6px solid #1e3c72; display: flex; align-items: center; gap: 25px;">
    <div style="flex-shrink: 0;">{hub_icon_svg}</div>
    <div><h3 style="margin: 0; color: #1e3c72; font-size: 38px !important; font-weight: 800 !important; line-height: 1.2;">Hub Analysis</h3></div>
</div>"""

    st.markdown(header_html, unsafe_allow_html=True)
    
    if len(hub_indices) == 0:
        st.info("No hubs detected with current criteria.")
        return
    
    sorted_hubs = sorted(hub_indices, key=lambda i: metrics['degree'][i], reverse=True)
    top_n = 7
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""<div class="metric-box"><div style="font-size: 24px; font-weight: 700; color: #2a5298;">{len(hub_indices)}</div><div style="font-size: 12px; color: #666;">Total Hubs</div></div>""", unsafe_allow_html=True)
    with col2:
        avg_degree = np.mean([metrics['degree'][i] for i in hub_indices])
        st.markdown(f"""<div class="metric-box"><div style="font-size: 24px; font-weight: 700; color: #2a5298;">{avg_degree:.1f}</div><div style="font-size: 12px; color: #666;">Avg Hub Degree</div></div>""", unsafe_allow_html=True)
    
    st.markdown("---")
    
    if len(sorted_hubs) > 0:
        csv_data = []
        for idx in sorted_hubs:
            label = labels[idx]
            res_name = residues[idx].get_resname().strip()
            res_type = res_type_map.get(res_name, "polar")
            bc_val = metrics['betweenness'][idx] if 'betweenness' in metrics else 0.0
            
            csv_data.append({
                "Residue Label": label, 
                "Residue Name": res_name, 
                "Residue Type": res_type,
                "Degree": int(metrics['degree'][idx]), 
                "Clustering Coeff": f"{metrics['clustering'][idx]:.4f}", 
                "Closeness Centrality": f"{metrics['closeness'][idx]:.4f}",
                "Betweenness Centrality": f"{bc_val:.4f}"
            })
        
        df_hubs = pd.DataFrame(csv_data)
        csv_string = df_hubs.to_csv(index=False).encode('utf-8')
        
        if len(sorted_hubs) > top_n:
            st.info(f"Showing top {top_n} of {len(sorted_hubs)} hubs. Download CSV for full list.")
        
        st.download_button(label="📥 Download Full Hub Report (CSV)", data=csv_string, file_name="hub_analysis_report.csv", mime="text/csv")
    
    st.markdown("---")
    
    if len(sorted_hubs) <= top_n:
        st.markdown("#### 🔍 Hub Details")
    else:
        st.markdown(f"#### 🔍 Hub Details (Top {top_n})")
    
    display_hubs = sorted_hubs[:top_n]
    for rank, idx in enumerate(display_hubs, 1):
        label = labels[idx]
        res_name = residues[idx].get_resname().strip()
        res_type = res_type_map.get(res_name, "polar")
        color = residue_type_colors[res_type]
        degree = int(metrics['degree'][idx])
        clustering = metrics['clustering'][idx]
        closeness = metrics['closeness'][idx]
        bc_val = metrics['betweenness'][idx] if 'betweenness' in metrics else 0.0
        
        community = communities.get(idx, {})
        community_size = community.get('size', 0)
        composition = community.get('composition', {})
        comp_str = ", ".join([f"{k}: {v}" for k, v in composition.items()])
        
        st.markdown(f"""
        <div class="hub-card">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div><span style="font-size: 18px; font-weight: 700; color: {color};">#{rank} {label}</span><span style="color: #666; font-size: 14px; margin-left: 10px;">({res_name} - {res_type})</span></div>
                <div style="background: {color}; color: white; padding: 5px 12px; border-radius: 20px; font-weight: 600;">Deg: {degree}</div>
            </div>
            <div style="margin-top: 10px; font-size: 13px; color: #555;">
                <div style="display: flex; gap: 20px;">
                    <div>• Clustering: <b>{clustering:.3f}</b></div>
                    <div>• Closeness: <b>{closeness:.3f}</b></div>
                </div>
                <div style="margin-top:2px;">• Betweenness: <b>{bc_val:.4f}</b></div>
                <div style="margin-top:5px;"><b>Community:</b> {community_size} neighbors</div>
                <div style="font-size: 11px; color: #777;">{comp_str}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

def render_community_detection_module(structure, model_choice, chain_choice, adj_np, labels, coords, residues, pdb_string):
    st.markdown("---")
    st.markdown("### Leiden Community Detection")

    G = nx.from_numpy_array(adj_np)
    ig_graph = ig.Graph.from_networkx(G)
    partition = leidenalg.find_partition(ig_graph, leidenalg.ModularityVertexPartition)
    
    mod_score = partition.modularity
    num_communities = len(partition)
    
    community_mapping = {}
    for comm_id, node_indices in enumerate(partition):
        for node_idx in node_indices:
            community_mapping[int(ig_graph.vs[node_idx]['_nx_name'])] = comm_id + 1
            
    community_labels = [community_mapping[i] for i in range(len(labels))]

    ss_labels = ['L'] * len(labels)
    dssp_error = False
    try:
        full_coords = []
        for res in residues:
            try:
                full_coords.append([res['N'].get_coord(), res['CA'].get_coord(), res['C'].get_coord(), res['O'].get_coord()])
            except KeyError:
                ca = res['CA'].get_coord()
                full_coords.append([ca, ca, ca, ca])
        
        c3_labels = pydssp.assign(np.array(full_coords), out_type='c3')
        mapped_ss = ['L' if x == '-' else ('B' if x == 'E' else 'H') for x in c3_labels]
        ss_labels = mapped_ss
    except Exception as e:
        dssp_error = True
        st.error(f"PyDSSP Error: {e}")

    view_mode = st.radio("Visualisation Mode:", ["Network Graph (Plotly)", "3D Structure (NGL)"], horizontal=True, key="community_view_toggle")
    palette = pc.qualitative.Alphabet[:20] 
    
    if view_mode == "Network Graph (Plotly)":
        node_colors = [palette[(c_id - 1) % len(palette)] for c_id in community_labels]
        traces = []
        edge_x, edge_y, edge_z = [], [], []
        
        N = len(labels)
        for i in range(N):
            for j in range(i + 1, N):
                if adj_np[i, j] == 1:
                    edge_x.extend([coords[i][0], coords[j][0], None])
                    edge_y.extend([coords[i][1], coords[j][1], None])
                    edge_z.extend([coords[i][2], coords[j][2], None])

        traces.append(go.Scatter3d(
            x=edge_x, y=edge_y, z=edge_z, mode="lines", 
            line=dict(color='rgba(100, 100, 100, 0.6)', width=2), hoverinfo="none", showlegend=False
        ))

        hover_texts = [f"<b>{labels[i]}</b><br>Cluster {community_labels[i]}<br>DSSP: {ss_labels[i]}" for i in range(N)]
        
        
        traces.append(go.Scatter3d(
            x=coords[:, 0], y=coords[:, 1], z=coords[:, 2], mode="markers",
            marker=dict(symbol='circle', size=15, color=node_colors, line=dict(width=1, color="white")),
            hovertext=hover_texts, hoverinfo="text", showlegend=False
        ))

        fig = go.Figure(data=traces)
        fig.update_layout(scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False), aspectmode="data"), margin=dict(l=0, r=0, t=0, b=0), height=600, paper_bgcolor="#f8f9fa", plot_bgcolor="#f8f9fa")
        st.plotly_chart(fig, use_container_width=True, config={'toImageButtonOptions': {'format': 'png', 'filename': 'community_network_hd', 'height': 1200, 'width': 1600, 'scale': 6}, 'displaylogo': False, 'modeBarButtons': [['toImage']]})

    else:
        col_ctrl1, col_ctrl2 = st.columns(2)
        with col_ctrl1:
            rep_choice = st.selectbox("Protein Representation:", ["Cartoon", "Backbone Trace", "Ball & Stick", "Surface"], key="comm_rep_choice")
        with col_ctrl2:
            # NEW: Highlight specific cluster
            cluster_opts = ["None (Show All)"] + [f"Cluster {i}" for i in range(1, num_communities + 1)]
            highlight_choice = st.selectbox("Highlight Cluster:", cluster_opts, key="comm_highlight")

        rep_map = {"Cartoon": "cartoon", "Backbone Trace": "backbone", "Ball & Stick": "ball+stick", "Surface": "surface"}
        color_map = {}
        
        target_c_id = None if highlight_choice == "None (Show All)" else int(highlight_choice.replace("Cluster ", ""))
        
        for idx, label in enumerate(labels):
            res_num = int(label.split('-')[-1])
            c_id = community_labels[idx] 
            
            # Highlighting logic: grey out non-targets
            if target_c_id is None or c_id == target_c_id:
                color_map[res_num] = palette[(c_id - 1) % len(palette)]
            else:
                color_map[res_num] = "#E0E0E0" 
            
        components.html(generate_ngl_html(pdb_string, color_map, rep_style=rep_map[rep_choice]), height=520)

    
    st.markdown(f"""
    <div style='text-align: center; margin-top: 10px; margin-bottom: 20px;'>
        <span style='font-size: 18px; font-weight: bold; color: #112e51; margin-right: 30px;'>Modularity Score: {mod_score:.4f}</span>
        <span style='font-size: 18px; font-weight: bold; color: #112e51;'>Communities Detected: {num_communities}</span>
    </div>
    """, unsafe_allow_html=True)

    if dssp_error: st.warning("Secondary structure could not be calculated. Defaulting to Loops (L).")

    st.markdown("""
    <div style='background: #f0f4f8; padding: 12px; border-radius: 6px; border: 1px solid #c9d6e5; margin-bottom: 15px; font-family: Arial, sans-serif; font-size: 14px;'>
        <b>Secondary Structure Composition:</b><br>
        <span style='display:inline-block; width:12px; height:12px; background-color: #112e51; margin-right:5px; margin-left: 5px; vertical-align: middle;'></span> <b>H</b> = Helix &nbsp;&nbsp;&nbsp;
        <span style='display:inline-block; width:12px; height:12px; background-color: #f57c00; margin-right:5px; vertical-align: middle;'></span> <b>B</b> = Beta Strand &nbsp;&nbsp;&nbsp;
        <span style='display:inline-block; width:12px; height:12px; background-image: radial-gradient(#9e9e9e 30%, transparent 30%); background-size: 4px 4px; background-color: #e0e0e0; margin-right:5px; vertical-align: middle;'></span> <b>L</b> = Loops
    </div>
    """, unsafe_allow_html=True)
    
    clusters = defaultdict(list)
    cluster_ss_counts = defaultdict(lambda: {'H': 0, 'B': 0, 'L': 0})
    
    for idx, c_id in enumerate(community_labels):
        res_name3 = residues[idx].get_resname().strip()
        res_name1 = seq1(res_name3) if len(res_name3) == 3 else res_name3
        res_num = residues[idx].id[1]
        clusters[c_id].append(f"{res_name1}{res_num}")
        cluster_ss_counts[c_id][ss_labels[idx]] += 1

    legend_html = "<div style='background: white; padding: 15px; border-radius: 4px; border: 1px solid #ddd; max-height: 400px; overflow-y: auto;'>"
    for c_id in sorted(clusters.keys()):
        members = clusters[c_id]
        color = palette[(c_id - 1) % len(palette)]
        members_display = ", ".join(members[:5]) + "..." if len(members) > 5 else ", ".join(members)
        counts = cluster_ss_counts[c_id]
        total = sum(counts.values())
        pct_h = (counts['H'] / total) * 100 if total > 0 else 0
        pct_b = (counts['B'] / total) * 100 if total > 0 else 0
        pct_l = (counts['L'] / total) * 100 if total > 0 else 0

        legend_html += (
            f"<div style='margin-bottom: 16px;'>"
            f"  <div style='line-height: 1.5;'>"
            f"      <span style='display:inline-block; width:16px; height:16px; background-color:{color}; border-radius:50%; margin-right:8px; vertical-align:middle; border: 1px solid #aaa;'></span>"
            f"      <span style='font-family: monospace; font-weight: bold; color: #333;'>Cluster {c_id}</span> &mdash; "
            f"      <span style='font-family: monospace; color: #444; font-size: 14px;'>{members_display}</span>"
            f"  </div>"
            f"  <div style='display: flex; align-items: center; margin-top: 6px; padding-left: 28px;'>"
            f"      <div style='display: flex; height: 12px; width: 150px; background-color: #eee; border-radius: 2px; overflow: hidden; margin-right: 15px;'>"
            f"          <div style='width: {pct_h}%; background-color: #112e51;' title='Helix'></div>"
            f"          <div style='width: {pct_b}%; background-color: #f57c00;' title='Beta Strand'></div>"
            f"          <div style='width: {pct_l}%; background-image: radial-gradient(#9e9e9e 30%, transparent 30%); background-size: 4px 4px; background-color: #e0e0e0;' title='Loops'></div>"
            f"      </div>"
            f"      <div style='font-family: monospace; font-size: 13px; color: #333; font-weight: 600;'>"
            f"          H: {int(round(pct_h))}% &nbsp;&nbsp; B: {int(round(pct_b))}% &nbsp;&nbsp; L: {int(round(pct_l))}%"
            f"      </div>"
            f"  </div>"
            f"</div>"
        )
    legend_html += "</div>"
    st.markdown(legend_html, unsafe_allow_html=True)
    
    csv_data = [{"Residue Number": residues[i].id[1], "Amino Acid": seq1(residues[i].get_resname().strip()), "Cluster ID": c_id, "Secondary Structure": ss_labels[i]} for i, c_id in enumerate(community_labels)]
    st.markdown("<br>", unsafe_allow_html=True)
    st.download_button("📥 Download Full Cluster Membership (CSV/Excel)", data=pd.DataFrame(csv_data).to_csv(index=False).encode('utf-8'), file_name="cluster_membership.csv", mime="text/csv")

def render_ngl_3d_viewer(pdb_string, metrics, community_dict, residues, labels):
    """
    Renders an interactive 3D protein structure using NGL Viewer embedded via HTML.
    """
    st.markdown("---")
    st.markdown("### 3D Molecular Viewer (NGL)")
    
    # 1. Toggle Selection
    view_mode = st.radio(
        "Select Coloring Mode:",
        ["Betweenness Centrality", "Community Membership"],
        horizontal=True
    )
    
    # 2. Compute Color Maps
    color_map = {}
    legend_html = ""
    
    if view_mode == "Betweenness Centrality":
        bc_values = np.array(metrics.get('betweenness', [0]*len(labels)))
        
        log_bc = np.log1p(bc_values)
        log_min, log_max = log_bc.min(), log_bc.max()
        
        if log_max - log_min == 0:
            norm_bc = np.zeros_like(log_bc)
        else:
            norm_bc = (log_bc - log_min) / (log_max - log_min)
            
        cmap = cm.get_cmap('coolwarm')
        top_3_indices = np.argsort(bc_values)[-3:][::-1]
        top_3_info = []
        
        for idx, label in enumerate(labels):
            res_num = int(label.split('-')[-1])
            
            rgba = cmap(norm_bc[idx])
            hex_color = mcolors.to_hex(rgba)
            color_map[res_num] = hex_color
            
            if idx in top_3_indices:
                res_name = label.split('-')[0]
                res_name1 = seq1(res_name) if len(res_name) == 3 else res_name
                top_3_info.append(f"{res_name1}{res_num}")

        top_3_str = ", ".join(top_3_info)

        legend_html = f"""
        <div style="margin-top: 15px; font-family: sans-serif; text-align: center;">
            <div style="margin-bottom: 5px; font-weight: bold; color: #333;">Betweenness Centrality (Log-Normalized)</div>
            <div style="display: flex; align-items: center; justify-content: center; width: 100%;">
                <span style="margin-right: 10px; font-size: 14px; color: #555;">Low BC</span>
                <div style="height: 20px; width: 300px; border-radius: 4px; border: 1px solid #ccc; background: linear-gradient(to right, #3b4cc0, #dddddd, #b40426);"></div>
                <span style="margin-left: 10px; font-size: 14px; color: #555;">High BC</span>
            </div>
            <div style="margin-top: 5px; font-size: 13px; color: #d32f2f; font-weight: bold;">Top Bottlenecks: {top_3_str}</div>
        </div>
        """

    elif view_mode == "Community Membership":
        palette = pc.qualitative.D3
        clusters = defaultdict(list)
        
        for idx, label in enumerate(labels):
            res_num = int(label.split('-')[-1])
            res_name = label.split('-')[0]
            
            c_id = community_dict.get(res_num, 1) 
            
            color = palette[(c_id - 1) % len(palette)]
            color_map[res_num] = color
            
            res_name1 = seq1(res_name) if len(res_name) == 3 else res_name
            clusters[c_id].append(f"{res_name1}{res_num}")
            
        legend_html = "<div style='margin-top: 15px; display: flex; flex-wrap: wrap; justify-content: center; gap: 15px; font-family: sans-serif;'>"
        for c_id in sorted(clusters.keys()):
            color = palette[(c_id - 1) % len(palette)]
            # FIX: Removed indentation and multiline f-strings to prevent Markdown code-block rendering
            legend_html += (
                f"<div style='display: flex; align-items: center;'>"
                f"<span style='display: inline-block; width: 16px; height: 16px; background-color: {color}; border-radius: 50%; margin-right: 6px; border: 1px solid #aaa;'></span>"
                f"<span style='font-size: 14px; font-weight: bold; color: #333;'>Cluster {c_id}</span>"
                f"</div>"
            )
        legend_html += "</div>"

   
    color_map_json = json.dumps(color_map)
    safe_pdb_string = pdb_string.replace('`', '\\`')

    ngl_html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <script src="https://unpkg.com/ngl@2.0.0-dev.37/dist/ngl.js"></script>
        <style>
            html, body {{ margin: 0; padding: 0; width: 100%; height: 100%; overflow: hidden; background-color: #fafafa; }}
            #viewport {{ width: 100%; height: 500px; border-radius: 8px; box-shadow: inset 0 0 10px rgba(0,0,0,0.05); border: 1px solid #ddd; }}
        </style>
    </head>
    <body>
        <div id="viewport"></div>
        <script>
            document.addEventListener("DOMContentLoaded", function () {{
                var stage = new NGL.Stage("viewport", {{ backgroundColor: "#fafafa" }});
                
                var colorMap = {color_map_json};
                
                var schemeId = NGL.ColormakerRegistry.addScheme(function (params) {{
                    this.atomColor = function (atom) {{
                        if (colorMap[atom.resno]) {{
                            return parseInt(colorMap[atom.resno].replace("#", "0x"));
                        }}
                        return 0xDDDDDD; 
                    }};
                }});

                var pdbData = `{safe_pdb_string}`;
                var blob = new Blob([pdbData], {{ type: 'text/plain' }});
                
                stage.loadFile(blob, {{ ext: "pdb" }}).then(function (o) {{
                    o.addRepresentation("backbone", {{ 
                        color: schemeId,
                        radiusScale: 1.5
                    }});
                    o.autoView();
                }});

                window.addEventListener("resize", function(event){{
                    stage.handleResize();
                }}, false);
            }});
        </script>
    </body>
    </html>
    """

    components.html(ngl_html, height=520)
    st.markdown(legend_html, unsafe_allow_html=True)
def draw_pcn_plot_enhanced(labels, vis_labels, coords, adjacency, dist_matrix, residues, threshold, metrics):
    if len(labels) == 0 or coords.size == 0:
        st.warning("No residues available for visualization.")
        return

    N = len(labels)
    adj_np = np.array(adjacency)
    degrees = metrics['degree']
    deg_min = degrees.min() if len(degrees) else 0
    deg_ptp = np.ptp(degrees) if np.ptp(degrees) > 0 else 1
    
    st.markdown("""<div style="background: white; padding: 15px; border-radius: 10px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.08);"><h3 style="color: #1e3c72; margin-bottom: 10px;">Interactive Network Explorer</h3></div>""", unsafe_allow_html=True)

    filter_col1, filter_col2 = st.columns([1, 1])
    with filter_col1:
        view_mode = st.radio(
            "View Filter:", 
            [
                "Show All", 
                "Show Hubs Only", 
                "Hydrophobic Core", 
                "Closeness Centrality", 
                "Shortest Path (Betweenness)", 
                "Degree Viewer"
            ]
        )
        
        col_opt1, col_opt2 = st.columns(2)
        with col_opt1:
            strict_filter = st.checkbox("Hide Unmatched Nodes", value=True, help="If checked, unmatched nodes appear as faint 'ghosts'.")
        with col_opt2:
            show_legend = st.checkbox("Show Legend Colors", value=True)

  
    hub_percentile = 10
    custom_min_degree = 0
    highlight_communities = False
    centrality_threshold = 0.0 
    exact_degree_match = False
    path_indices = []
    
    
    focused_hub_idx = None
    focused_hub_coords = None

    
    with filter_col2:
        if view_mode == "Degree Viewer":
            max_deg = int(degrees.max()) if N > 0 else 0
            custom_min_degree = st.slider("Degree Value", 0, max_deg, 0)
            exact_degree_match = st.checkbox("Match Exact Degree Only", value=False)
            
        elif view_mode == "Show Hubs Only":
            hub_percentile = st.number_input("Top % (Percentile):", min_value=1, max_value=50, value=10, step=1)
            highlight_communities = st.checkbox("Highlight Communities", value=True)
            
        elif view_mode == "Closeness Centrality":
            
            max_val = np.max(metrics['closeness']) if len(metrics['closeness']) > 0 else 0.0
            st.info(f"Max Closeness in this network: **{max_val:.4f}**")
            
           
            centrality_threshold = st.number_input(
                "Minimum Closeness Centrality", 
                min_value=0.000, 
                max_value=1.000, 
                value=0.000, 
                step=0.001,
                format="%.3f"
            )

        elif view_mode == "Shortest Path (Betweenness)":
            st.markdown("Select residues to visualize path.")
            path_col1, path_col2 = st.columns(2)
            with path_col1:
                start_res = st.selectbox("Start", labels, key="path_start")
            with path_col2:
                end_res = st.selectbox("End", labels, index=len(labels)-1, key="path_end")
            
            if start_res and end_res:
                try:
                    G = nx.from_numpy_array(adj_np)
                    start_idx = labels.index(start_res)
                    end_idx = labels.index(end_res)
                    
                    if nx.has_path(G, start_idx, end_idx):
                        path_indices = nx.shortest_path(G, source=start_idx, target=end_idx)
                        path_str = " ➔ ".join([labels[i] for i in path_indices])
                        st.success(f"**Path ({len(path_indices)} steps):** {path_str}")
                    else:
                        st.error("No path exists between these residues.")
                except Exception as e:
                    st.error(f"Error: {e}")

   
    raw_threshold = np.percentile(degrees, 100 - hub_percentile)
    degree_threshold_hub = int(raw_threshold) 
    hub_indices_global = np.where(degrees >= degree_threshold_hub)[0]

    with filter_col2:
        if view_mode == "Show Hubs Only":
            st.info(f"Showing top {hub_percentile}% (Degree ≥ {degree_threshold_hub})")
            
           
            if len(hub_indices_global) > 0:
                st.markdown("---")
                hub_options = ["None (Overview)"] + [f"{labels[idx]} (Deg: {int(degrees[idx])})" for idx in hub_indices_global]
                selected_hub_option = st.selectbox(" Focus on specific Hub:", hub_options)
                
                if selected_hub_option != "None (Overview)":
                    selected_label = selected_hub_option.split(" (Deg:")[0]
                    focused_hub_idx = labels.index(selected_label)
                    focused_hub_coords = coords[focused_hub_idx]

  
    final_colors, final_sizes, final_text_labels = [], [], []

    for i in range(N):
        is_selected = False
        
        
        if view_mode == "Shortest Path (Betweenness)":
            if i in path_indices:
                is_selected = True
                if i == path_indices[0] or i == path_indices[-1]:
                    final_colors.append("red")
                    final_sizes.append(25)
                else:
                    final_colors.append("orange") 
                    final_sizes.append(15)
                final_text_labels.append(vis_labels[i])
                continue 
            else:
                is_selected = False

        
        else:
            if view_mode == "Show All": 
                is_selected = True
            elif view_mode == "Show Hubs Only" and i in hub_indices_global:
                if i == focused_hub_idx:
                    final_colors.append("yellow") 
                    final_sizes.append(35)        
                    final_text_labels.append(vis_labels[i])
                    continue 
                is_selected = True
            elif view_mode == "Degree Viewer":
                if exact_degree_match and degrees[i] == custom_min_degree: is_selected = True
                elif not exact_degree_match and degrees[i] >= custom_min_degree: is_selected = True
            
            # UPDATED: Absolute comparison for Closeness Centrality
            elif view_mode == "Closeness Centrality":
                 if metrics['closeness'][i] >= centrality_threshold:
                    is_selected = True
            
            elif view_mode == "Hydrophobic Core":
                name = residues[i].get_resname().strip().upper()
                if res_type_map.get(name, "UNK") == "hydrophobic": is_selected = True

        # Apply Standard Coloring
        if is_selected:
            if not show_legend:
                final_colors.append("#2196F3") 
            else:
                name = residues[i].get_resname().strip().upper()
                if name not in res_type_map: name = "UNK"
                base_hex = residue_type_colors.get(res_type_map[name], default_color)
                deg = degrees[i]
                norm_deg = (deg - deg_min) / deg_ptp
                opacity_val = 0.8 + (norm_deg * 0.2)
                h = base_hex.lstrip('#')
                rgb = tuple(int(h[x:x+2], 16) for x in (0, 2, 4))
                final_colors.append(f"rgba({rgb[0]},{rgb[1]},{rgb[2]},{float(opacity_val):.2f})")
            
            deg = degrees[i]
            norm_deg = (deg - deg_min) / deg_ptp
            final_sizes.append(15 + (norm_deg * 25))
            final_text_labels.append(vis_labels[i]) 
        else:
            # Ghost Nodes
            if strict_filter:
                final_colors.append("rgba(200, 200, 200, 0.05)")
                final_sizes.append(5)
                final_text_labels.append("") 
            else:
                final_colors.append("rgba(200, 200, 200, 0.15)")
                final_sizes.append(8)
                final_text_labels.append(vis_labels[i])
            
    communities = identify_hub_communities(adj_np, hub_indices_global, residues)
    
    fig = build_3d_figure_enhanced(
        labels, final_text_labels, coords, adj_np, dist_matrix,
        final_colors, final_sizes, residues,
        hub_indices_global, metrics,
        highlight_communities=(view_mode == "Show Hubs Only" and highlight_communities),
        view_mode=view_mode,
        path_indices=path_indices,
        show_legend=show_legend,
        focused_hub_coords=focused_hub_coords 
    )
    
    st.plotly_chart(fig, use_container_width=True, config={
        'toImageButtonOptions': {
            'format': 'png', 
            'filename': 'protein_network_hd',
            'height': 1200, 
            'width': 1600, 
            'scale': 6 
        },
        'displaylogo': False,           
        'modeBarButtons': [['toImage']]  
    })
    
    render_hub_analysis_panel(labels, residues, metrics, hub_indices_global, communities, adj_np)

def render_degree_betweenness_scatter(labels, metrics, residues, pdb_string):
    st.markdown("---")
    st.markdown("### Degree–Betweenness Analysis")
    
    col_input1, col_input2 = st.columns(2)
    with col_input1:
        deg_percentile = st.number_input("Top % for High Degree (Structural Hubs)", min_value=1, max_value=50, value=10, step=1)
    with col_input2:
        bet_percentile = st.number_input("Top % for High Betweenness (Bottlenecks)", min_value=1, max_value=50, value=5, step=1)

    degrees = metrics['degree']
    betweenness = metrics['betweenness']
    clustering = metrics['clustering']
    
    raw_deg_cutoff = np.percentile(degrees, 100 - deg_percentile)
    deg_cutoff = int(raw_deg_cutoff) 
    
    N_nodes = len(labels)
    normalization_factor = (N_nodes - 1) * (N_nodes - 2) if N_nodes > 2 else 1
    raw_betweenness = betweenness * normalization_factor
    raw_bet_cutoff = np.percentile(raw_betweenness, 100 - bet_percentile)
    
    region_map = {}
    for i, label in enumerate(labels):
        res_num = int(label.split('-')[-1])
        deg = degrees[i]
        bet = raw_betweenness[i]
        
        is_high_deg = deg >= deg_cutoff
        is_high_bet = bet >= raw_bet_cutoff
        
        if is_high_deg and is_high_bet:
            region_map[res_num] = "Global Critical"
        elif is_high_deg and not is_high_bet:
            region_map[res_num] = "Structural Hubs"
        elif not is_high_deg and is_high_bet:
            region_map[res_num] = "Bottlenecks"
        else:
            region_map[res_num] = "Peripheral"

    view_mode = st.radio("Visualisation Mode:", ["Scatter Plot Analysis", "3D Structure (NGL)"], horizontal=True, key="betweenness_view_toggle")
    
    if view_mode == "Scatter Plot Analysis":
        y_values_log = np.log10(raw_betweenness + 1)
        log_bet_cutoff = np.log10(raw_bet_cutoff + 1)
        
        hover_texts = []
        count_global, count_hub, count_bottleneck, count_peripheral = 0, 0, 0, 0
        classification_data = []
        
        for i, label in enumerate(labels):
            res_num = int(label.split('-')[-1])
            res_name = residues[i].get_resname().strip()
            region = region_map[res_num]
            
            
            if region == "Global Critical": count_global += 1
            elif region == "Structural Hubs": count_hub += 1
            elif region == "Bottlenecks": count_bottleneck += 1
            else: count_peripheral += 1
                
            hover_texts.append(f"<b>{label} ({res_name})</b><br>Region: {region}<br>Degree: {int(degrees[i])}<br>Raw Betweenness: {raw_betweenness[i]:.4f}<br>Clustering: {clustering[i]:.3f}")
            classification_data.append({"Residue Label": label, "Residue Name": res_name, "Region": region, "Degree": int(degrees[i]), "Raw Betweenness": round(raw_betweenness[i], 4), "Clustering Coeff": round(clustering[i], 4)})

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=degrees, y=y_values_log, mode='markers',
            marker=dict(size=12, color=clustering, colorscale='Viridis', showscale=True, colorbar=dict(title="Clustering Coeff"), line=dict(width=1, color='black'), opacity=0.9),
            text=hover_texts, hoverinfo='text', name='Residues'
        ))

        fig.add_vline(x=deg_cutoff, line_width=2, line_dash="dash", line_color="rgba(255, 0, 0, 0.6)", annotation_text=f"Deg={deg_cutoff}", annotation_position="top left", layer="below")
        fig.add_hline(y=log_bet_cutoff, line_width=2, line_dash="dash", line_color="rgba(255, 0, 0, 0.6)", annotation_text=f"Betw={raw_bet_cutoff:.3f}", annotation_position="bottom right", layer="below")

        fig.add_annotation(xref="paper", yref="paper", x=1, y=1, text=f"<b>Global Critical</b><br>(N={count_global})", showarrow=False, xanchor="right", yanchor="top", font=dict(size=14, color="#d32f2f"), bgcolor="rgba(255, 255, 255, 0.8)", bordercolor="#d32f2f", borderwidth=1)
        fig.add_annotation(xref="paper", yref="paper", x=0, y=1, text=f"<b>Bottlenecks</b><br>(N={count_bottleneck})", showarrow=False, xanchor="left", yanchor="top", font=dict(size=14, color="#fbc02d"), bgcolor="rgba(255, 255, 255, 0.8)", bordercolor="#fbc02d", borderwidth=1)
        fig.add_annotation(xref="paper", yref="paper", x=1, y=0, text=f"<b>Structural Hubs</b><br>(N={count_hub})", showarrow=False, xanchor="right", yanchor="bottom", font=dict(size=14, color="#f57c00"), bgcolor="rgba(255, 255, 255, 0.8)", bordercolor="#f57c00", borderwidth=1)
        fig.add_annotation(xref="paper", yref="paper", x=0, y=0, text=f"<b>Peripheral</b><br>(N={count_peripheral})", showarrow=False, xanchor="left", yanchor="bottom", font=dict(size=14, color="#757575"), bgcolor="rgba(255, 255, 255, 0.8)", bordercolor="#757575", borderwidth=1)
        
        fig.update_layout(title={'text': "<b>Degree vs Betweenness Centrality</b>", 'y': 0.95, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top'}, xaxis_title="Node Degree (Linear)", yaxis_title="Log10 (Raw Betweenness + 1)", height=700, width=900, template="plotly_white", showlegend=False, margin=dict(t=80, b=50, l=50, r=50))
        st.plotly_chart(fig, use_container_width=True, config={'toImageButtonOptions': {'format': 'png', 'height': 1200, 'width': 1600, 'scale': 6}, 'displaylogo': False, 'modeBarButtons': [['toImage']]})

        if classification_data:
            st.download_button(label="📥 Download Regional Classification Report (CSV)", data=pd.DataFrame(classification_data).to_csv(index=False).encode('utf-8'), file_name="degree_betweenness_classification.csv", mime="text/csv")
            
    else:
        col_ctrl1, col_ctrl2 = st.columns(2)
        with col_ctrl1:
            rep_choice = st.selectbox("Protein Representation:", ["Cartoon", "Backbone Trace", "Ball & Stick", "Surface"], key="bc_rep_choice")
        with col_ctrl2:
            highlight_choice = st.selectbox(
                "Highlight Region:", 
                ["None (Show All continuous)", "Global Critical", "Structural Hubs", "Bottlenecks", "Peripheral"],
                key="bc_highlight"
            )

        rep_map = {"Cartoon": "cartoon", "Backbone Trace": "backbone", "Ball & Stick": "ball+stick", "Surface": "surface"}
        
        bc_values = np.array(metrics.get('betweenness', [0]*len(labels)))
        log_bc = np.log1p(bc_values)
        log_min, log_max = log_bc.min(), log_bc.max()
        norm_bc = np.zeros_like(log_bc) if log_max - log_min == 0 else (log_bc - log_min) / (log_max - log_min)
        cmap = cm.get_cmap('coolwarm')
        
       
        region_colors = {
            "Global Critical": "#d32f2f", 
            "Structural Hubs": "#f57c00", 
            "Bottlenecks": "#fbc02d",     
            "Peripheral": "#1A036500"       
        }

        color_map = {}
        for idx, label in enumerate(labels):
            res_num = int(label.split('-')[-1])
            region = region_map[res_num]
            
            if highlight_choice == "None (Show All continuous)":
                rgba = cmap(norm_bc[idx])
                color_map[res_num] = mcolors.to_hex(rgba)
            else:
                if region == highlight_choice:
                    color_map[res_num] = region_colors[region]
                else:
                    color_map[res_num] = "#E0E0E0"

        components.html(generate_ngl_html(pdb_string, color_map, rep_style=rep_map[rep_choice]), height=520)

def load_structure_from_upload(uploaded_file, progress=None, progress_label=None):
    
    uploaded_file.seek(0)
    
    if progress is not None:
        progress.progress(5)
        if progress_label is not None:
            progress_label.text("5% complete — starting to read file")
    
    text = uploaded_file.read().decode("utf-8")
    
    if progress is not None:
        progress.progress(25)
        if progress_label is not None:
            progress_label.text("25% complete — file read into memory")
            
    structure = PDBParser(QUIET=True).get_structure("uploaded", StringIO(text))
    
    if progress is not None:
        progress.progress(45)
        if progress_label is not None:
            progress_label.text("45% complete — parsed structure")
    return structure


def load_demo_structure(pdb_id, progress=None, progress_label=None):
    local_demo_path = os.path.join(os.path.dirname(__file__), "demo_data", f"{pdb_id}.pdb")
    if os.path.exists(local_demo_path):
        if progress is not None:
            progress.progress(50)
            if progress_label is not None:
                progress_label.text("50% complete — loading local demo file")
        return PDBParser(QUIET=True).get_structure(pdb_id.lower(), local_demo_path)

    pdbl = PDBList()
    temp = tempfile.gettempdir()
    if progress is not None:
        progress.progress(30)
        if progress_label is not None:
            progress_label.text("30% complete — fetching demo from RCSB")
    path = pdbl.retrieve_pdb_file(pdb_id, file_format="pdb", pdir=temp)
    if progress is not None:
        progress.progress(70)
        if progress_label is not None:
            progress_label.text("70% complete — demo file downloaded")
    structure = PDBParser(QUIET=True).get_structure(pdb_id.lower(), path)
    if progress is not None:
        progress.progress(95)
        if progress_label is not None:
            progress_label.text("95% complete — parsing demo file")
    return structure


def process_and_render_pcn(structure, pdb_string, rep_mode, threshold, model_choice=None, chain_choice=None, progress_bar=None, progress_label=None):
    if not structure:
        st.error("Structure file could not be parsed. Please check if it's a valid PDB.")
        return

    model_ids = list(range(1, len(structure) + 1))
    
    if model_choice is None:
        if len(model_ids) > 1:
            st.info(f"Multi-model structure detected ({len(model_ids)} models).")
            model_choice = st.selectbox("Select NMR Model", model_ids)
        else:
            st.info("Single structural model detected (no NMR ensemble). Using Model 1.")
            model_choice = 1
    
    chains = list(structure[model_choice - 1].get_chains())
    chain_ids = [c.id for c in chains]
    
    if chain_choice is None:
        if len(chain_ids) > 1:
            st.info(f"Multiple chains detected ({len(chain_ids)} chains).")
            chain_choice = st.selectbox("Select Chain", chain_ids)
        elif len(chain_ids) == 1:
            st.info(f"Single chain detected (Chain {chain_ids[0]}).")
            chain_choice = chain_ids[0]
        else:
            st.error("No chains found in this structure.")
            return

    if progress_bar:
        progress_bar.progress(55)
        progress_label.text("55% complete — preparing computation")
    
   
    adj_df, dist_df, labels, vis_labels, coords, residues, metrics = compute_pcn_df(
        structure, model_choice, chain_choice, threshold, rep_mode,
        progress=progress_bar, 
        progress_label=progress_label
    )

    if labels is None or len(labels) == 0:
        st.error("No valid C-alpha atoms found in the selected chain.")
        return

    num_nodes = len(labels)
    num_edges = int(np.sum(adj_df.values) // 2) if num_nodes else 0
    density = num_edges / (num_nodes * (num_nodes - 1) / 2) if num_nodes > 1 else 0

    # --- EXTRACT METADATA ---
    header = structure.header
    
    def get_compound_info(key, default="Unknown"):
        if 'compound' in header and header['compound']:
            first_mol = next(iter(header['compound'].values()), {})
            return first_mol.get(key, default)
        return default

    def get_source_info(key, default="Unknown"):
        if 'source' in header and header['source']:
            first_src = next(iter(header['source'].values()), {})
            return first_src.get(key, default)
        return default

    meta_pdb_id = header.get('idcode', 'USER UPLOAD').upper()
    meta_name = get_compound_info('molecule', 'Unknown Protein').capitalize()
    meta_class = header.get('head', 'Unclassified').capitalize()
    meta_organism = get_source_info('organism_scientific', 'Unknown Organism').capitalize()
    meta_method = header.get('structure_method', 'unknown').upper()
    
    is_engineered = get_compound_info('engineered', '').lower() == 'yes' or 'mutation' in get_compound_info('other_details', '').lower()
    meta_engineered = "Yes" if is_engineered else "No"
    
    if len(meta_pdb_id) == 4 and meta_pdb_id != "USER UPLOAD":
        pdb_display = f'<a href="https://www.rcsb.org/structure/{meta_pdb_id}" target="_blank" style="text-decoration:none; color:#1e3c72; border-bottom: 2px solid #1e3c72;">{meta_pdb_id} ↗</a>'
    else:
        pdb_display = meta_pdb_id

    # --- STRUCTURE OVERVIEW ---
    structure_card_html = f"""
    <div style="background: white; padding: 30px; border-radius: 12px; margin: 30px 0; box-shadow: 0 4px 12px rgba(0,0,0,0.1); border-left: 6px solid #2a5298;">
        <h3 style="color: #1e3c72; margin-top: 0; margin-bottom: 25px; font-size: 28px; font-weight: 700; border-bottom: 2px solid #eee; padding-bottom: 15px;">
            Structure Overview
        </h3>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 40px; font-size: 20px; line-height: 1.8; color: #333;">
            <div>
                <p style="margin: 10px 0;"><b>PDB ID:</b> {pdb_display}</p>
                <p style="margin: 10px 0;"><b>Protein:</b> {meta_name}</p>
                <p style="margin: 10px 0;"><b>Functional Class:</b> {meta_class}</p>
                <p style="margin: 10px 0;"><b>Organism:</b> <i>{meta_organism}</i></p>
            </div>
            <div>
                <p style="margin: 10px 0;"><b>Chain Analyzed:</b> <span style="background: #fff3e0; padding: 4px 12px; border-radius: 6px; color: #e65100; font-weight:bold;">{chain_choice}</span></p>
                <p style="margin: 10px 0;"><b>Method:</b> {meta_method}</p>
                <p style="margin: 10px 0;"><b>Engineered:</b> {meta_engineered}</p>
            </div>
        </div>
    </div>
    """
    st.markdown(structure_card_html, unsafe_allow_html=True)

    # --- NETWORK SUMMARY ---
    st.markdown(
        f"""
        <div style="background: white; padding: 20px; border-radius: 10px; margin: 20px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.08);">
            <h3 style="color: #1e3c72; margin-bottom: 15px;">Network Summary</h3>
            <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px;">
                <div style="text-align: center; padding: 15px; background: #f0f4f8; border-radius: 8px;">
                    <div style="margin-bottom: 10px;">
                        <svg width="40" height="40" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                            <circle cx="12" cy="12" r="10" stroke="#2a5298" stroke-width="2" fill="none"/>
                            <circle cx="12" cy="12" r="4" fill="#2a5298"/>
                        </svg>
                    </div>
                    <div style="font-size: 32px; font-weight: 700; color: #2a5298;">{num_nodes}</div>
                    <div style="font-size: 14px; color: #666;">Nodes</div>
                </div>
                <div style="text-align: center; padding: 15px; background: #f0f4f8; border-radius: 8px;">
                     <div style="margin-bottom: 10px;">
                        <svg width="40" height="40" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                            <line x1="4" y1="20" x2="20" y2="4" stroke="#2a5298" stroke-width="2"/>
                            <circle cx="4" cy="20" r="3" fill="#2a5298"/>
                            <circle cx="20" cy="4" r="3" fill="#2a5298"/>
                        </svg>
                    </div>
                    <div style="font-size: 32px; font-weight: 700; color: #2a5298;">{num_edges}</div>
                    <div style="font-size: 14px; color: #666;">Edges</div>
                </div>
                <div style="text-align: center; padding: 15px; background: #f0f4f8; border-radius: 8px;">
                    <div style="margin-bottom: 10px;">
                        <svg width="40" height="40" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                            <circle cx="6" cy="6" r="3" fill="#2a5298" />
                            <circle cx="18" cy="6" r="3" fill="#2a5298" />
                            <circle cx="12" cy="12" r="3" fill="#2a5298" />
                            <circle cx="6" cy="18" r="3" fill="#2a5298" />
                            <circle cx="18" cy="18" r="3" fill="#2a5298" />
                            <path d="M6 6 L18 18 M18 6 L6 18 M6 6 L18 6 M6 18 L18 18" stroke="#2a5298" stroke-width="2" />
                        </svg>
                    </div>
                    <div style="font-size: 32px; font-weight: 700; color: #2a5298;">{density:.4f}</div>
                    <div style="font-size: 14px; color: #666;">Graph Density</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # --- MATRICES ---
    st.markdown("<h3 style='color: #1e3c72; margin-top: 30px;'>Distance Matrix (Preview)</h3>", unsafe_allow_html=True)
    st.dataframe(dist_df.iloc[:10, :10])
    st.download_button("Download Distance Matrix (CSV)", dist_df.to_csv().encode(), "distance.csv")
    
    with st.expander("Values Visualization (Heatmap)", expanded=True):
        render_distance_heatmap(dist_df)

    st.markdown("<h3 style='color: #1e3c72; margin-top: 30px;'>Adjacency Matrix (Preview)</h3>", unsafe_allow_html=True)
    st.dataframe(adj_df.iloc[:10, :10])
    st.download_button("Download Adjacency Matrix (CSV)", adj_df.to_csv().encode(), "adjacency.csv")
    
    with st.expander("Binary Visualization (Adjacency Map)", expanded=True):
        render_adjacency_heatmap(adj_df)
    
    # --- DOWNLOADS ---
    st.markdown("<h3 style='color: #1e3c72; margin-top: 30px;'>Download Network Files</h3>", unsafe_allow_html=True)
    
    edges_sif = []
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if adj_df.values[i][j] == 1:
                edges_sif.append(f"{labels[i]} pp {labels[j]}")
    sif_text = "\n".join(edges_sif)
    
    edges_txt = []
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if adj_df.values[i][j] == 1:
                edges_txt.append(f"{labels[i]} {labels[j]}")
    txt_text = "\n".join(edges_txt)

    col_sif, col_edge = st.columns(2)
    with col_sif:
        st.markdown("**SIF (Cytoscape)**\n*Residue_A pp Residue_B*")
        st.download_button("Download SIF", sif_text, "network.sif")
    with col_edge:
        st.markdown("**Edge List (Text)**\n*Residue_A Residue_B*")
        st.download_button("Download Edge List", txt_text, "edges.txt")

    # --- MAIN VISUALIZATION ---
    draw_pcn_plot_enhanced(labels, vis_labels, coords, adj_df.values, dist_df.values, residues, threshold, metrics)
    
    # --- NEW: LEIDEN COMMUNITY DETECTION ---
    render_community_detection_module(structure, model_choice, chain_choice, adj_df.values, labels, coords, residues, pdb_string)

    # --- NEW: DEGREE vs BETWEENNESS SCATTER ---
    render_degree_betweenness_scatter(labels, metrics, residues, pdb_string)
    
    # --- DEGREE DISTRIBUTION ---
    st.markdown("<h3 style='color: #1e3c72; margin-top: 30px;'>Residue Degree Distribution</h3>", unsafe_allow_html=True)
    if len(labels) > 0:
        degree_counts = pd.Series(metrics['degree']).value_counts().sort_index()
        fig_hist = go.Figure(go.Bar(
            x=degree_counts.index, 
            y=degree_counts.values,
            text=degree_counts.values,
            textposition='outside',
            marker_color="#2a5298"
        ))
        
        # UPDATED: Added axis titles to the layout
        fig_hist.update_layout(
            height=360, 
            margin=dict(l=0, r=0, t=20, b=0),
            xaxis_title="Degree",
            yaxis_title="Number of residues"
        )
        
        
        st.plotly_chart(fig_hist, use_container_width=True, config=high_res_config)
    

   
    st.markdown("---")
    st.markdown("<h3 style='color: #1e3c72; margin-top: 20px;'>Full Statistical Report (Preview)</h3>", unsafe_allow_html=True)
    
    full_stats_data = []
    for i in range(num_nodes):
        r_name = residues[i].get_resname().strip().upper()
        r_type = res_type_map.get(r_name, "Unknown")
        bc_val = metrics['betweenness'][i] if 'betweenness' in metrics else 0.0
        
        full_stats_data.append({
            "Residue_No": labels[i],
            "Type": r_type,
            "Degree": int(metrics['degree'][i]),
            "Closeness Centrality": round(metrics['closeness'][i], 4),
            "Betweenness": round(bc_val, 4),
            "Clustering Coefficient": round(metrics['clustering'][i], 4)
        })
        
    full_stats_df = pd.DataFrame(full_stats_data)
    
    st.dataframe(
        full_stats_df.head(50), 
        use_container_width=True, 
        height=800
    )
    
    # CSV Download
    full_csv = full_stats_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Complete Statistics (CSV)",
        data=full_csv,
        file_name="full_residue_statistics.csv",
        mime="text/csv"
    )

    if progress_bar:
        progress_bar.progress(100)
        progress_label.text("Processing complete")

# -----------------------------------------------------------------------------
# MAIN LAYOUT
# -----------------------------------------------------------------------------

st.markdown("""
<div class="main-header">
    <h1 style='text-align:center; font-size:52px; color: white; margin: 0; font-weight: 700; letter-spacing: -0.5px;'>
        Protein Contact Network Explorer
    </h1>
    <p style='text-align:center; font-size:18px; color: #e3f2fd; margin-top: 10px; font-weight: 400;'>
        Analyze residue contacts • Visualize networks • Advanced hub analysis • Export data
    </p>
</div>
""", unsafe_allow_html=True)

tab_analysis, tab_help, tab_about = st.tabs(["🧪 Analysis Tool", "❓ Help Guide", "🔍 About"])

with tab_analysis:
    left_col, mid_col, right_col = st.columns([2.5, 4, 3.5])

    with left_col:
        st.markdown("""
        <div class="section-box">
            <div style="text-align:center;">
                <svg width="40" height="40" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                    <path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20" stroke="#2a5298" stroke-width="2" fill="none" stroke-linecap="round"/>
                    <path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z" stroke="#2a5298" stroke-width="2" fill="none" stroke-linecap="round"/>
                </svg>
            </div>
            <h3 style="text-align:center; color:#1e3c72;">Try with Demo</h3>
        </div>
        """, unsafe_allow_html=True)
        st.write("Try a preloaded PDB to see example networks quickly.")
        demo_choices = ["None", "1CRN", "1UBQ", "4HHB"]
        demo_selection = st.selectbox("Demo protein", demo_choices, index=0, key="demo_selection")

    with mid_col:
        st.markdown("""
        <div class="section-box" style="text-align: center; min-height: 200px; display: flex; flex-direction: column; align-items: center; justify-content: center;">
            <div style="margin-bottom: 10px;">
                <svg width="80" height="80" viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg" style="display: block; margin: 0 auto;">
                    <line x1="50" y1="20" x2="20" y2="75" stroke="#2a5298" stroke-width="4" stroke-linecap="round"/>
                    <line x1="50" y1="20" x2="80" y2="75" stroke="#2a5298" stroke-width="4" stroke-linecap="round"/>
                    <line x1="20" y1="75" x2="80" y2="75" stroke="#2a5298" stroke-width="4" stroke-linecap="round"/>
                    <circle cx="50" cy="20" r="10" fill="#FF5252" stroke="white" stroke-width="2"/>
                    <circle cx="20" cy="75" r="10" fill="#2196F3" stroke="white" stroke-width="2"/>
                    <circle cx="80" cy="75" r="10" fill="#4CAF50" stroke="white" stroke-width="2"/>
                </svg>
            </div>
            <h3 style="text-align:center; color: #1e3c72; margin: 0; padding: 0;">
                Upload PDB Files
            </h3>
        </div>
        """, unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Upload a PDB file", type=["pdb"], label_visibility="collapsed")

    with right_col:
        st.markdown("""
        <div class="section-box">
            <div style="text-align:center;">
                <svg width="40" height="40" viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
                    <line x1="30" y1="70" x2="70" y2="30" stroke="#000" stroke-width="3"/>
                    <circle cx="30" cy="70" r="18" fill="#FF5252"/>
                    <circle cx="70" cy="30" r="18" fill="#2196F3"/>
                </svg>
            </div>
            <h3 style="text-align:center; color:#1e3c72; font-size: 25px; margin-top: 5px;">Network Selection & Thresholds</h3>
        </div>
        """, unsafe_allow_html=True)
        
        rep_mode = st.radio("Node Representation:", ["C-alpha", "C-beta", "Side-chain Centroid"])
        
        default_thresholds = {
            "C-alpha": 7.5, 
            "C-beta": 7.0, 
            "Side-chain Centroid": 7.0
        }
        
      
        threshold = st.number_input(
            "Contact threshold (Å)", 
            min_value=1.0, max_value=20.0, 
            value=default_thresholds[rep_mode], 
            step=0.1, 
            key=f"thresh_slider_{rep_mode}"
        )

    st.divider()

    if uploaded_file:
        pdb_string = uploaded_file.getvalue().decode("utf-8")
        st.session_state["is_demo"] = False
        progress_bar = mid_col.progress(0)
        progress_label = mid_col.empty()
        progress_label.text("0% complete — waiting to start")
        with st.spinner("Processing uploaded PDB file..."):
            structure = load_structure_from_upload(uploaded_file, progress_bar, progress_label)
           
            process_and_render_pcn(structure, pdb_string, rep_mode, threshold, progress_bar=progress_bar, progress_label=progress_label)

    else:
        st.session_state["is_demo"] = (demo_selection != "None")
        demo_active = st.session_state.get("is_demo", False)
        demo_id = st.session_state.get("demo_selection", demo_selection)

        if demo_active and demo_id and demo_id != "None":
            progress_bar = mid_col.progress(0)
            progress_label = mid_col.empty()
            with st.spinner(f"Loading demo {demo_id}..."):
                structure = load_demo_structure(demo_id, progress_bar, progress_label)
                from Bio.PDB import PDBList
                import tempfile
                local_demo_path = os.path.join(os.path.dirname(__file__), "demo_data", f"{demo_id}.pdb")
                if os.path.exists(local_demo_path):
                    demo_path = local_demo_path
                else:
                    pdbl = PDBList()
                    temp = tempfile.gettempdir()
                    demo_path = pdbl.retrieve_pdb_file(demo_id, file_format="pdb", pdir=temp)
                    
                with open(demo_path, 'r') as f:
                    pdb_string = f.read()
                
                # UPDATED: Added rep_mode and threshold here as well
                process_and_render_pcn(structure, pdb_string, rep_mode, threshold, progress_bar=progress_bar, progress_label=progress_label)
        else:
            st.info("Upload a PDB file or try the demo to begin.")

# --- TAB 2: Help ---
with tab_help:
    st.markdown("""
    <div style="background: white; padding: 25px; border-radius: 10px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.08); border-left: 5px solid #1e3c72;">
        <h3 style="color: #1e3c72; margin-top: 0;">How to Use PCNE</h3>
        <p style="font-size: 16px;">Follow these steps to construct, validate, and analyze your protein contact networks.</p>
    </div>
    """, unsafe_allow_html=True)

    col_guide_1, col_guide_2 = st.columns(2)

    with col_guide_1:
        st.markdown("""
        #### 1. Load your protein
        Upload a PDB file using the file uploader, or select a demo protein from the dropdown to try the tool instantly. 
        * **NMR Structures:** Select the specific model number you wish to analyse.
        * **Multi-chain Structures:** Select the individual chain of interest.
        
        *Note: Analysis is executed on a single chain–model combination at a time.*

        ---
        #### 2. Choose node representation
        Select how amino acid residues are represented as network nodes:
        * **Cα** — Alpha carbon positions only. Computationally fast and standard for backbone topology analyses.
        * **Cβ** — Beta carbon positions. Captures side-chain orientation tracking.
        * **Side-chain Centroid** — Mean spatial position of all non-hydrogen side-chain heavy atoms. Most chemically representative.
        """)

    with col_guide_2:
        st.markdown("""
        #### 3. Set the contact threshold
        The threshold ($r_c$ in Å) defines the maximum Euclidean distance between reference coordinates for a network edge to be drawn. 
        * **Defaults:** Automatically calibrated per node representation selection.
        * **Logic:** Pairs with distance $\leq$ threshold = **1** (edge present), else **0** (no edge).

        ---
        #### 4. Export results
        PCNE generates standard structural and graph-theoretic outputs available for downstream research:
        * **SIF Network Export:** Download as a `.sif` file for instant graph rendering and style mapping inside Cytoscape.
        * **Matrix Downloads:** Download the raw pairwise Distance Matrix and binary Adjacency Matrix in standard CSV formats.
        """)

    st.markdown("---")
    
    st.markdown("""
    #### 5. Explore the network
    Use the dedicated view filters to inspect diverse topological and biochemical components of your target protein structure.
    
    * **View Filters:**
        * **Show All:** Displays the complete network graph.
        * **Show Hubs Only:** Highlights top connected nodes. Use the focus selection menu to isolate and zoom into specific core structural hubs.
        * **Closeness Centrality:** Input a minimum threshold value to isolate nodes with high global reachability metrics.
        * **Hydrophobic Core:** Highlights all hydrophobic residues to isolate packed interior network motifs.
        * **Degree Viewer:** Highlights individual residues colored continuously by local connectivity profile.
        * **Betweenness Viewer:** Maps information bottlenecks based on calculated shortest-path traffic properties.
    * **Biochemical Color Coding:**
        * <span style="color:#D94E1E"><b>● Hydrophobic</b></span>
        * <span style="color:#003B6F"><b>● Polar</b></span>
        * <span style="color:#007A55"><b>● Positive</b></span>
        * <span style="color:#B32630"><b>● Negative</b></span>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    #### 6. Degree–Betweenness Analysis
    This scatter plot partitions all system residues into four distinct functional roles based on your custom percentile cutoffs.
    
    * **Analytical Axes:**
        * **X-Axis (Node Degree):** Measures local connectivity (number of structural contacts).
        * **Y-Axis (Betweenness Centrality):** Measures global communication importance ($\log_{10}$-transformed raw values).
    * **Functional Network Regions:**
        * **Global Critical (Top-Right):** High degree & high betweenness. Key residues essential for both local packing stability and long-range allosteric communication.
        * **Structural Hubs (Bottom-Right):** High degree & low betweenness. Densely packed interior clusters responsible for structural stability.
        * **Bottlenecks (Top-Left):** Low degree & high betweenness. Critical dynamic bridges connecting distinct modules or domains.
        * **Peripheral (Bottom-Left):** Low degree & low betweenness. Highly flexible or solvent-exposed surface regions.
    """)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### Common Errors and Fixes")

    with st.expander("Network is disconnected", expanded=False):
        st.markdown("""
        **Symptom:**
        A warning banner appears indicating that the largest connected component contains less than 95% of total system residues.
        
        **Fix:**
        Incrementally increase the contact threshold ($r_c$) by 0.5–1.0 Å steps using the slider interface to close isolated network gaps until the structural warning banner disappears.
        """)

    with st.expander("No structure loaded", expanded=False):
        st.markdown("""
        **Symptom:**
        The coordinates cannot be parsed or errors are thrown during file initialization steps.
        
        **Fix:**
        Ensure your coordinate file strictly matches standard compliance formats for PDB files (`.pdb`). Files structured in mmCIF, PDBx, or alternative file configurations are not currently supported by the input stream.
        """)

    with st.expander("DSSP failed or community composition not showing", expanded=False):
        st.markdown("""
        **Symptom:**
        Community assignments are complete, but secondary structure composition color breakdown bars remain blank.
        
        **Fix:**
        This occurs when local structural validation issues prevent the secondary structure assignment loops. Verify that your source PDB file contains complete, non-truncated `ATOM` records. Try downloading a clean copy of the coordinates directly from the RCSB Protein Data Bank.
        """)

    with st.expander("Glycine residues missing from Cβ or Centroid network", expanded=False):
        st.markdown("""
        **Symptom:**
        Glycine positions do not shift when switching from Cα reference configurations.
        
        **Context:**
        This is expected chemical behavior. Because Glycine has no side-chain heavy atoms beyond its alpha carbon, the script engine automatically applies a fallback routing to its native Cα coordinate reference across both Cβ and Side-chain Centroid network modes.
        """)

    with st.expander("3D viewer not rendering", expanded=False):
        st.markdown("""
        **Symptom:**
        The embedded HTML WebGL frame area remains white, blank, or frozen.
        
        **Fix:**
        Refresh your current browser session and reload the structure. If the WebGL graphic context hangs repeatedly, ensure hardware acceleration is toggled on inside your browser settings—Google Chrome and Mozilla Firefox are recommended.
        """)

    st.markdown("---")
    st.markdown("""
    #### Contact
    For technical queries, data issues, or feature requests, contact the corresponding author at **i_arnoldemerson@yahoo.com** or open a formal tracking issue directly on the official source repository at **https://github.com/akhuuu2303**
    """)

with tab_about:
    st.markdown("""
    <div style="text-align: center; margin-bottom: 30px;">
        <h2 style="color: #1e3c72; font-weight: 800;">About PCNE</h2>
        <p style="font-size: 18px; color: #555;">Protein Contact Network Explorer</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="background: white; padding: 22px; border-radius: 10px; border-left: 5px solid #1e3c72; box-shadow: 0 2px 4px rgba(0,0,0,0.08); margin-bottom: 25px;">
        <h4 style="margin-top: 0; color: #1e3c72; font-weight: 700;">Institutional Affiliation</h4>
        <p style="margin-bottom: 0; line-height: 1.6;">
            PCNE (Protein Contact Network Explorer) was developed at the <b>Bioinformatics Programming Laboratory</b>, 
            Department of Bioscience, School of Bio Sciences and Technology, <b>Vellore Institute of Technology (VIT)</b>, Vellore, Tamil Nadu, India.
        </p>
        <p style="margin-top: 10px; margin-bottom: 0; line-height: 1.6;">
            The tool was engineered to make advanced protein contact network topological analysis universally accessible for both structural biology 
            research frameworks and bioinformatics pedagogy, effectively mitigating the historical requirement for fragmented workflows across multiple standalone platforms.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # --- DEVELOPERS SECTION ---
    st.markdown("###  Research & Development Team")
    
    col_dev1, col_dev2 = st.columns(2)
    
    with col_dev1:
        st.markdown("""
        <div style="background: #f8f9fa; padding: 20px; border-radius: 8px; border: 1px solid #e9ecef; height: 180px;">
            <h5 style="color: #1e3c72; margin-top: 0; font-weight: 700;">Akhurath Ganapathy</h5>
            <p style="font-size: 14px; color: #6c757d; margin-bottom: 15px;">Researcher & Developer<br>Vellore Institute of Technology</p>
            <a href="https://github.com/akhuuu2303" target="_blank" style="text-decoration: none; margin-right: 15px; color: #24292e; font-weight: 600;">💻 GitHub</a>
            <a href="mailto:akhurath2303@gmail.com" style="text-decoration: none; color: #d93025; font-weight: 600;"> Email</a>
        </div>
        """, unsafe_allow_html=True)

    with col_dev2:
        st.markdown("""
        <div style="background: #f8f9fa; padding: 20px; border-radius: 8px; border: 1px solid #e9ecef; height: 180px;">
            <h5 style="color: #1e3c72; margin-top: 0; font-weight: 700;">Sanjana V. Krishnan</h5>
            <p style="font-size: 14px; color: #6c757d; margin-bottom: 15px;">Researcher<br>Vellore Institute of Technology</p>
            <br>
            <a href="mailto:sjana.vijay2024@vitstudent.ac.in" style="text-decoration: none; color: #d93025; font-weight: 600;">✉️ Email</a>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background: #f4f7f6; padding: 20px; border-radius: 8px; border-left: 5px solid #007A55;">
        <h5 style="color: #007A55; margin-top: 0; font-weight: 700;">Professor Arnold Emerson Isaac</h5>
        <p style="font-size: 14px; color: #495057; margin-bottom: 10px;">
            <b>Corresponding Author</b><br>
            Bioinformatics Programming Laboratory, SBST<br>
            Vellore Institute of Technology, Vellore, India
        </p>
        <a href="mailto:i_arnoldemerson@yahoo.com" style="text-decoration: none; margin-right: 20px; color: #d93025; font-weight: 600;"> i_arnoldemerson@yahoo.com</a>
        <a href="https://orcid.org/0000-0003-4212-0927" target="_blank" style="text-decoration: none; color: #A6CE39; font-weight: 600;"> ORCID: 0000-0003-4212-0927</a>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # --- PUBLICATION & CITATION ---
    st.markdown("###  Publication & Citation Reference")
    st.markdown(
        "This interactive analytical framework accompanies the following primary research manuscript:\n\n"
        "> **Protein Contact Network Explorer: Topological Analysis of Protein Structures**\n"
        "> *Akhurath Ganapathy, Sanjana V. Krishnan, and Arnold Emerson Isaac*\n"
        "> **Frontiers in Bioinformatics**, 2026 — *Currently under peer review*"
    )
    
    st.markdown("#### **Cite This Tool**")
    st.markdown(
        "If you implement the PCNE platform, graph construction logic, or secondary structure profiling subsets within your research "
        "pipelines, please formally reference and cite the manuscript listed above."
    )
    
    with st.expander(" View BibTeX Format for Reference Citations"):
        st.code("""@article{ganapathy2026pcne,
  title   = {Protein Contact Network Explorer: Topological Analysis of Protein Structures},
  author  = {Ganapathy, Akhurath and Krishnan, Sanjana V. and Isaac, Arnold Emerson},
  journal = {Frontiers in Bioinformatics},
  year    = {2026},
  note    = {Under Review}
}""", language="bibtex")

    st.markdown("---")

    # --- ACKNOWLEDGEMENTS ---
    st.markdown("###  Acknowledgements")
    st.markdown(
        "The authors express sincere gratitude to **Vellore Institute of Technology (VIT), Vellore**, for provisioning the essential "
        "computational infrastructure, database accesses, and laboratory resources necessary to fully execute this research work. "
        "No explicit financial support or structural grants were received for the primary research, application development, or open publication pipelines of this article."
    )

    st.markdown("""
    <div style="margin-top: 50px; text-align: center; color: #aaas; font-size: 13px; letter-spacing: 0.5px;">
        <hr style="border: 0; border-top: 1px solid #eee; margin-bottom: 15px;">
        Bioinformatics Programming Laboratory • School of Bio Sciences and Technology • VIT Vellore (2026)
    </div>
    """, unsafe_allow_html=True)
