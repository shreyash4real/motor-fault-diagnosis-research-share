import json
import os
from pathlib import Path

def generate_visjs_html(output_path="pipeline_visjs_graph.html"):
    # Community Colors based on graphify theme
    COLORS = {
        0: "#E15759", # Red - Raw Data
        1: "#F28E2B", # Orange - Manifests/Splits
        2: "#4E79A7", # Blue - Scripts
        3: "#59A14F", # Green - Processed Data
        4: "#B07AA1", # Purple - Output Artifacts
    }

    COMMUNITIES = {
        0: "Raw Data",
        1: "Manifests",
        2: "Scripts",
        3: "Processed Hub",
        4: "Outputs & Artifacts"
    }

    raw_nodes_data = [
        {"id": "dataset", "label": "Dataset", "c": 0, "file": "Dataset/Electric/Motor-2", "type": "Folder"},
        
        {"id": "setup_pc", "label": "setup_pc.py", "c": 2, "file": "Scripts/setup_pc.py", "type": "Python Script"},
        {"id": "manifest", "label": "manifest.csv", "c": 1, "file": "Outputs/main/manifest.csv", "type": "CSV"},
        
        {"id": "validate_dataset", "label": "validate_dataset.py", "c": 2, "file": "Scripts/validate_dataset.py", "type": "Python Script"},
        {"id": "validated_manifest", "label": "validated_manifest.csv", "c": 1, "file": "Outputs/main/validated_manifest.csv", "type": "CSV"},
        
        {"id": "denoise_all", "label": "denoise_all.py", "c": 2, "file": "Scripts/denoise_all.py", "type": "Python Script"},
        {"id": "denoised", "label": "Denoised", "c": 3, "file": "Denoised/Motor-2", "type": "Folder"},
        
        {"id": "generate_splits", "label": "generate_splits_v2.py", "c": 2, "file": "Scripts_4/generate_splits_v2.py", "type": "Python Script"},
        {"id": "splits", "label": "splits.csv", "c": 1, "file": "Outputs/4class/v2_speed_strat/splits.csv", "type": "CSV"},
        
        # Precomputes
        {"id": "precompute_stft", "label": "precompute_stft_v2.py", "c": 2, "file": "Scripts_4/precompute_stft_v2.py", "type": "Python Script"},
        {"id": "stft_features", "label": "STFT Features/Images", "c": 4, "file": "Outputs/4class/v2_speed_strat/features", "type": "Tensors & PNGs"},
        
        {"id": "precompute_dwt", "label": "precompute_dwt_v1.py", "c": 2, "file": "Scripts_4/precompute_dwt_v1.py", "type": "Python Script"},
        {"id": "dwt_features", "label": "DWT Features/Images", "c": 4, "file": "Outputs/4class/v2_speed_strat/dwt", "type": "Tensors & PNGs"},
        
        {"id": "precompute_envelope", "label": "precompute_envelope_v1.py", "c": 2, "file": "Scripts_4/precompute_envelope_v1.py", "type": "Python Script"},
        {"id": "env_features", "label": "Envelope Features/Images", "c": 4, "file": "Outputs/4class/v2_speed_strat/envelope", "type": "Tensors & PNGs"},
        
        {"id": "precompute_env_stft", "label": "precompute_envelope_stft_v1.py", "c": 2, "file": "Scripts_4/precompute_envelope_stft_v1.py", "type": "Python Script"},
        {"id": "env_stft_features", "label": "Envelope STFT Features", "c": 4, "file": "Outputs/4class/v2_speed_strat/envelope_stft", "type": "Tensors & PNGs"},

        # Training
        {"id": "train_stft", "label": "train_stft_cnn_*.py", "c": 2, "file": "Scripts_4/train_stft_cnn_...", "type": "Python Script"},
        {"id": "train_dwt", "label": "train_dwt_cnn.py", "c": 2, "file": "Scripts_4/train_dwt_cnn.py", "type": "Python Script"},
        {"id": "train_env", "label": "train_envelope_cnn_*.py", "c": 2, "file": "Scripts_4/train_envelope_cnn_...", "type": "Python Script"},
        {"id": "best_models", "label": "best_model.pt (All)", "c": 4, "file": "Outputs/4class/training/.../best_model.pt", "type": "PyTorch Checkpoints"},
        
        # Ensemble
        {"id": "compute_logits", "label": "compute_logits.py", "c": 2, "file": "Scripts_4/compute_logits.py", "type": "Python Script"},
        {"id": "logits", "label": "val/test_logits.npy", "c": 4, "file": "Outputs/4class/training/.../logits.npy", "type": "Numpy Arrays"},
        {"id": "ensemble_evaluate", "label": "ensemble_evaluate_v2...py", "c": 2, "file": "Scripts_4/ensemble_evaluate_v2_temperature.py", "type": "Python Script"},
        {"id": "final_leaderboard", "label": "ensemble_leaderboard.csv", "c": 4, "file": "Outputs/4class/training/ensemble_leaderboard_temperature.csv", "type": "CSV Leaderboard"},
    ]

    raw_edges_data = [
        # Setup & Validation
        ("dataset", "setup_pc", "reads"),
        ("setup_pc", "manifest", "writes"),
        ("dataset", "validate_dataset", "reads"),
        ("validate_dataset", "validated_manifest", "writes"),
        
        # Denoising
        ("dataset", "denoise_all", "reads"),
        ("validated_manifest", "denoise_all", "filters via"),
        ("denoise_all", "denoised", "writes to"),
        
        # Splits
        ("validated_manifest", "generate_splits", "filters via"),
        ("denoised", "generate_splits", "reads"),
        ("generate_splits", "splits", "writes"),
        
        # Precomputes (read from denoised & splits)
        ("denoised", "precompute_stft", "reads"),
        ("splits", "precompute_stft", "stratifies via"),
        ("precompute_stft", "stft_features", "writes"),
        
        ("denoised", "precompute_dwt", "reads"),
        ("splits", "precompute_dwt", "stratifies via"),
        ("precompute_dwt", "dwt_features", "writes"),
        
        ("denoised", "precompute_envelope", "reads"),
        ("splits", "precompute_envelope", "stratifies via"),
        ("precompute_envelope", "env_features", "writes"),
        
        ("denoised", "precompute_env_stft", "reads"),
        ("splits", "precompute_env_stft", "stratifies via"),
        ("precompute_env_stft", "env_stft_features", "writes"),

        # Training
        ("splits", "train_stft", "reads splits"),
        ("stft_features", "train_stft", "reads features"),
        ("train_stft", "best_models", "writes"),
        
        ("splits", "train_dwt", "reads splits"),
        ("dwt_features", "train_dwt", "reads features"),
        ("train_dwt", "best_models", "writes"),
        
        ("splits", "train_env", "reads splits"),
        ("env_features", "train_env", "reads features"),
        ("env_stft_features", "train_env", "reads features"),
        ("train_env", "best_models", "writes"),
        
        # Ensemble
        ("best_models", "compute_logits", "loads"),
        ("compute_logits", "logits", "writes"),
        ("logits", "ensemble_evaluate", "loads"),
        ("ensemble_evaluate", "final_leaderboard", "writes"),
    ]

    # Calculate degrees
    degrees = {n["id"]: 0 for n in raw_nodes_data}
    for e in raw_edges_data:
        degrees[e[0]] += 1
        degrees[e[1]] += 1

    # Format nodes for JSON
    nodes_json = []
    for n in raw_nodes_data:
        color = COLORS[n["c"]]
        size = 14 + min(degrees[n["id"]], 10) * 1.5
        font_size = 12 if n["c"] == 2 else 14
        
        nodes_json.append({
            "id": n["id"],
            "label": n["label"],
            "color": {
                "background": color,
                "border": color,
                "highlight": {"background": "#ffffff", "border": color}
            },
            "size": size,
            "font": {"size": font_size, "color": "#ffffff"},
            "title": f"{n['label']} ({n['type']})",
            "_community": n["c"],
            "_community_name": COMMUNITIES[n["c"]],
            "_source_file": n["file"],
            "_file_type": n["type"],
            "_degree": degrees[n["id"]]
        })

    # Format edges for JSON
    edges_json = []
    for f, t, lbl in raw_edges_data:
        edges_json.append({
            "from": f,
            "to": t,
            "label": lbl,
            "title": lbl,
            "dashes": False,
            "width": 2,
            "color": {"opacity": 0.7}
        })

    # Format legend
    legend_json = []
    for c_id, name in COMMUNITIES.items():
        count = sum(1 for n in raw_nodes_data if n["c"] == c_id)
        legend_json.append({
            "cid": c_id,
            "color": COLORS[c_id],
            "label": name,
            "count": count
        })

    # HTML Template (Extracted & minified from graphify-out/graph.html)
    html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Data Pipeline Graph (graphify theme)</title>
<script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: #0f0f1a; color: #e0e0e0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; display: flex; height: 100vh; overflow: hidden; }}
  #graph {{ flex: 1; }}
  #sidebar {{ width: 280px; background: #1a1a2e; border-left: 1px solid #2a2a4e; display: flex; flex-direction: column; overflow: hidden; }}
  #search-wrap {{ padding: 12px; border-bottom: 1px solid #2a2a4e; }}
  #search {{ width: 100%; background: #0f0f1a; border: 1px solid #3a3a5e; color: #e0e0e0; padding: 7px 10px; border-radius: 6px; font-size: 13px; outline: none; }}
  #search:focus {{ border-color: #4E79A7; }}
  #search-results {{ max-height: 140px; overflow-y: auto; padding: 4px 12px; border-bottom: 1px solid #2a2a4e; display: none; }}
  .search-item {{ padding: 4px 6px; cursor: pointer; border-radius: 4px; font-size: 12px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
  .search-item:hover {{ background: #2a2a4e; }}
  #info-panel {{ padding: 14px; border-bottom: 1px solid #2a2a4e; min-height: 140px; }}
  #info-panel h3 {{ font-size: 13px; color: #aaa; margin-bottom: 8px; text-transform: uppercase; letter-spacing: 0.05em; }}
  #info-content {{ font-size: 13px; color: #ccc; line-height: 1.6; }}
  #info-content .field {{ margin-bottom: 5px; }}
  #info-content .field b {{ color: #e0e0e0; }}
  #info-content .empty {{ color: #555; font-style: italic; }}
  .neighbor-link {{ display: block; padding: 2px 6px; margin: 2px 0; border-radius: 3px; cursor: pointer; font-size: 12px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; border-left: 3px solid #333; }}
  .neighbor-link:hover {{ background: #2a2a4e; }}
  #neighbors-list {{ max-height: 160px; overflow-y: auto; margin-top: 4px; }}
  #legend-wrap {{ flex: 1; overflow-y: auto; padding: 12px; }}
  #legend-wrap h3 {{ font-size: 13px; color: #aaa; margin-bottom: 10px; text-transform: uppercase; letter-spacing: 0.05em; }}
  .legend-item {{ display: flex; align-items: center; gap: 8px; padding: 4px 0; cursor: pointer; border-radius: 4px; font-size: 12px; }}
  .legend-item:hover {{ background: #2a2a4e; padding-left: 4px; }}
  .legend-item.dimmed {{ opacity: 0.35; }}
  .legend-dot {{ width: 12px; height: 12px; border-radius: 50%; flex-shrink: 0; }}
  .legend-label {{ flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
  .legend-count {{ color: #666; font-size: 11px; }}
  #stats {{ padding: 10px 14px; border-top: 1px solid #2a2a4e; font-size: 11px; color: #555; }}
</style>
</head>
<body>
<div id="graph"></div>
<div id="sidebar">
  <div id="search-wrap">
    <input id="search" type="text" placeholder="Search nodes..." autocomplete="off">
    <div id="search-results"></div>
  </div>
  <div id="info-panel">
    <h3>Node Info</h3>
    <div id="info-content"><span class="empty">Click a node to inspect it</span></div>
  </div>
  <div id="legend-wrap">
    <h3>Categories</h3>
    <div id="legend"></div>
  </div>
  <div id="stats">{len(nodes_json)} nodes &middot; {len(edges_json)} edges</div>
</div>
<script>
const RAW_NODES = {json.dumps(nodes_json)};
const RAW_EDGES = {json.dumps(edges_json)};
const LEGEND = {json.dumps(legend_json)};

function esc(s) {{
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;').replace(/'/g,'&#39;');
}}

const nodesDS = new vis.DataSet(RAW_NODES.map(n => ({{
  id: n.id, label: n.label, color: n.color, size: n.size,
  font: n.font, title: n.title,
  _community: n._community, _community_name: n._community_name,
  _source_file: n._source_file, _file_type: n._file_type, _degree: n._degree,
}})));

const edgesDS = new vis.DataSet(RAW_EDGES.map((e, i) => ({{
  id: i, from: e.from, to: e.to,
  label: e.label,
  title: e.title,
  dashes: e.dashes,
  width: e.width,
  color: e.color,
  arrows: {{ to: {{ enabled: true, scaleFactor: 0.5 }} }},
  font: {{ size: 10, align: 'middle', color: '#888' }}
}})));

const container = document.getElementById('graph');
const network = new vis.Network(container, {{ nodes: nodesDS, edges: edgesDS }}, {{
  physics: {{
    enabled: true,
    solver: 'forceAtlas2Based',
    forceAtlas2Based: {{
      gravitationalConstant: -100,
      centralGravity: 0.01,
      springLength: 150,
      springConstant: 0.05,
      damping: 0.4,
      avoidOverlap: 1,
    }},
    stabilization: {{ iterations: 200, fit: true }},
  }},
  interaction: {{
    hover: true,
    tooltipDelay: 100,
    hideEdgesOnDrag: true,
    navigationButtons: false,
    keyboard: false,
  }},
  nodes: {{ shape: 'dot', borderWidth: 1.5 }},
  edges: {{ smooth: {{ type: 'continuous', roundness: 0.2 }}, selectionWidth: 3 }},
}});

network.once('stabilizationIterationsDone', () => {{
  network.setOptions({{ physics: {{ enabled: false }} }});
}});

function showInfo(nodeId) {{
  const n = nodesDS.get(nodeId);
  if (!n) return;
  const neighborIds = network.getConnectedNodes(nodeId);
  const neighborItems = neighborIds.map(nid => {{
    const nb = nodesDS.get(nid);
    const color = nb ? nb.color.background : '#555';
    return `<span class="neighbor-link" style="border-left-color:${{esc(color)}}" onclick="focusNode('${{esc(nid)}}')">${{esc(nb ? nb.label : nid)}}</span>`;
  }}).join('');
  document.getElementById('info-content').innerHTML = `
    <div class="field"><b>${{esc(n.label)}}</b></div>
    <div class="field">Type: ${{esc(n._file_type || 'unknown')}}</div>
    <div class="field">Category: ${{esc(n._community_name)}}</div>
    <div class="field">Source: ${{esc(n._source_file || '-')}}</div>
    <div class="field">Connections: ${{n._degree}}</div>
    ${{neighborIds.length ? `<div class="field" style="margin-top:8px;color:#aaa;font-size:11px">Neighbors (${{neighborIds.length}})</div><div id="neighbors-list">${{neighborItems}}</div>` : ''}}
  `;
}}

function focusNode(nodeId) {{
  network.focus(nodeId, {{ scale: 1.2, animation: true }});
  network.selectNodes([nodeId]);
  showInfo(nodeId);
}}

let hoveredNodeId = null;
network.on('hoverNode', params => {{
  hoveredNodeId = params.node;
  container.style.cursor = 'pointer';
}});
network.on('blurNode', () => {{
  hoveredNodeId = null;
  container.style.cursor = 'default';
}});
container.addEventListener('click', () => {{
  if (hoveredNodeId !== null) {{
    showInfo(hoveredNodeId);
    network.selectNodes([hoveredNodeId]);
  }}
}});
network.on('click', params => {{
  if (params.nodes.length > 0) {{
    showInfo(params.nodes[0]);
  }} else if (hoveredNodeId === null) {{
    document.getElementById('info-content').innerHTML = '<span class="empty">Click a node to inspect it</span>';    
  }}
}});

const searchInput = document.getElementById('search');
const searchResults = document.getElementById('search-results');
searchInput.addEventListener('input', () => {{
  const q = searchInput.value.toLowerCase().trim();
  searchResults.innerHTML = '';
  if (!q) {{ searchResults.style.display = 'none'; return; }}
  const matches = RAW_NODES.filter(n => n.label.toLowerCase().includes(q)).slice(0, 20);
  if (!matches.length) {{ searchResults.style.display = 'none'; return; }}
  searchResults.style.display = 'block';
  matches.forEach(n => {{
    const el = document.createElement('div');
    el.className = 'search-item';
    el.textContent = n.label;
    el.style.borderLeft = `3px solid ${{n.color.background}}`;
    el.style.paddingLeft = '8px';
    el.onclick = () => {{
      focusNode(n.id);
      searchResults.style.display = 'none';
      searchInput.value = '';
    }};
    searchResults.appendChild(el);
  }});
}});
document.addEventListener('click', e => {{
  if (!searchResults.contains(e.target) && e.target !== searchInput)
    searchResults.style.display = 'none';
}});

const legendEl = document.getElementById('legend');
LEGEND.forEach(c => {{
  const item = document.createElement('div');
  item.className = 'legend-item';
  item.innerHTML = `<div class="legend-dot" style="background:${{c.color}}"></div>
    <span class="legend-label">${{c.label}}</span>
    <span class="legend-count">${{c.count}}</span>`;
  legendEl.appendChild(item);
}});
</script>
</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_template)
    
    print(f"Data pipeline visualization created at: {Path(output_path).absolute()}")

if __name__ == "__main__":
    generate_visjs_html()