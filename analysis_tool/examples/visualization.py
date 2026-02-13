"""
Visualization utilities for latent feature attribution heatmaps.
Supports both matplotlib and interactive HTML rendering.
"""

import numpy as np
import matplotlib.pyplot as plt


def clean_token(token):
    """Clean special tokenizer characters from token strings.
    
    Replaces:
        - Ġ (space marker in GPT-2/BPE tokenizers) -> space
        - Ċ (newline marker) -> \\n
        - Other common special markers
    """
    replacements = {
        'Ġ': ' ',   # Space marker
        'Ċ': '\\n', # Newline marker
        'ĉ': '\\t', # Tab marker
        '▁': ' ',   # SentencePiece space marker
    }
    for old, new in replacements.items():
        token = token.replace(old, new)
    return token


def save_heatmap_matplotlib(values, tokens, figsize, title, save_path):
    """Create a heatmap visualization using matplotlib.
    
    Args:
        values: 2D array of shape (num_tokens, num_layers) with relevance values
        tokens: List of token strings
        figsize: Tuple of (width, height) for the figure
        title: Title for the heatmap
        save_path: Path to save the PNG file
    """
    # Clean tokens
    tokens = [clean_token(t) for t in tokens]
    
    fig, ax = plt.subplots(figsize=figsize)

    abs_max = abs(values).max()
    im = ax.imshow(values, cmap='bwr', vmin=-abs_max, vmax=abs_max)

    layers = np.arange(values.shape[-1])

    ax.set_xticks(np.arange(len(layers)))
    ax.set_yticks(np.arange(len(tokens)))

    ax.set_xticklabels(layers)
    ax.set_yticklabels(tokens)

    plt.title(title)
    plt.xlabel('Layers')
    plt.ylabel('Tokens')
    plt.colorbar(im)

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Matplotlib heatmap saved to {save_path}")
    return save_path


def create_html_heatmap(values, tokens, title, save_path):
    """Create an interactive HTML heatmap visualization.
    
    Args:
        values: 2D array of shape (num_tokens, num_layers) with relevance values
        tokens: List of token strings
        title: Title for the heatmap
        save_path: Path to save the HTML file
    """
    # Clean tokens
    tokens = [clean_token(t) for t in tokens]
    
    # Prepare data
    abs_max = abs(values).max()
    norm_values = values / abs_max if abs_max > 0 else values
    
    num_tokens, num_layers = values.shape
    
    # Generate color for each cell
    def value_to_color(val):
        """Convert normalized value to RGB color (blue-white-red)."""
        if val > 0:
            # Positive: white to red
            intensity = int(255 * (1 - val))
            return f'rgb(255, {intensity}, {intensity})'
        else:
            # Negative: white to blue
            intensity = int(255 * (1 + val))
            return f'rgb({intensity}, {intensity}, 255)'
    
    # Build HTML
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
            display: flex;
            flex-direction: column;
            align-items: center;
        }}
        
        .container {{
            background: white;
            border-radius: 15px;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
            padding: 30px;
            max-width: 95%;
            overflow-x: auto;
        }}
        
        h1 {{
            text-align: center;
            color: #333;
            margin-bottom: 30px;
            font-size: 28px;
            font-weight: 600;
        }}
        
        .heatmap-wrapper {{
            display: flex;
            align-items: flex-start;
            gap: 10px;
            overflow-x: auto;
        }}
        
        .token-labels {{
            display: flex;
            flex-direction: column;
            gap: 2px;
            padding-top: 40px;
        }}
        
        .token-label {{
            height: 30px;
            display: flex;
            align-items: center;
            padding: 0 10px;
            font-family: 'Courier New', monospace;
            font-size: 13px;
            background: #f8f9fa;
            border-radius: 4px;
            white-space: nowrap;
            font-weight: 500;
            color: #2c3e50;
        }}
        
        .heatmap {{
            display: flex;
            flex-direction: column;
            gap: 2px;
        }}
        
        .layer-labels {{
            display: flex;
            gap: 2px;
            margin-bottom: 5px;
        }}
        
        .layer-label {{
            width: 30px;
            height: 35px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 11px;
            font-weight: 600;
            color: #555;
            background: #f0f0f0;
            border-radius: 4px;
        }}
        
        .heatmap-row {{
            display: flex;
            gap: 2px;
        }}
        
        .heatmap-cell {{
            width: 30px;
            height: 30px;
            border-radius: 3px;
            cursor: pointer;
            transition: transform 0.2s, box-shadow 0.2s;
            border: 1px solid rgba(0, 0, 0, 0.05);
        }}
        
        .heatmap-cell:hover {{
            transform: scale(1.2);
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
            z-index: 10;
            border: 2px solid #333;
        }}
        
        .tooltip {{
            position: fixed;
            background: rgba(0, 0, 0, 0.9);
            color: white;
            padding: 10px 15px;
            border-radius: 8px;
            pointer-events: none;
            display: none;
            z-index: 1000;
            font-size: 13px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4);
            max-width: 300px;
        }}
        
        .tooltip.show {{
            display: block;
        }}
        
        .legend {{
            margin-top: 30px;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 15px;
        }}
        
        .legend-gradient {{
            width: 300px;
            height: 20px;
            background: linear-gradient(to right, rgb(100, 100, 255), rgb(255, 255, 255), rgb(255, 100, 100));
            border-radius: 10px;
            border: 1px solid #ddd;
        }}
        
        .legend-labels {{
            display: flex;
            justify-content: space-between;
            width: 300px;
            font-size: 12px;
            color: #666;
            margin-top: 5px;
        }}
        
        .stats {{
            margin-top: 20px;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 8px;
            display: flex;
            justify-content: space-around;
            flex-wrap: wrap;
            gap: 15px;
        }}
        
        .stat-item {{
            text-align: center;
        }}
        
        .stat-label {{
            font-size: 12px;
            color: #666;
            margin-bottom: 5px;
        }}
        
        .stat-value {{
            font-size: 18px;
            font-weight: 600;
            color: #333;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{title}</h1>
        
        <div class="heatmap-wrapper">
            <div class="token-labels">
"""
    
    # Add token labels
    for token in tokens:
        # Escape HTML special characters
        token_escaped = token.replace('<', '&lt;').replace('>', '&gt;').replace('&', '&amp;')
        html_content += f'                <div class="token-label">{token_escaped}</div>\n'
    
    html_content += """            </div>
            
            <div class="heatmap">
                <div class="layer-labels">
"""
    
    # Add layer labels
    for layer_idx in range(num_layers):
        html_content += f'                    <div class="layer-label">L{layer_idx}</div>\n'
    
    html_content += """                </div>
"""
    
    # Add heatmap cells
    for token_idx in range(num_tokens):
        html_content += '                <div class="heatmap-row">\n'
        for layer_idx in range(num_layers):
            val = norm_values[token_idx, layer_idx]
            raw_val = values[token_idx, layer_idx]
            color = value_to_color(val)
            token_escaped = tokens[token_idx].replace('<', '&lt;').replace('>', '&gt;').replace('&', '&amp;')
            html_content += f'                    <div class="heatmap-cell" style="background-color: {color};" data-token="{token_escaped}" data-layer="{layer_idx}" data-value="{raw_val:.6f}" data-norm="{val:.4f}"></div>\n'
        html_content += '                </div>\n'
    
    # Calculate statistics
    max_val = float(values.max())
    min_val = float(values.min())
    mean_val = float(values.mean())
    std_val = float(np.std(values))
    
    html_content += f"""            </div>
        </div>
        
        <div class="legend">
            <div>
                <div class="legend-gradient"></div>
                <div class="legend-labels">
                    <span>Negative</span>
                    <span>Zero</span>
                    <span>Positive</span>
                </div>
            </div>
        </div>
        
        <div class="stats">
            <div class="stat-item">
                <div class="stat-label">Max Value</div>
                <div class="stat-value">{max_val:.4f}</div>
            </div>
            <div class="stat-item">
                <div class="stat-label">Min Value</div>
                <div class="stat-value">{min_val:.4f}</div>
            </div>
            <div class="stat-item">
                <div class="stat-label">Mean</div>
                <div class="stat-value">{mean_val:.4f}</div>
            </div>
            <div class="stat-item">
                <div class="stat-label">Std Dev</div>
                <div class="stat-value">{std_val:.4f}</div>
            </div>
            <div class="stat-item">
                <div class="stat-label">Tokens</div>
                <div class="stat-value">{num_tokens}</div>
            </div>
            <div class="stat-item">
                <div class="stat-label">Layers</div>
                <div class="stat-value">{num_layers}</div>
            </div>
        </div>
    </div>
    
    <div class="tooltip" id="tooltip"></div>
    
    <script>
        const cells = document.querySelectorAll('.heatmap-cell');
        const tooltip = document.getElementById('tooltip');
        
        cells.forEach(cell => {{
            cell.addEventListener('mouseenter', (e) => {{
                const token = cell.dataset.token;
                const layer = cell.dataset.layer;
                const value = parseFloat(cell.dataset.value);
                const norm = parseFloat(cell.dataset.norm);
                
                tooltip.innerHTML = `
                    <strong>Token:</strong> ${{token}}<br>
                    <strong>Layer:</strong> ${{layer}}<br>
                    <strong>Value:</strong> ${{value.toFixed(6)}}<br>
                    <strong>Normalized:</strong> ${{norm.toFixed(4)}}
                `;
                tooltip.classList.add('show');
            }});
            
            cell.addEventListener('mousemove', (e) => {{
                tooltip.style.left = (e.clientX + 15) + 'px';
                tooltip.style.top = (e.clientY + 15) + 'px';
            }});
            
            cell.addEventListener('mouseleave', () => {{
                tooltip.classList.remove('show');
            }});
        }});
    </script>
</body>
</html>
"""
    
    # Save to file
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"Interactive HTML heatmap saved to {save_path}")
    return save_path
