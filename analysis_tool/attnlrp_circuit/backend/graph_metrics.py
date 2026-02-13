import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import entropy

class GraphMetrics:
    """
    A unified class for calculating metrics on attribution graphs.
    Metrics are organized into three groups:
    1. Connectivity & Sparsity
    2. Information Flow & Concentration
    3. Temporal Dynamics
    
    Each metric provides:
    - Global Summary (single scalar)
    - Layer-wise Profile (DataFrame/Series)
    """
    def __init__(self, G, tokens=None, total_seq_len=None):
        self.G = G
        self.tokens = tokens
        
        if total_seq_len is None and tokens is not None:
            self.total_seq_len = len(tokens)
        else:
            self.total_seq_len = total_seq_len
            
        self.nodes_by_layer = self._group_nodes_by_layer()
        self.sorted_layers = sorted(self.nodes_by_layer.keys())

    def _get_token_str(self, index):
        if self.tokens and 0 <= index < len(self.tokens):
            return self.tokens[index]
        return str(index)

    def _group_nodes_by_layer(self):
        groups = {}
        for n in self.G.nodes():
            layer = n[0]
            if layer not in groups:
                groups[layer] = []
            groups[layer].append(n)
        return groups

    def _calculate_gini(self, array):
        """Auxiliary function to calculate Gini coefficient."""
        array = np.abs(array)
        if np.sum(array) == 0:
            return 0.0
        array = np.sort(array)
        index = np.arange(1, array.shape[0] + 1)
        n = array.shape[0]
        return ((np.sum((2 * index - n  - 1) * array)) / (n * np.sum(array)))

    def _calculate_top_mass(self, values, fraction=0.9):
        """Calculates count and percentage of items needed to reach mass fraction."""
        values = np.abs(values)
        total = values.sum()
        if total == 0:
            return 0, 0.0
        
        sorted_vals = np.sort(values)[::-1]
        cumsum = np.cumsum(sorted_vals)
        cutoff = total * fraction
        
        # Find index where cumsum >= cutoff
        idx = np.searchsorted(cumsum, cutoff)
        count = idx + 1
        pct = (count / len(values)) * 100.0
        return count, pct

    def _get_top_nodes_by_mass(self, node_val_pairs, fraction=0.9):
        """
        Calculates top nodes making up mass fraction.
        Args:
            node_val_pairs: list of (node, value)
        Returns:
            count, pct, list of (token_idx, token_str)
        """
        if not node_val_pairs:
            return 0, 0.0, []
            
        values = np.array([abs(v) for _, v in node_val_pairs])
        total = values.sum()
        if total == 0:
            return 0, 0.0, []
        
        # Sort indices by value descending
        sorted_indices = np.argsort(values)[::-1]
        sorted_vals = values[sorted_indices]
        
        cumsum = np.cumsum(sorted_vals)
        cutoff = total * fraction
        
        idx_cutoff = np.searchsorted(cumsum, cutoff)
        count = idx_cutoff + 1
        pct = (count / len(values)) * 100.0
        
        # Extract top nodes
        top_indices = sorted_indices[:count]
        top_nodes = []
        for i in top_indices:
            node, _ = node_val_pairs[i]
            # node is (layer, token_idx)
            token_idx = int(node[1])
            token_str = self._get_token_str(token_idx)
            top_nodes.append((token_idx, token_str))
            
        return count, pct, top_nodes

    # ==========================================
    # Group 1: Connectivity & Sparsity
    # ==========================================
    
    def get_connectivity_stats(self):
        """Calculates Scale, Complexity, and Branching metrics."""
        if self.G.number_of_nodes() == 0:
            return pd.DataFrame()

        layer_stats = []
        
        # --- Layer-wise ---
        for layer in self.sorted_layers:
            nodes = self.nodes_by_layer[layer]
            
            # Fan-In (Inputs from prev layers)
            in_degrees = [self.G.in_degree(n) for n in nodes]
            avg_in = np.mean(in_degrees) if in_degrees else 0
            # std_in = np.std(in_degrees) if in_degrees else 0
            
            # Fan-Out (Outputs to next layers)
            out_degrees = [self.G.out_degree(n) for n in nodes]
            avg_out = np.mean(out_degrees) if out_degrees else 0
            # std_out = np.std(out_degrees) if out_degrees else 0
            
            # Derived Layer Metrics
            node_count = len(nodes)
            edge_count = np.sum(out_degrees)
            
            # Edge/Node Ratio
            edge_node_ratio = edge_count / node_count if node_count > 0 else 0
            
            stats = {
                "Layer": layer,
                "Node_Count": node_count,
                "Edge_Count": edge_count,
                "Edge_Node_Ratio": edge_node_ratio,
                "Avg_Fan_In": avg_in,
                # "Std_Fan_In": std_in,
                "Avg_Fan_Out": avg_out,
                # "Std_Fan_Out": std_out
            }

            if self.total_seq_len:
                stats["Sparsity"] = 1.0 - (node_count / self.total_seq_len)

            layer_stats.append(stats)
            
        df_layer = pd.DataFrame(layer_stats)
        
        return df_layer

    # ==========================================
    # Group 2: Information Flow & Concentration
    # ==========================================

    def get_flow_stats(self):
        """Calculates Effective Branching, Gini, and Verticality."""
        if self.G.number_of_nodes() == 0:
            return pd.DataFrame()

        layer_stats = []

        for layer in self.sorted_layers:
            nodes = self.nodes_by_layer[layer]
            # eff_degrees_in = []
            # eff_degrees_out = []
            node_ginis_in = []
            node_ginis_out = []
            
            # New Collectors
            layer_node_rels = []
            layer_out_edge_rels = []
            
            layer_vertical_mass = 0.0
            layer_total_mass = 0.0
            
            for n in nodes:
                # Node Relevance
                rel = abs(self.G.nodes[n].get('relevance', 0.0))
                layer_node_rels.append(rel)

                # Incoming Edges analysis
                in_edges = self.G.in_edges(n, data=True)
                weights_in = np.array([abs(d.get('weight', 0.0)) for u, v, d in in_edges])
                
                # Flow Mass
                if len(weights_in) > 0:
                    w_sum = weights_in.sum()
                    layer_total_mass += w_sum
                    
                    # Verticality check
                    for u, v, d in in_edges:
                        if u[1] == v[1]: # Same token index
                            layer_vertical_mass += abs(d.get('weight', 0.0))
                
                

                # Outgoing Edges analysis
                out_edges = self.G.out_edges(n, data=True)
                weights_out = np.array([abs(d.get('weight', 0.0)) for u, v, d in out_edges])
                
                # Collect Out Edge Relevances
                if len(weights_out) > 0:
                    layer_out_edge_rels.extend(weights_out)
                    
                    
                # Gini (On Inputs)
                if len(weights_in) > 1:
                    g = self._calculate_gini(weights_in)
                    node_ginis_in.append(g)
                elif len(weights_in) == 1:
                    node_ginis_in.append(1.0)
                    
                # Gini (On Outputs)
                if len(weights_out) > 1:
                    g = self._calculate_gini(weights_out)
                    node_ginis_out.append(g)
                elif len(weights_out) == 1:
                    node_ginis_out.append(1.0)

            # Node Statistics
            layer_node_rels = np.array(layer_node_rels)
            avg_node_rel = np.mean(layer_node_rels) if len(layer_node_rels) > 0 else 0
            node_rel_gini = self._calculate_gini(layer_node_rels) if len(layer_node_rels) > 0 else 0
            
            # Node Mass Fractions (Moved to Group 3)

            # Out Edge Statistics
            layer_out_edge_rels = np.array(layer_out_edge_rels)
            avg_out_edge_rel = np.mean(layer_out_edge_rels) if len(layer_out_edge_rels) > 0 else 0
            out_edge_rel_gini = self._calculate_gini(layer_out_edge_rels) if len(layer_out_edge_rels) > 0 else 0

            layer_stats.append({
                "Layer": layer,
                                
                # Node Relevance Stats
                "Avg_Node_Rel": avg_node_rel,
                "Node_Rel_Gini": node_rel_gini,

                # Out Edge Relevance Stats
                "Avg_Out_Edge_Rel": avg_out_edge_rel,
                "Gini_Out_Edge_Rel": out_edge_rel_gini,

                # by node Gini Stats
                "Avg_Edge_Gini_In_by_Node": np.mean(node_ginis_in) if node_ginis_in else 0,
                "Avg_Edge_Gini_Out_by_Node": np.mean(node_ginis_out) if node_ginis_out else 0,

                "Verticality_Ratio_by_Node": layer_vertical_mass / layer_total_mass if layer_total_mass > 0 else 0
            })

        df_layer = pd.DataFrame(layer_stats)
        
        return df_layer

    # ==========================================
    # Group 3: Node Hubs & Significant Tokens
    # ==========================================

    def get_node_hub_stats(self):
        """Calculates Top Mass Nodes, Degree Hubs, and Signed Relevance Hubs."""
        if self.G.number_of_nodes() == 0:
            return pd.DataFrame()
        
        layer_stats = []
        
        for layer in self.sorted_layers:
            nodes = self.nodes_by_layer[layer]
            
            # 1. Gather Data
            node_rels = [] # Pairs of (node, rel)
            in_degrees = [] # Pairs of (node, deg)
            out_degrees = [] # Pairs of (node, deg)
            
            pos_node_rels = []
            neg_node_rels = []
            
            for n in nodes:
                # Relevance
                rel = self.G.nodes[n].get('relevance', 0.0)
                node_rels.append((n, rel))
                
                if rel >= 0:
                    pos_node_rels.append((n, rel))
                else:
                    neg_node_rels.append((n, rel))
                
                # Degree
                in_deg = self.G.in_degree(n)
                in_degrees.append((n, in_deg))
                
                out_deg = self.G.out_degree(n)
                out_degrees.append((n, out_deg))
            
            # 2. General Top Mass (Abs)
            n_90, pct_90, top_90_nodes = self._get_top_nodes_by_mass(node_rels, fraction=0.9)
            
            # 3. Top Mass Positive & Negative
            n_pos, pct_pos, top_pos_nodes = self._get_top_nodes_by_mass(pos_node_rels, fraction=0.9)
            n_neg, pct_neg, top_neg_nodes = self._get_top_nodes_by_mass(neg_node_rels, fraction=0.9)
            
            # 4. Degree Hubs (Mean + Std)
            # In-Degree
            in_deg_vals = [d for _, d in in_degrees]
            if len(in_deg_vals) > 0:
                avg_in = np.mean(in_deg_vals)
                std_in = np.std(in_deg_vals)
                thresh_in = avg_in + std_in
                hub_in_nodes = []
                for n, deg in in_degrees:
                    if deg > thresh_in:
                        token_idx = int(n[1])
                        token_str = self._get_token_str(token_idx)
                        hub_in_nodes.append((token_idx, token_str))
            else:
                hub_in_nodes = []
            
            # Out-Degree
            out_deg_vals = [d for _, d in out_degrees]
            if len(out_deg_vals) > 0:
                avg_out = np.mean(out_deg_vals)
                std_out = np.std(out_deg_vals)
                thresh_out = avg_out + std_out
                hub_out_nodes = []
                for n, deg in out_degrees:
                    if deg > thresh_out:
                         token_idx = int(n[1])
                         token_str = self._get_token_str(token_idx)
                         hub_out_nodes.append((token_idx, token_str))
            else:
                hub_out_nodes = []
                
            layer_stats.append({
                "Layer": layer,
                
                # Abs Relevance Hubs
                "Node_Rel_Top_90_Pct_Count": n_90,
                "Node_Rel_Top_90_Pct": pct_90,
                # "Top_Rel_Nodes": top_90_nodes,
                
                # Signed Relevance Hubs
                "Pos_Rel_Top_90_Pct_Count": n_pos,
                "Pos_Rel_Top_90_Pct": pct_pos,
                # "Top_Pos_Nodes": top_pos_nodes,
                
                "Neg_Rel_Top_90_Pct_Count": n_neg,
                "Neg_Rel_Top_90_Pct": pct_neg,
                # "Top_Neg_Nodes": top_neg_nodes,
                
                # Structural Hubs
                "Hub_In_Count": len(hub_in_nodes),
                # "Hub_In_Nodes": hub_in_nodes,
                "Hub_Out_Count": len(hub_out_nodes),
                # "Hub_Out_Nodes": hub_out_nodes
            })
            
        return pd.DataFrame(layer_stats)

    # ==========================================
    # Group 4: Temporal Dynamics
    # ==========================================

    def get_temporal_stats(self):
        """Calculates Lookback, Drift, and Locality."""
        if self.G.number_of_nodes() == 0:
            return pd.DataFrame()

        layer_stats = []
        
        for layer in self.sorted_layers:
            nodes = self.nodes_by_layer[layer]
            
            # --- Center of Mass (Drift) ---
            positions = []
            relevances = []
            
            # For CoM, we look at the nodes THEMSELVES, not edges
            for n in nodes:
                # We need node relevance. If not stored, default to 1
                rel = self.G.nodes[n].get('relevance', 1.0)
                positions.append(n[1]) # Token Index
                relevances.append(abs(rel))
                
            positions = np.array(positions)
            relevances = np.array(relevances)
            
            if relevances.sum() > 0:
                com = np.average(positions, weights=relevances)
            else:
                com = np.mean(positions) if len(positions) > 0 else 0
                
            # --- Lookback & Locality (Edges entering this layer) ---
            layer_lookbacks = []
            
            for n in nodes:
                in_edges = self.G.in_edges(n)
                for u, v in in_edges:
                    # u is source, v is target (n)
                    # Lookback = v_pos - u_pos
                    dist = v[1] - u[1]
                    layer_lookbacks.append(dist)
            
            layer_lookbacks = np.array(layer_lookbacks)
            
            layer_stats.append({
                "Layer": layer,
                "Center_Of_Mass_Idx": com,
                "Mean_Lookback": np.mean(layer_lookbacks) if len(layer_lookbacks) > 0 else 0,
                "Local_Processing_Ratio": np.mean(layer_lookbacks == 0) if len(layer_lookbacks) > 0 else 0
            })
            
        df_layer = pd.DataFrame(layer_stats)
        
        return df_layer

