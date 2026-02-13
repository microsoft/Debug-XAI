import networkx as nx
import torch
import gc
import numpy as np
import pandas as pd

class CircuitAnalyzer:
    """
    A class to build and analyze attribution circuits using an AttributionEngine.
    """
    def __init__(self, attribution_engine):
        """
        Args:
            attribution_engine: An instance of AttributionEngine.
        """
        self.engine = attribution_engine

    def compute_connection_matrices(self, backprop_config, layers=None):
        """
        Step 1: Compute dense connection matrices for all layer transitions.
        Returns a list of dictionaries containing matrix data for each transition.
        """
        model = self.engine.manager.get_model()
        model_layers = model.model.layers
        n_layers = len(model_layers)

        if layers is None:
            # Auto-detect if mid activations are available (captured in compute_logits)
            check_layer = model_layers[0]
            has_mid = False
            # Check for mid_activation attribute on the normative layer where we hook
            if hasattr(check_layer, 'post_attention_layernorm') and hasattr(check_layer.post_attention_layernorm, 'mid_activation'):
                has_mid = True
                print("Auto-detected mid activations: Using fine-grained circuit (Attn/MLP separation).")

            nodes = [-1]
            for i in range(n_layers):
                if has_mid:
                    nodes.append((i, 'mid'))
                nodes.append(i) # integers imply 'post'
            
            # Sort/Sequence is crucial.
            # [-1, (0,'mid'), 0, (1,'mid'), 1, ...]
            layer_pairs = list(zip(nodes[:-1], nodes[1:]))
        else:
            # Use provided layers list
            # Ensure it's sorted? User responsibility if custom list.
            nodes = layers
            layer_pairs = list(zip(nodes[:-1], nodes[1:]))
        
        # 1. Run Backward Pass
        print("Running backward pass...")
        self.engine.run_backward_pass(backprop_config)
        
        connection_data = []
        
        print(f"Computing matrices for {len(nodes)} nodes ({len(layer_pairs)} transitions)...")
        
        for i, (src_layer, tgt_layer) in enumerate(layer_pairs):
            print(f"Computing transition: {src_layer} -> {tgt_layer}...")
            
            gen = self.engine.compute_connection_matrix_gen(
                source=src_layer, 
                target=tgt_layer
            )
            
            final_res = None
            for item in gen:
                if item['type'] == 'result':
                    final_res = item['payload']
                    
            if final_res:
                # Store the dense matrix and relevance vectors
                connection_data.append({
                    'src_layer': src_layer,
                    'tgt_layer': tgt_layer,
                    'matrix': final_res['matrix'], # Dense numpy array
                    'real_source_rel': final_res['real_source_rel'],
                    'real_target_rel': final_res['real_target_rel']
                })
                
                # Basic cleanup of engine internals, but we keep the matrix in connection_data
                torch.cuda.empty_cache()
                
        return connection_data

    def build_graph_from_matrices(self, connection_data, edge_rel_threshold=0.01, pruning_mode="by_global_threshold", top_p=0.9):
        """
        Step 2: Prune based on threshold and build NetworkX graph.
        
        Args:
            connection_data: List of dictionaries containing matrix data.
            edge_rel_threshold: Threshold for 'by_global_threshold' mode.
            pruning_mode: Pruning strategy. Options:
                          1. "by_global_threshold" (default): Prune globally using edge_rel_threshold.
                          2. "by_per_layer_cum_mass_percentile": Prune per layer to keep top_p mass.
            top_p: The cumulative mass percentile (0.0-1.0) for "by_per_layer_cum_mass_percentile". Default 0.9.
            
        Returns:
            G: The built NetworkX graph.
            pruning_details: A list of dictionaries containing pruning stats per layer pair.
        """
        G = nx.DiGraph()
        print(f"Building graph from {len(connection_data)} transitions. Mode: {pruning_mode}...")
        
        pruning_details = []
        
        for data in connection_data:
            src_layer = data['src_layer']
            tgt_layer = data['tgt_layer']
            matrix = data['matrix']
            real_source_rel = data['real_source_rel']
            real_target_rel = data['real_target_rel']
            abs_matrix = np.abs(matrix)
            
            # Count only active edges (non-zero) as total_elements for percentage calculation
            nonzero_mask = abs_matrix > 1e-9
            total_elements = np.sum(nonzero_mask)
            
            used_threshold = edge_rel_threshold

            # Determine mask for edges to keep based on mode
            if pruning_mode == "by_per_layer_cum_mass_percentile":
                # Dynamic thresholding per layer
                flattened = np.sort(abs_matrix.flatten())[::-1]
                total_mass = flattened.sum()
                
                dynamic_threshold = 1.0 # Default High if empty
                
                if total_mass > 1e-12:
                     cumsum = np.cumsum(flattened)
                     cutoff_mass = total_mass * top_p
                     # Find first index where cumsum >= cutoff_mass
                     # searchsorted returns insertion point index i such that a[i-1] < v <= a[i]
                     # If we want elements up to index k such sum(0..k) >= target
                     cutoff_idx = np.searchsorted(cumsum, cutoff_mass)
                     
                     if cutoff_idx >= len(flattened):
                         cutoff_idx = len(flattened) - 1
                         
                     dynamic_threshold = flattened[cutoff_idx]
                     # Ensure we do not include effective zeros
                     if dynamic_threshold < 1e-9:
                         dynamic_threshold = 1e-9
                
                used_threshold = dynamic_threshold
                rows, cols = np.where(abs_matrix >= dynamic_threshold)
            else:
                # "by_global_threshold"
                rows, cols = np.where(abs_matrix > edge_rel_threshold)
                
            # Record pruning details
            num_kept = len(rows)
            percentage = (num_kept / total_elements * 100) if total_elements > 0 else 0
            
            pruning_details.append({
                'src_layer': src_layer,
                'tgt_layer': tgt_layer,
                'threshold': float(used_threshold),
                'kept_edges': int(num_kept),
                'total_edges': int(total_elements),
                'kept_percentage': percentage
            })

            # Add Source Nodes
            for t_idx in np.where(np.abs(real_source_rel) > 0)[0]:
                    src_node_id = (src_layer, t_idx)
                    if not G.has_node(src_node_id):
                        G.add_node(src_node_id, layer=src_layer, token=t_idx, relevance=real_source_rel[t_idx])

            # Add Target Nodes
            for t_idx in np.where(np.abs(real_target_rel) > 0)[0]:
                    tgt_node_id = (tgt_layer, t_idx)
                    if not G.has_node(tgt_node_id):
                        G.add_node(tgt_node_id, layer=tgt_layer, token=t_idx, relevance=real_target_rel[t_idx])

            # Add Edges (Sparse)
            # rows, cols calculated above
            weights = matrix[rows, cols]
            
            edges_to_add = []
            for r, c, w in zip(rows, cols, weights):
                # Edge direction: Source (col/c) -> Target (row/r)
                u = (src_layer, c)
                v = (tgt_layer, r)
                # Ensure nodes exist (redundant safety check)
                if not G.has_node(u): G.add_node(u, layer=src_layer, token=c, relevance=real_source_rel[c])
                if not G.has_node(v): G.add_node(v, layer=tgt_layer, token=r, relevance=real_target_rel[r])
                
                edges_to_add.append((u, v, {'weight': float(w)}))
            
            G.add_edges_from(edges_to_add)
        
        print(f"Graph construction complete. Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
        return G, pruning_details

    def build_graph(self, backprop_config, layers=None, edge_rel_threshold=0.01, pruning_mode="by_global_threshold", top_p=0.9):
        """
        Wrapper that performs both steps: compute matrices and build graph.
        """
        connection_data = self.compute_connection_matrices(backprop_config, layers)
        return self.build_graph_from_matrices(connection_data, edge_rel_threshold, pruning_mode=pruning_mode, top_p=top_p)

    def get_connected_subgraph(self, G, target_node=None):
        """
        Extracts the subgraph connected to the target node (backward reachability).
        If target_node is None, it tries to infer the last token at the last layer.
        
        Args:
            G: The full attribution graph
            target_node: tuple (layer, token_idx) of the target.
            
        Returns:
            (subgraph, target_node): The connected subgraph and the resolved target node.
        """
        # 1. Identify Target Node
        if target_node is None:
            if G.number_of_nodes() == 0:
                print("Graph is empty.")
                return None, None
            
            # Infer: Max layer
            try:
                max_layer = max([n[0] for n in G.nodes()])
            except ValueError:
                 print("Error finding max layer.")
                 return None, None

            # Check nodes in max_layer
            nodes_in_last = [n for n in G.nodes() if n[0] == max_layer]
            if not nodes_in_last:
                 print(f"No nodes found in layer {max_layer}.")
                 return None, None
            
            # Max token index
            max_token = max([n[1] for n in nodes_in_last])
            target_node = (max_layer, max_token)
            
        print(f"Extracting connected component for Target Node: {target_node}")
        
        if not G.has_node(target_node):
            print(f"Target node {target_node} not found in graph (maybe thresholded out?).")
            return None, target_node

        # 2. Get Ancestors (Backward Reachability)
        ancestors = nx.ancestors(G, target_node)
        ancestors.add(target_node) # Include self
        
        subgraph = G.subgraph(ancestors).copy()
        print(f"Connected Subgraph: {subgraph.number_of_nodes()} nodes, {subgraph.number_of_edges()} edges")
        
        return subgraph, target_node
