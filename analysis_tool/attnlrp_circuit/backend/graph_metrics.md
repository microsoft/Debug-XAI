# Evaluating Attribution Circuits: A Metric Taxonomy

This document outlines a consolidated taxonomy for graph metrics designed to evaluate relevance attribution graphs extracted from transformer models. The metrics are organized into four groups, analyzing structure, flow, significant nodes, and temporal dynamics.

The python implementation returns these metrics primarily as **Layer-wise Profiles** (Pandas DataFrames), allowing for depth-wise analysis of the circuit's behavior.

---

## Group 1: Connectivity & Sparsity (The "Skeleton")
*Evaluating the physical topology of the circuit. These metrics are generally **unweighted**, focusing on the existence of connections rather than their strength.*

**Core Question:** "Is the relevant sub-network dense or sparse? How interconnected are the layers?"

### 1.1 Scale
*   **Metric:** `Node_Count`, `Edge_Count`
    *   *Description:* Number of active nodes/edges at each layer.
    *   *Interpretation:* Forms a "funnel" or "diamond". Sudden changes indicate aggregation or broadcasting.

### 1.2 Complexity
*   **Metric:** `Edge_Node_Ratio` ($\frac{|E_L|}{|V_L|}$)
    *   *Description:* Average number of outgoing edges per node in the layer.
    *   *Interpretation:* Low ($\approx 1$) implies simple chains; High ($>2$) implies information mixing.

### 1.3 Branching
*   **Metric:** `Avg_Fan_In`, `Avg_Fan_Out`
    *   *Description:* The average number of incoming edges (from prev layer) and outgoing edges (to next layer).
    *   *Interpretation:* Indicates if the layer acts as an aggregator (High Fan-In) or broadcaster (High Fan-Out).

### 1.4 Sparsity
*   **Metric:** `Sparsity`
    *   *Formula:* $1 - \frac{|V_{active}|}{SeqLen}$ (if total sequence length is known).
    *   *Description:* Percentage of the sequence tokens that are *irrelevant*/pruned at this layer.

---

## Group 2: Information Flow & Concentration (The "Hydraulics")
*Evaluating the distribution of relevance mass. These metrics are **weighted**, analyzing how relevance flows through the topology.*

**Core Question:** "Is the computation diffuse (democratic) or driven by a few critical components? Is it mostly residual or mixing?"

### 2.1 Inequality (Gini Coefficients)
All Gini coefficients range from 0 (perfect equality) to 1 (perfect inequality/concentration).

*   **Metric:** `Avg_Edge_Gini_In`
    *   *Description:* Gini of incoming edge weights.
    *   *Interpretation:* High = "Picky" nodes (attend to 1 specific input). Low = "Diffuse" mixing.
*   **Metric:** `Avg_Edge_Gini_Out`, `Gini_Out_Edge_Rel`
    *   *Description:* Gini of outgoing edge weights.
    *   *Interpretation:* High = Node sends mass to specific targets.
*   **Metric:** `Node_Rel_Gini`
    *   *Description:* Gini of the node relevance magnitudes within the layer.
    *   *Interpretation:* High = Layer dominated by a few "Super-Nodes".

### 2.2 Magnitude
*   **Metric:** `Avg_Node_Rel`, `Avg_Out_Edge_Rel`
    *   *Description:* Mean absolute relevance of nodes/edges in the layer.

### 2.3 Flow Mode (Verticality)
*   **Metric:** `Verticality_Ratio`
    *   *Formula:* $\frac{\text{Mass}(SamePosition)}{\text{TotalMass}}$
    *   *Description:* Proportion of incoming relevance coming from the *same* token position (Residual Stream).
    *   *Interpretation:* High = Independent processing (Syntax/Memory). Low = Mixing (Attention/Reasoning).

---

## Group 3: Node Hubs & Significant Tokens (The "VIPs")
*Identifying specific, critical tokens. Unlike Group 2 (summaries), this group extracts the identity of the top contributors.*

**Core Question:** "Which specific tokens are doing the heavy lifting?"

### 3.1 Relevance Hubs (Mass Fraction)
Identifies the smallest set of nodes required to account for **90%** of the layer's total relevance mass.

*   **Metric:** `Node_Rel_Top_90_Pct_Count`
    *   *Description:* Number of nodes forming the top 90% relevance mass.
*   **Metric:** `Node_Rel_Top_90_Pct`
    *   *Description:* The actual pct mass captured (approx 90%).
*   **Metric:** `Top_Rel_Nodes`
    *   *Description:* List of `(token_index, token_string)` tuples for these high-relevance nodes.

### 3.2 Signed Relevance Hubs
Separates Positive (Excitatory) and Negative (Inhibitory) contributions.

*   **Metric:** `Pos_Rel_Top_90_Pct_Count`, `Top_Pos_Nodes`
    *   *Description:* Analysis restricted to nodes with positive relevance.
*   **Metric:** `Neg_Rel_Top_90_Pct_Count`, `Top_Neg_Nodes`
    *   *Description:* Analysis restricted to nodes with negative relevance.

### 3.3 Structural Hubs (Degree Centrality)
Identifies nodes with statistically high connectivity.
*   **Threshold:** $Degree > Mean + StdDev$ (calculated per layer).

*   **Metric:** `Hub_In_Count`, `Hub_In_Nodes`
    *   *Description:* Nodes with anomalously high Fan-In (Aggregation Hubs).
*   **Metric:** `Hub_Out_Count`, `Hub_Out_Nodes`
    *   *Description:* Nodes with anomalously high Fan-Out (Broadcasting Hubs).

---

## Group 4: Temporal Dynamics (The "Sequence")
*Evaluating how the graph maps onto the token sequence.*

**Core Question:** "Where in the context does the model look, and does this change over depth?"

### 4.1 Temporal Drift
*   **Metric:** `Center_Of_Mass_Idx`
    *   *Description:* The weighted average position index of active nodes at Layer $L$.
    *   *Interpretation:* Visualizes the "reasoning locus" moving through the text.

### 4.2 Lookback Dynamics
*   **Metric:** `Mean_Lookback`
    *   *Formula:* Avg($pos_{target} - pos_{source}$) for incoming edges.
    *   *Description:* Average distance (in tokens) information travels to reach this layer.
*   **Metric:** `Local_Processing_Ratio`
    *   *Description:* Proportion of edges where lookback distance is 0.
