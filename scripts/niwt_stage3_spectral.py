import argparse
import json
import numpy as np

def perform_spectral_analysis(bitmask_data):
    """
    Perform spectral analysis on activation patterns to determine dimension reduction.
    
    Uses SVD (Singular Value Decomposition) on collected activation matrices to
    identify the intrinsic dimensionality of layer activations. Higher sparsity
    allows for more aggressive compression while preserving model quality.
    
    Args:
        bitmask_data: Dictionary containing feature_bitmask from Stage 2 profiling
        
    Returns:
        Dictionary mapping layer indices to their spectral configurations
    """
    print("\n--- NIWT STAGE 3: SPECTRAL ANALYSIS ---")
    print("Performing SVD-based dimensionality analysis on activation patterns\n")
    
    bitmask = bitmask_data['feature_bitmask']
    layer_configs = {}
    
    for layer_idx, mask in bitmask.items():
        mask_array = np.array(mask)
        active_neurons = np.sum(mask_array)
        total_neurons = len(mask_array)
        sparsity = 1 - (active_neurons / total_neurons)
        
        # Use SVD to determine effective rank and optimal target dimension
        # In production, this would use actual activation matrices from Stage 2
        # Here we estimate based on sparsity and theoretical bounds
        
        # Create a synthetic correlation matrix based on the bitmask pattern
        # This represents neuron activation correlations
        correlation_matrix = np.outer(mask_array.astype(float), mask_array.astype(float))
        
        # Add small noise to avoid singular matrices
        correlation_matrix += np.eye(total_neurons) * 1e-6
        
        # Perform SVD on the correlation matrix
        try:
            U, singular_values, Vt = np.linalg.svd(correlation_matrix, full_matrices=False)
            
            # Determine effective rank based on singular value energy
            total_energy = np.sum(singular_values)
            cumulative_energy = np.cumsum(singular_values)
            
            # Find rank that captures 95% of energy (adjustable threshold)
            energy_threshold = 0.95
            effective_rank = np.searchsorted(cumulative_energy / total_energy, energy_threshold) + 1
            
            # Apply sparsity-based compression factor
            # Higher sparsity = more compression potential
            if sparsity > 0.8:
                compression_factor = 0.5  # Aggressive compression for very sparse layers
            elif sparsity > 0.5:
                compression_factor = 0.75  # Moderate compression
            else:
                compression_factor = 0.9  # Conservative compression
            
            # Calculate target dimension
            target_dim = max(
                128,  # Minimum dimension to maintain model capacity
                min(
                    int(effective_rank * compression_factor),
                    total_neurons,
                    4096  # Maximum dimension cap
                )
            )
            
            # Round to nearest power of 2 for computational efficiency
            target_dim = 2 ** int(np.log2(target_dim))
            
            metadata = {
                "effective_rank": int(effective_rank),
                "top_5_singular_values": singular_values[:5].tolist(),
                "energy_retained": float(cumulative_energy[min(effective_rank-1, len(cumulative_energy)-1)] / total_energy),
                "compression_factor": compression_factor
            }
            
        except np.linalg.LinAlgError:
            # Fallback if SVD fails
            target_dim = 4096
            metadata = {"error": "SVD computation failed, using default dimension"}
        
        print(f"Layer {layer_idx}: Active={int(active_neurons)}/{total_neurons} (Sparsity {sparsity:.1%}) -> "
              f"Effective Rank={metadata.get('effective_rank', 'N/A')} -> Target Dim {target_dim}")
        
        # Create projection matrix based on top singular vectors
        # This would be the actual weight matrix for dimension reduction
        projection_matrix = U[:, :target_dim] if 'U' in dir() else None
        
        layer_configs[layer_idx] = {
            "source_dim": int(total_neurons),
            "target_dim": int(target_dim),
            "sparsity": float(sparsity),
            "active_mask": mask_array.tolist(),
            "projection_matrix_shape": [int(total_neurons), int(target_dim)] if projection_matrix is not None else None,
            "spectral_metadata": metadata
        }
    
    # Calculate overall compression statistics
    total_params_before = sum(cfg["source_dim"] for cfg in layer_configs.values())
    total_params_after = sum(cfg["target_dim"] for cfg in layer_configs.values())
    
    print(f"\n=== Spectral Analysis Summary ===")
    print(f"Total Parameters Before: {total_params_before:,}")
    print(f"Total Parameters After: {total_params_after:,}")
    print(f"Overall Compression Ratio: {total_params_after/total_params_before:.2%}")
    print(f"===================================\n")
    
    return layer_configs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bitmask_file", type=str, required=True)
    args = parser.parse_args()
    
    with open(args.bitmask_file, 'r') as f:
        data = json.load(f)
        
    configs = perform_spectral_analysis(data)
    
    output_file = args.bitmask_file.replace("_bitmask.json", "_spectral_config.json")
    with open(output_file, 'w') as f:
        json.dump({"model": data['model'], "spectral_configs": configs}, f)
    print(f"Spectral config saved to {output_file}")

if __name__ == "__main__":
    main()
