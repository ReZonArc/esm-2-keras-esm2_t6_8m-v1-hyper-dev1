"""
Tensor Shape Type System with Prime Factorization
Represents tensor dimensions as prime factorizations to enable unique shape expressions
and create a typed hypergraph (metagraph) representation.
"""

from typing import Dict, List, Tuple, Set, Any, Optional
from dataclasses import dataclass
from collections import defaultdict
import math


def prime_factorization(n: int) -> List[int]:
    """Compute prime factorization of a positive integer.
    
    Args:
        n: Positive integer to factorize
        
    Returns:
        List of prime factors in ascending order
    """
    if n <= 1:
        return []
    
    factors = []
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors.append(d)
            n //= d
        d += 1
    if n > 1:
        factors.append(n)
    return factors


def factorization_signature(factors: List[int]) -> str:
    """Create a unique string signature from prime factorization.
    
    Args:
        factors: List of prime factors
        
    Returns:
        String signature like "2^3*3^1*5^2"
    """
    if not factors:
        return "1"
    
    factor_counts = defaultdict(int)
    for f in factors:
        factor_counts[f] += 1
    
    signature_parts = []
    for prime in sorted(factor_counts.keys()):
        count = factor_counts[prime]
        if count == 1:
            signature_parts.append(str(prime))
        else:
            signature_parts.append(f"{prime}^{count}")
    
    return "*".join(signature_parts)


@dataclass
class TensorShapeType:
    """Represents tensor shape as prime factorizations for each dimension.
    
    This creates a unique type identifier for each topologically distinct
    tensor shape, enabling type-based clustering in the metagraph.
    """
    dimensions: Tuple[Optional[int], ...]  # Original dimensions (None for batch)
    prime_factors: Tuple[List[int], ...]   # Prime factorization per dimension
    type_signature: str                    # Unique type identifier
    canonical_form: str                    # Canonical mathematical representation
    
    def __post_init__(self):
        """Validate consistency between dimensions and prime factors."""
        if len(self.dimensions) != len(self.prime_factors):
            raise ValueError("Dimensions and prime factors must have same length")
        
        for dim, factors in zip(self.dimensions, self.prime_factors):
            if dim is not None:
                expected_product = math.prod(factors) if factors else 1
                if dim != expected_product:
                    raise ValueError(f"Dimension {dim} doesn't match prime factors {factors}")


def create_tensor_shape_type(shape: Tuple[Optional[int], ...]) -> TensorShapeType:
    """Create a TensorShapeType from a tensor shape tuple.
    
    Args:
        shape: Tensor shape tuple, with None for batch dimensions
        
    Returns:
        TensorShapeType with prime factorization representation
    """
    prime_factors = []
    signature_parts = []
    canonical_parts = []
    
    for i, dim in enumerate(shape):
        if dim is None:
            # Batch dimension - represented as variable
            prime_factors.append([])
            signature_parts.append("B")
            canonical_parts.append("B")
        else:
            factors = prime_factorization(dim)
            prime_factors.append(factors)
            
            if factors:
                sig = factorization_signature(factors)
                signature_parts.append(sig)
                canonical_parts.append(f"({sig})")
            else:
                signature_parts.append("1")
                canonical_parts.append("1")
    
    type_signature = "×".join(signature_parts)
    canonical_form = " ⊗ ".join(canonical_parts)
    
    return TensorShapeType(
        dimensions=shape,
        prime_factors=tuple(prime_factors),
        type_signature=type_signature,
        canonical_form=canonical_form
    )


class TensorShapeTypeRegistry:
    """Registry for tensor shape types to enable type-based clustering."""
    
    def __init__(self):
        self.types: Dict[str, TensorShapeType] = {}
        self.type_clusters: Dict[str, Set[str]] = defaultdict(set)
        self.dimension_families: Dict[int, Set[str]] = defaultdict(set)
        
    def register_shape_type(self, shape: Tuple[Optional[int], ...], node_id: str) -> TensorShapeType:
        """Register a tensor shape type and associate it with a node.
        
        Args:
            shape: Tensor shape tuple
            node_id: ID of the node with this shape
            
        Returns:
            TensorShapeType for the shape
        """
        shape_type = create_tensor_shape_type(shape)
        signature = shape_type.type_signature
        
        # Register the type if not already present
        if signature not in self.types:
            self.types[signature] = shape_type
        
        # Add node to the type cluster
        self.type_clusters[signature].add(node_id)
        
        # Add to dimension family clusters
        rank = len(shape)
        self.dimension_families[rank].add(signature)
        
        return shape_type
        
    def get_type_clusters(self) -> Dict[str, Set[str]]:
        """Get nodes clustered by tensor shape type."""
        return dict(self.type_clusters)
        
    def get_dimension_families(self) -> Dict[int, Set[str]]:
        """Get tensor types grouped by dimensionality (rank)."""
        return dict(self.dimension_families)
        
    def get_compatible_types(self, signature: str) -> Set[str]:
        """Find tensor types compatible for operations with given type.
        
        Compatible types share the same rank and similar structure.
        """
        if signature not in self.types:
            return set()
            
        shape_type = self.types[signature]
        rank = len(shape_type.dimensions)
        
        # Find types with same rank
        compatible = set()
        for other_sig in self.dimension_families[rank]:
            if other_sig != signature:
                other_type = self.types[other_sig]
                # Check if shapes are broadcast-compatible
                if self._are_broadcast_compatible(shape_type, other_type):
                    compatible.add(other_sig)
                    
        return compatible
        
    def _are_broadcast_compatible(self, type1: TensorShapeType, type2: TensorShapeType) -> bool:
        """Check if two tensor shape types are broadcast compatible."""
        if len(type1.dimensions) != len(type2.dimensions):
            return False
            
        for dim1, dim2 in zip(type1.dimensions, type2.dimensions):
            # None (batch) dimensions are always compatible
            if dim1 is None or dim2 is None:
                continue
            # Same dimensions are compatible
            if dim1 == dim2:
                continue
            # Dimension 1 is broadcastable
            if dim1 == 1 or dim2 == 1:
                continue
            # Otherwise incompatible
            return False
            
        return True


def analyze_tensor_type_distribution(registry: TensorShapeTypeRegistry) -> Dict[str, Any]:
    """Analyze the distribution of tensor types in the registry.
    
    Args:
        registry: TensorShapeTypeRegistry to analyze
        
    Returns:
        Analysis report with type statistics and patterns
    """
    type_clusters = registry.get_type_clusters()
    dimension_families = registry.get_dimension_families()
    
    # Count nodes per type
    type_node_counts = {sig: len(nodes) for sig, nodes in type_clusters.items()}
    
    # Analyze dimension patterns
    dimension_stats = {}
    for rank, type_sigs in dimension_families.items():
        dimension_stats[rank] = {
            'type_count': len(type_sigs),
            'node_count': sum(len(type_clusters[sig]) for sig in type_sigs),
            'types': list(type_sigs)
        }
    
    # Find most common types
    common_types = sorted(type_node_counts.items(), key=lambda x: x[1], reverse=True)
    
    # Identify unique mathematical structures
    unique_structures = set()
    for shape_type in registry.types.values():
        unique_structures.add(shape_type.canonical_form)
    
    return {
        'total_types': len(registry.types),
        'total_nodes': sum(len(nodes) for nodes in type_clusters.values()),
        'type_distribution': type_node_counts,
        'dimension_families': dimension_stats,
        'most_common_types': common_types[:10],
        'unique_mathematical_structures': len(unique_structures),
        'canonical_forms': sorted(unique_structures)
    }