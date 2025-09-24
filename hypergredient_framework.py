#!/usr/bin/env python3
"""
Hypergredient Framework Architecture

Revolutionary formulation design system for cosmetic ingredients optimization.
Implements advanced algorithms for ingredient selection, compatibility analysis,
and multi-objective formulation optimization.

Based on the Hypergredient Framework specification:
Hypergredient(*) := {ingredient_i | function(*) ∈ F_i, 
                     constraints ∈ C_i, 
                     performance ∈ P_i}
"""

import json
import math
import random
from typing import Dict, List, Tuple, Any, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum


class HypergredientClass(Enum):
    """Core Hypergredient Classes"""
    CT = "H.CT"  # Cellular Turnover Agents
    CS = "H.CS"  # Collagen Synthesis Promoters
    AO = "H.AO"  # Antioxidant Systems
    BR = "H.BR"  # Barrier Repair Complex
    ML = "H.ML"  # Melanin Modulators
    HY = "H.HY"  # Hydration Systems
    AI = "H.AI"  # Anti-Inflammatory Agents
    MB = "H.MB"  # Microbiome Balancers
    SE = "H.SE"  # Sebum Regulators
    PD = "H.PD"  # Penetration/Delivery Enhancers


@dataclass
class Hypergredient:
    """Core Hypergredient data structure"""
    id: str
    name: str
    inci_name: str
    hypergredient_class: HypergredientClass
    primary_function: str
    secondary_functions: List[str] = field(default_factory=list)
    
    # Performance metrics
    efficacy_score: float = 0.0  # 0-10 scale
    bioavailability: float = 0.0  # 0-1 scale
    safety_score: float = 0.0  # 0-10 scale
    stability_index: float = 0.0  # 0-1 scale
    
    # Physical properties
    ph_min: float = 4.0
    ph_max: float = 9.0
    cost_per_gram: float = 0.0  # ZAR
    
    # Interactions
    incompatibilities: List[str] = field(default_factory=list)
    synergies: List[str] = field(default_factory=list)
    
    # Clinical evidence
    clinical_evidence_level: str = "moderate"  # weak, moderate, strong
    
    def calculate_composite_score(self, weights: Dict[str, float]) -> float:
        """Calculate weighted composite score"""
        metrics = {
            'efficacy': self.efficacy_score / 10.0,
            'bioavailability': self.bioavailability,
            'safety': self.safety_score / 10.0,
            'stability': self.stability_index,
            'cost_efficiency': 1.0 / (self.cost_per_gram + 1.0)  # Inverse cost
        }
        
        return sum(metrics.get(metric, 0) * weight 
                  for metric, weight in weights.items())


@dataclass
class FormulationConstraints:
    """Formulation constraints and requirements"""
    ph_range: Tuple[float, float] = (4.5, 7.0)
    total_actives_range: Tuple[float, float] = (5.0, 25.0)  # percentage
    max_budget: float = 1000.0  # ZAR
    max_irritation_score: float = 5.0
    required_stability_months: int = 24
    excluded_ingredients: List[str] = field(default_factory=list)
    preferred_ingredients: List[str] = field(default_factory=list)


@dataclass
class FormulationRequest:
    """User formulation request"""
    target_concerns: List[str]
    secondary_concerns: List[str] = field(default_factory=list)
    skin_type: str = "normal"
    budget: float = 1000.0
    preferences: List[str] = field(default_factory=list)
    constraints: FormulationConstraints = field(default_factory=FormulationConstraints)


@dataclass
class FormulationResult:
    """Optimized formulation result"""
    selected_hypergredients: Dict[str, Dict[str, Any]]
    total_cost: float
    predicted_efficacy: float
    safety_score: float
    stability_months: int
    synergy_score: float
    reasoning: Dict[str, str]


class HypergredientDatabase:
    """Dynamic Hypergredient Database"""
    
    def __init__(self):
        self.hypergredients: Dict[str, Hypergredient] = {}
        self.interaction_matrix: Dict[Tuple[str, str], float] = {}
        self._initialize_database()
        self._initialize_interactions()
    
    def _initialize_database(self):
        """Initialize database with core hypergredients"""
        
        # H.CT - Cellular Turnover Agents
        self.add_hypergredient(Hypergredient(
            id="tretinoin",
            name="Tretinoin",
            inci_name="Tretinoin",
            hypergredient_class=HypergredientClass.CT,
            primary_function="Accelerated cellular turnover",
            secondary_functions=["Collagen stimulation", "Hyperpigmentation reduction"],
            efficacy_score=10.0,
            bioavailability=0.85,
            safety_score=6.0,
            stability_index=0.3,  # UV-sensitive
            ph_min=5.5,
            ph_max=6.5,
            cost_per_gram=15.00,
            incompatibilities=["benzoyl_peroxide", "strong_acids"],
            clinical_evidence_level="strong"
        ))
        
        self.add_hypergredient(Hypergredient(
            id="bakuchiol",
            name="Bakuchiol",
            inci_name="Bakuchiol",
            hypergredient_class=HypergredientClass.CT,
            primary_function="Gentle retinol alternative",
            secondary_functions=["Antioxidant", "Anti-inflammatory"],
            efficacy_score=7.0,
            bioavailability=0.70,
            safety_score=9.0,
            stability_index=0.9,
            ph_min=4.0,
            ph_max=9.0,
            cost_per_gram=240.00,
            synergies=["vitamin_c", "niacinamide"],
            clinical_evidence_level="moderate"
        ))
        
        self.add_hypergredient(Hypergredient(
            id="retinol",
            name="Retinol",
            inci_name="Retinol",
            hypergredient_class=HypergredientClass.CT,
            primary_function="Cellular turnover stimulation",
            secondary_functions=["Wrinkle reduction", "Texture improvement"],
            efficacy_score=8.0,
            bioavailability=0.60,
            safety_score=7.0,
            stability_index=0.4,  # Oxygen-sensitive
            ph_min=5.5,
            ph_max=6.5,
            cost_per_gram=180.00,
            incompatibilities=["aha", "bha", "vitamin_c"],
            clinical_evidence_level="strong"
        ))
        
        # H.CS - Collagen Synthesis Promoters
        self.add_hypergredient(Hypergredient(
            id="matrixyl_3000",
            name="Matrixyl 3000",
            inci_name="Palmitoyl Tripeptide-1, Palmitoyl Tetrapeptide-7",
            hypergredient_class=HypergredientClass.CS,
            primary_function="Signal peptide collagen stimulation",
            secondary_functions=["Wrinkle reduction", "Skin firmness"],
            efficacy_score=9.0,
            bioavailability=0.75,
            safety_score=9.0,
            stability_index=0.8,
            ph_min=5.0,
            ph_max=7.0,
            cost_per_gram=120.00,
            synergies=["vitamin_c", "niacinamide"],
            clinical_evidence_level="strong"
        ))
        
        self.add_hypergredient(Hypergredient(
            id="vitamin_c_sap",
            name="Vitamin C (Sodium Ascorbyl Phosphate)",
            inci_name="Sodium Ascorbyl Phosphate",
            hypergredient_class=HypergredientClass.CS,
            primary_function="Stable vitamin C for collagen synthesis",
            secondary_functions=["Antioxidant", "Brightening"],
            efficacy_score=6.0,
            bioavailability=0.70,
            safety_score=9.0,
            stability_index=0.8,
            ph_min=6.0,
            ph_max=8.0,
            cost_per_gram=70.00,
            synergies=["niacinamide", "peptides"],
            clinical_evidence_level="moderate"
        ))
        
        # Re-add the original vitamin C LAA
        self.add_hypergredient(Hypergredient(
            id="vitamin_c_laa",
            name="Vitamin C (L-Ascorbic Acid)",
            inci_name="L-Ascorbic Acid",
            hypergredient_class=HypergredientClass.CS,
            primary_function="Collagen synthesis cofactor",
            secondary_functions=["Antioxidant", "Brightening"],
            efficacy_score=8.0,
            bioavailability=0.85,
            safety_score=7.0,
            stability_index=0.2,  # Very unstable
            ph_min=3.0,
            ph_max=4.0,
            cost_per_gram=85.00,
            incompatibilities=["copper_peptides", "retinol"],
            synergies=["vitamin_e", "ferulic_acid"],
            clinical_evidence_level="strong"
        ))
        
        # Add more antioxidants
        self.add_hypergredient(Hypergredient(
            id="resveratrol",
            name="Resveratrol",
            inci_name="Resveratrol",
            hypergredient_class=HypergredientClass.AO,
            primary_function="Polyphenol antioxidant",
            secondary_functions=["Anti-inflammatory", "Longevity"],
            efficacy_score=7.0,
            bioavailability=0.60,
            safety_score=8.0,
            stability_index=0.6,
            ph_min=4.0,
            ph_max=7.0,
            cost_per_gram=190.00,
            synergies=["vitamin_e", "ferulic_acid"],
            clinical_evidence_level="moderate"
        ))
        
        self.add_hypergredient(Hypergredient(
            id="ferulic_acid",
            name="Ferulic Acid",
            inci_name="Ferulic Acid",
            hypergredient_class=HypergredientClass.AO,
            primary_function="Antioxidant stabilizer",
            secondary_functions=["UV protection", "Vitamin C stabilizer"],
            efficacy_score=6.0,
            bioavailability=0.75,
            safety_score=9.0,
            stability_index=0.7,
            ph_min=4.0,
            ph_max=6.0,
            cost_per_gram=125.00,
            synergies=["vitamin_c_laa", "vitamin_e"],
            clinical_evidence_level="strong"
        ))
        
        # H.AO - Antioxidant Systems
        self.add_hypergredient(Hypergredient(
            id="astaxanthin",
            name="Astaxanthin",
            inci_name="Astaxanthin",
            hypergredient_class=HypergredientClass.AO,
            primary_function="Powerful antioxidant protection",
            secondary_functions=["UV protection", "Anti-inflammatory"],
            efficacy_score=9.0,
            bioavailability=0.65,
            safety_score=9.0,
            stability_index=0.6,  # Light-sensitive
            ph_min=4.0,
            ph_max=8.0,
            cost_per_gram=360.00,
            synergies=["vitamin_e", "resveratrol"],
            clinical_evidence_level="strong"
        ))
        
        self.add_hypergredient(Hypergredient(
            id="vitamin_e",
            name="Vitamin E",
            inci_name="Tocopherol",
            hypergredient_class=HypergredientClass.AO,
            primary_function="Lipid antioxidant protection",
            secondary_functions=["Stabilizer", "Moisturizer"],
            efficacy_score=6.0,
            bioavailability=0.90,
            safety_score=9.0,
            stability_index=0.9,
            ph_min=4.0,
            ph_max=9.0,
            cost_per_gram=50.00,
            synergies=["vitamin_c", "ferulic_acid"],
            clinical_evidence_level="strong"
        ))
        
        # Add more brightening agents  
        self.add_hypergredient(Hypergredient(
            id="tranexamic_acid",
            name="Tranexamic Acid",
            inci_name="Tranexamic Acid",
            hypergredient_class=HypergredientClass.ML,
            primary_function="Melasma and hyperpigmentation treatment",
            secondary_functions=["Anti-inflammatory", "Vascular protection"],
            efficacy_score=8.0,
            bioavailability=0.85,
            safety_score=9.0,
            stability_index=0.9,
            ph_min=4.0,
            ph_max=8.0,
            cost_per_gram=220.00,
            synergies=["vitamin_c", "niacinamide"],
            clinical_evidence_level="strong"
        ))
        
        self.add_hypergredient(Hypergredient(
            id="kojic_acid",
            name="Kojic Acid",
            inci_name="Kojic Acid",
            hypergredient_class=HypergredientClass.ML,
            primary_function="Tyrosinase inhibitor",
            secondary_functions=["Antioxidant"],
            efficacy_score=7.0,
            bioavailability=0.75,
            safety_score=7.0,
            stability_index=0.5,  # Can be unstable
            ph_min=4.0,
            ph_max=6.0,
            cost_per_gram=95.00,
            synergies=["alpha_arbutin", "vitamin_c"],
            clinical_evidence_level="moderate"
        ))
        
        # Add barrier repair ingredients
        self.add_hypergredient(Hypergredient(
            id="ceramide_np",
            name="Ceramide NP",
            inci_name="Ceramide NP",
            hypergredient_class=HypergredientClass.BR,
            primary_function="Barrier lipid restoration",
            secondary_functions=["Moisturizing", "Anti-aging"],
            efficacy_score=8.0,
            bioavailability=0.70,
            safety_score=10.0,
            stability_index=0.8,
            ph_min=4.0,
            ph_max=8.0,
            cost_per_gram=280.00,
            synergies=["cholesterol", "fatty_acids"],
            clinical_evidence_level="strong"
        ))
        
        self.add_hypergredient(Hypergredient(
            id="cholesterol",
            name="Cholesterol",
            inci_name="Cholesterol",
            hypergredient_class=HypergredientClass.BR,
            primary_function="Barrier lipid component",
            secondary_functions=["Membrane fluidity"],
            efficacy_score=6.0,
            bioavailability=0.60,
            safety_score=10.0,
            stability_index=0.9,
            ph_min=4.0,
            ph_max=9.0,
            cost_per_gram=85.00,
            synergies=["ceramide_np", "fatty_acids"],
            clinical_evidence_level="strong"
        ))
        
        # H.HY - Hydration Systems
        self.add_hypergredient(Hypergredient(
            id="hyaluronic_acid",
            name="Hyaluronic Acid",
            inci_name="Sodium Hyaluronate",
            hypergredient_class=HypergredientClass.HY,
            primary_function="Multi-depth hydration",
            secondary_functions=["Plumping", "Barrier support"],
            efficacy_score=8.0,
            bioavailability=0.85,
            safety_score=10.0,
            stability_index=0.9,
            ph_min=4.0,
            ph_max=9.0,
            cost_per_gram=150.00,
            synergies=["ceramides", "peptides"],
            clinical_evidence_level="strong"
        ))
        
        # H.AI - Anti-Inflammatory Agents
        self.add_hypergredient(Hypergredient(
            id="niacinamide",
            name="Niacinamide",
            inci_name="Niacinamide",
            hypergredient_class=HypergredientClass.AI,
            primary_function="Anti-inflammatory",
            secondary_functions=["Sebum regulation", "Barrier repair", "Brightening"],
            efficacy_score=8.0,
            bioavailability=0.90,
            safety_score=9.0,
            stability_index=0.95,
            ph_min=5.0,
            ph_max=7.0,
            cost_per_gram=45.00,
            synergies=["zinc", "peptides", "hyaluronic_acid"],
            clinical_evidence_level="strong"
        ))
    
    def add_hypergredient(self, hypergredient: Hypergredient):
        """Add hypergredient to database"""
        self.hypergredients[hypergredient.id] = hypergredient
    
    def _initialize_interactions(self):
        """Initialize interaction matrix"""
        interactions = {
            ("H.CT", "H.CS"): 1.5,  # Positive synergy
            ("H.CT", "H.AO"): 0.8,  # Mild antagonism (oxidation)
            ("H.CS", "H.AO"): 2.0,  # Strong synergy
            ("H.BR", "H.HY"): 2.5,  # Excellent synergy
            ("H.ML", "H.AO"): 1.8,  # Good synergy
            ("H.AI", "H.MB"): 2.2,  # Strong synergy
            ("H.SE", "H.CT"): 0.6,  # Potential irritation
            ("H.CS", "H.HY"): 1.6,  # Good synergy
            ("H.AI", "H.HY"): 1.4,  # Moderate synergy
        }
        
        # Create bidirectional interactions
        for (class1, class2), score in interactions.items():
            self.interaction_matrix[(class1, class2)] = score
            self.interaction_matrix[(class2, class1)] = score
    
    def get_by_class(self, hypergredient_class: HypergredientClass) -> List[Hypergredient]:
        """Get all hypergredients by class"""
        return [h for h in self.hypergredients.values() 
                if h.hypergredient_class == hypergredient_class]
    
    def search(self, query: str) -> List[Hypergredient]:
        """Search hypergredients by name or function"""
        query_lower = query.lower()
        results = []
        
        for hypergredient in self.hypergredients.values():
            if (query_lower in hypergredient.name.lower() or
                query_lower in hypergredient.primary_function.lower() or
                any(query_lower in func.lower() for func in hypergredient.secondary_functions)):
                results.append(hypergredient)
        
        return results


class HypergredientOptimizer:
    """Multi-Objective Formulation Optimizer"""
    
    def __init__(self, database: HypergredientDatabase):
        self.database = database
        self.concern_to_class_mapping = {
            'wrinkles': [HypergredientClass.CT, HypergredientClass.CS],
            'aging': [HypergredientClass.CT, HypergredientClass.CS, HypergredientClass.AO],
            'acne': [HypergredientClass.CT, HypergredientClass.AI, HypergredientClass.SE],
            'hyperpigmentation': [HypergredientClass.ML, HypergredientClass.AO],
            'dryness': [HypergredientClass.HY, HypergredientClass.BR],
            'sensitivity': [HypergredientClass.AI, HypergredientClass.BR],
            'dullness': [HypergredientClass.ML, HypergredientClass.AO, HypergredientClass.CT],
            'firmness': [HypergredientClass.CS, HypergredientClass.AO],
            'texture': [HypergredientClass.CT, HypergredientClass.HY],
            'redness': [HypergredientClass.AI, HypergredientClass.BR]
        }
    
    def optimize_formulation(self, request: FormulationRequest) -> FormulationResult:
        """Generate optimal formulation using hypergredients"""
        
        # Define objective weights
        objective_weights = {
            'efficacy': 0.35,
            'safety': 0.25,
            'stability': 0.20,
            'cost_efficiency': 0.15,
            'bioavailability': 0.05
        }
        
        selected_hypergredients = {}
        total_cost = 0.0
        reasoning = {}
        
        # Process each concern
        for concern in request.target_concerns + request.secondary_concerns:
            if concern not in self.concern_to_class_mapping:
                continue
            
            classes = self.concern_to_class_mapping[concern]
            weight = 1.0 if concern in request.target_concerns else 0.5
            
            for hypergredient_class in classes:
                if hypergredient_class.value in selected_hypergredients:
                    continue  # Already selected for this class
                
                candidates = self.database.get_by_class(hypergredient_class)
                if not candidates:
                    continue
                
                # Score each candidate
                best_candidate = None
                best_score = -1.0
                
                for candidate in candidates:
                    if candidate.id in request.constraints.excluded_ingredients:
                        continue
                    
                    # Check budget constraint
                    estimated_usage = self._estimate_usage_percentage(candidate)
                    estimated_cost = candidate.cost_per_gram * estimated_usage / 100.0 * 50  # 50g formulation
                    
                    if total_cost + estimated_cost > request.budget:
                        continue
                    
                    # Calculate compatibility with already selected ingredients
                    compatibility_score = self._calculate_compatibility(
                        candidate, list(selected_hypergredients.keys())
                    )
                    
                    if compatibility_score < 0.5:  # Too incompatible
                        continue
                    
                    # Calculate composite score
                    score = candidate.calculate_composite_score(objective_weights)
                    score *= weight * compatibility_score
                    
                    if score > best_score:
                        best_score = score
                        best_candidate = candidate
                
                if best_candidate:
                    usage_percentage = self._estimate_usage_percentage(best_candidate)
                    ingredient_cost = best_candidate.cost_per_gram * usage_percentage / 100.0 * 50
                    
                    selected_hypergredients[hypergredient_class.value] = {
                        'ingredient': best_candidate,
                        'percentage': usage_percentage,
                        'cost': ingredient_cost,
                        'score': best_score,
                        'reasoning': self._generate_reasoning(best_candidate, concern)
                    }
                    
                    total_cost += ingredient_cost
                    reasoning[hypergredient_class.value] = self._generate_reasoning(best_candidate, concern)
        
        # Calculate overall metrics
        efficacy_score = self._calculate_predicted_efficacy(selected_hypergredients)
        safety_score = self._calculate_overall_safety(selected_hypergredients)
        synergy_score = self._calculate_synergy_score(selected_hypergredients)
        stability_months = self._estimate_stability(selected_hypergredients)
        
        return FormulationResult(
            selected_hypergredients=selected_hypergredients,
            total_cost=total_cost,
            predicted_efficacy=efficacy_score,
            safety_score=safety_score,
            stability_months=stability_months,
            synergy_score=synergy_score,
            reasoning=reasoning
        )
    
    def _estimate_usage_percentage(self, hypergredient: Hypergredient) -> float:
        """Estimate typical usage percentage for hypergredient"""
        usage_map = {
            HypergredientClass.CT: 1.0,  # Low percentage actives
            HypergredientClass.CS: 3.0,  # Peptides typically 2-5%
            HypergredientClass.AO: 0.5,  # Strong antioxidants
            HypergredientClass.BR: 2.0,  # Barrier ingredients
            HypergredientClass.ML: 2.0,  # Brightening agents
            HypergredientClass.HY: 1.0,  # Hyaluronic acid
            HypergredientClass.AI: 5.0,  # Niacinamide can go higher
            HypergredientClass.MB: 1.0,  # Prebiotics/probiotics
            HypergredientClass.SE: 2.0,  # Sebum regulators
            HypergredientClass.PD: 1.0,  # Penetration enhancers
        }
        
        base_percentage = usage_map.get(hypergredient.hypergredient_class, 1.0)
        
        # Adjust based on potency and safety
        if hypergredient.efficacy_score > 8.0 and hypergredient.safety_score < 7.0:
            base_percentage *= 0.5  # Reduce for high potency, lower safety
        elif hypergredient.safety_score > 9.0:
            base_percentage *= 1.2  # Can use more of very safe ingredients
        
        return min(base_percentage, 10.0)  # Cap at 10%
    
    def _calculate_compatibility(self, candidate: Hypergredient, selected_ids: List[str]) -> float:
        """Calculate compatibility score with selected ingredients"""
        if not selected_ids:
            return 1.0
        
        compatibility_scores = []
        
        for selected_id in selected_ids:
            selected_class = None
            for hypergredient in self.database.hypergredients.values():
                if hypergredient.hypergredient_class.value == selected_id:
                    selected_class = hypergredient.hypergredient_class.value
                    break
            
            if selected_class:
                interaction_key = (candidate.hypergredient_class.value, selected_class)
                interaction_score = self.database.interaction_matrix.get(interaction_key, 1.0)
                compatibility_scores.append(min(interaction_score, 2.0) / 2.0)  # Normalize to 0-1
        
        return sum(compatibility_scores) / len(compatibility_scores) if compatibility_scores else 1.0
    
    def _calculate_predicted_efficacy(self, selected: Dict[str, Dict[str, Any]]) -> float:
        """Calculate predicted formulation efficacy"""
        if not selected:
            return 0.0
        
        efficacy_scores = []
        for data in selected.values():
            ingredient = data['ingredient']
            percentage = data['percentage']
            
            # Weight by usage percentage and bioavailability
            weighted_efficacy = (ingredient.efficacy_score / 10.0) * (percentage / 10.0) * ingredient.bioavailability
            efficacy_scores.append(weighted_efficacy)
        
        # Apply synergy bonus
        base_efficacy = sum(efficacy_scores) / len(efficacy_scores)
        synergy_bonus = self._calculate_synergy_score(selected) * 0.2
        
        return min(base_efficacy + synergy_bonus, 1.0)
    
    def _calculate_overall_safety(self, selected: Dict[str, Dict[str, Any]]) -> float:
        """Calculate overall formulation safety score"""
        if not selected:
            return 10.0
        
        safety_scores = [data['ingredient'].safety_score for data in selected.values()]
        return sum(safety_scores) / len(safety_scores)
    
    def _calculate_synergy_score(self, selected: Dict[str, Dict[str, Any]]) -> float:
        """Calculate synergy score between selected ingredients"""
        if len(selected) < 2:
            return 0.0
        
        synergy_scores = []
        classes = list(selected.keys())
        
        for i, class1 in enumerate(classes):
            for class2 in classes[i+1:]:
                interaction_key = (class1, class2)
                interaction_score = self.database.interaction_matrix.get(interaction_key, 1.0)
                if interaction_score > 1.0:  # Positive synergy
                    synergy_scores.append((interaction_score - 1.0) / 1.5)  # Normalize
        
        return sum(synergy_scores) / len(synergy_scores) if synergy_scores else 0.0
    
    def _estimate_stability(self, selected: Dict[str, Dict[str, Any]]) -> int:
        """Estimate formulation stability in months"""
        if not selected:
            return 24
        
        stability_indices = [data['ingredient'].stability_index for data in selected.values()]
        min_stability = min(stability_indices)
        
        # Convert stability index to months (0.0 = 6 months, 1.0 = 24 months)
        return int(6 + (min_stability * 18))
    
    def _generate_reasoning(self, hypergredient: Hypergredient, concern: str) -> str:
        """Generate reasoning for ingredient selection"""
        reasons = []
        
        if hypergredient.efficacy_score >= 8.0:
            reasons.append("High efficacy")
        if hypergredient.safety_score >= 9.0:
            reasons.append("Excellent safety profile")
        if hypergredient.stability_index >= 0.8:
            reasons.append("Good stability")
        if hypergredient.cost_per_gram <= 100.0:
            reasons.append("Cost-effective")
        if hypergredient.clinical_evidence_level == "strong":
            reasons.append("Strong clinical evidence")
        
        base_reason = f"Selected for {concern} targeting"
        if reasons:
            return f"{base_reason}: {', '.join(reasons)}"
        else:
            return base_reason


class HypergredientAnalyzer:
    """Analysis and reporting tools for hypergredients"""
    
    def __init__(self, database: HypergredientDatabase):
        self.database = database
    
    def generate_compatibility_report(self, ingredient_ids: List[str]) -> Dict[str, Any]:
        """Generate compatibility analysis report"""
        ingredients = [self.database.hypergredients[id] for id in ingredient_ids 
                      if id in self.database.hypergredients]
        
        if len(ingredients) < 2:
            return {"error": "Need at least 2 ingredients for compatibility analysis"}
        
        compatibility_matrix = {}
        warnings = []
        recommendations = []
        
        for i, ing1 in enumerate(ingredients):
            for ing2 in ingredients[i+1:]:
                key = f"{ing1.name} + {ing2.name}"
                
                # Check direct incompatibilities
                if (ing2.id in ing1.incompatibilities or 
                    ing1.id in ing2.incompatibilities):
                    compatibility_matrix[key] = "Incompatible"
                    warnings.append(f"⚠️ {ing1.name} and {ing2.name} are incompatible")
                    continue
                
                # Check pH compatibility
                ph_overlap = min(ing1.ph_max, ing2.ph_max) - max(ing1.ph_min, ing2.ph_min)
                if ph_overlap <= 0:
                    compatibility_matrix[key] = "pH Incompatible"
                    warnings.append(f"⚠️ {ing1.name} and {ing2.name} have incompatible pH ranges")
                    continue
                
                # Check for synergies
                class_interaction = self.database.interaction_matrix.get(
                    (ing1.hypergredient_class.value, ing2.hypergredient_class.value), 1.0
                )
                
                if class_interaction > 1.5:
                    compatibility_matrix[key] = "Excellent Synergy"
                    recommendations.append(f"✅ {ing1.name} and {ing2.name} work synergistically")
                elif class_interaction > 1.0:
                    compatibility_matrix[key] = "Good Synergy"
                elif class_interaction >= 0.8:
                    compatibility_matrix[key] = "Compatible"
                else:
                    compatibility_matrix[key] = "Potentially Problematic"
                    warnings.append(f"⚠️ {ing1.name} and {ing2.name} may interfere with each other")
        
        return {
            "compatibility_matrix": compatibility_matrix,
            "warnings": warnings,
            "recommendations": recommendations,
            "overall_compatibility": "Good" if not warnings else "Needs Attention"
        }
    
    def generate_ingredient_profile(self, ingredient_id: str) -> Dict[str, Any]:
        """Generate detailed ingredient profile"""
        if ingredient_id not in self.database.hypergredients:
            return {"error": f"Ingredient '{ingredient_id}' not found"}
        
        ingredient = self.database.hypergredients[ingredient_id]
        
        # Calculate derived metrics
        cost_efficiency = ingredient.efficacy_score / max(ingredient.cost_per_gram, 1.0)
        risk_benefit_ratio = ingredient.efficacy_score / max(10.0 - ingredient.safety_score, 1.0)
        
        return {
            "basic_info": {
                "name": ingredient.name,
                "inci_name": ingredient.inci_name,
                "class": ingredient.hypergredient_class.value,
                "primary_function": ingredient.primary_function,
                "secondary_functions": ingredient.secondary_functions
            },
            "performance_metrics": {
                "efficacy_score": ingredient.efficacy_score,
                "bioavailability": ingredient.bioavailability,
                "safety_score": ingredient.safety_score,
                "stability_index": ingredient.stability_index
            },
            "formulation_properties": {
                "ph_range": f"{ingredient.ph_min}-{ingredient.ph_max}",
                "cost_per_gram": ingredient.cost_per_gram,
                "typical_usage": f"{self._get_typical_usage(ingredient)}%"
            },
            "interactions": {
                "incompatibilities": ingredient.incompatibilities,
                "synergies": ingredient.synergies
            },
            "derived_metrics": {
                "cost_efficiency": round(cost_efficiency, 2),
                "risk_benefit_ratio": round(risk_benefit_ratio, 2),
                "clinical_evidence": ingredient.clinical_evidence_level
            }
        }
    
    def _get_typical_usage(self, ingredient: Hypergredient) -> float:
        """Get typical usage percentage for ingredient"""
        optimizer = HypergredientOptimizer(self.database)
        return optimizer._estimate_usage_percentage(ingredient)


def main():
    """Demo of hypergredient framework capabilities"""
    print("🧬 Hypergredient Framework Architecture Demo")
    print("=" * 50)
    
    # Initialize system
    database = HypergredientDatabase()
    optimizer = HypergredientOptimizer(database)
    analyzer = HypergredientAnalyzer(database)
    
    print(f"\nInitialized database with {len(database.hypergredients)} hypergredients")
    print(f"Hypergredient classes: {[cls.value for cls in HypergredientClass]}")
    
    # Demo 1: Generate anti-aging formulation
    print("\n1. Anti-Aging Formulation Optimization")
    print("-" * 40)
    
    anti_aging_request = FormulationRequest(
        target_concerns=['wrinkles', 'firmness'],
        secondary_concerns=['dryness', 'dullness'],
        skin_type='normal_to_dry',
        budget=800.0,
        preferences=['gentle', 'stable']
    )
    
    result = optimizer.optimize_formulation(anti_aging_request)
    
    print(f"✓ Generated formulation with {len(result.selected_hypergredients)} hypergredients")
    print(f"  Total cost: R{result.total_cost:.2f}")
    print(f"  Predicted efficacy: {result.predicted_efficacy:.2%}")
    print(f"  Safety score: {result.safety_score:.1f}/10")
    print(f"  Synergy score: {result.synergy_score:.2f}")
    print(f"  Stability: {result.stability_months} months")
    
    print("\nSelected Hypergredients:")
    for class_name, data in result.selected_hypergredients.items():
        ingredient = data['ingredient']
        print(f"  • {class_name}: {ingredient.name} ({data['percentage']:.1f}%)")
        print(f"    Reasoning: {data['reasoning']}")
    
    # Demo 2: Compatibility analysis
    print("\n2. Compatibility Analysis")
    print("-" * 40)
    
    test_ingredients = ['retinol', 'vitamin_c_laa', 'niacinamide']
    compatibility_report = analyzer.generate_compatibility_report(test_ingredients)
    
    print("Compatibility Matrix:")
    for pair, status in compatibility_report['compatibility_matrix'].items():
        print(f"  {pair}: {status}")
    
    if compatibility_report['warnings']:
        print("\nWarnings:")
        for warning in compatibility_report['warnings']:
            print(f"  {warning}")
    
    if compatibility_report['recommendations']:
        print("\nRecommendations:")
        for rec in compatibility_report['recommendations']:
            print(f"  {rec}")
    
    # Demo 3: Ingredient profile
    print("\n3. Ingredient Profile Analysis")
    print("-" * 40)
    
    profile = analyzer.generate_ingredient_profile('bakuchiol')
    print(f"Ingredient: {profile['basic_info']['name']}")
    print(f"Class: {profile['basic_info']['class']}")
    print(f"Function: {profile['basic_info']['primary_function']}")
    print(f"Efficacy: {profile['performance_metrics']['efficacy_score']}/10")
    print(f"Safety: {profile['performance_metrics']['safety_score']}/10")
    print(f"Cost efficiency: {profile['derived_metrics']['cost_efficiency']}")
    
    # Save demo results
    demo_results = {
        "formulation_result": {
            "request": {
                "target_concerns": anti_aging_request.target_concerns,
                "budget": anti_aging_request.budget,
                "skin_type": anti_aging_request.skin_type
            },
            "result": {
                "total_cost": result.total_cost,
                "predicted_efficacy": result.predicted_efficacy,
                "safety_score": result.safety_score,
                "synergy_score": result.synergy_score,
                "stability_months": result.stability_months,
                "selected_ingredients": {
                    class_name: {
                        "name": data['ingredient'].name,
                        "percentage": data['percentage'],
                        "cost": data['cost'],
                        "reasoning": data['reasoning']
                    }
                    for class_name, data in result.selected_hypergredients.items()
                }
            }
        },
        "compatibility_analysis": compatibility_report,
        "ingredient_profile": profile
    }
    
    with open("hypergredient_demo_results.json", "w") as f:
        json.dump(demo_results, f, indent=2, default=str)
    
    print(f"\n✓ Demo results saved to hypergredient_demo_results.json")
    print("\nHypergredient Framework successfully demonstrates:")
    print("• Multi-objective formulation optimization")
    print("• Real-time compatibility analysis")
    print("• Ingredient profiling and scoring")
    print("• Synergy calculation and recommendations")


if __name__ == "__main__":
    main()