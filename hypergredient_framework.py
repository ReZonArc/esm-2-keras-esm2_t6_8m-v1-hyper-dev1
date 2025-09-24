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


class FormulationEvolution:
    """Evolutionary Formulation Improvement System"""
    
    def __init__(self, base_formula: FormulationResult):
        self.generation = 0
        self.formula = base_formula
        self.performance_history = []
        self.market_feedback = []
    
    def add_market_feedback(self, feedback: Dict[str, Any]):
        """Add market feedback for evolutionary improvement"""
        self.market_feedback.append({
            'generation': self.generation,
            'feedback': feedback,
            'timestamp': self.generation  # Simplified timestamp
        })
    
    def evolve(self, database: HypergredientDatabase, 
               target_improvements: Dict[str, float]) -> FormulationResult:
        """
        Evolve formulation based on feedback and target improvements
        
        Args:
            database: Hypergredient database
            target_improvements: Dict of metrics to improve with target values
        """
        # Analyze performance gaps
        gaps = self._analyze_performance_gaps(target_improvements)
        
        # Search for better hypergredients
        optimizer = HypergredientOptimizer(database)
        
        # Create enhanced request based on gaps
        enhanced_request = self._create_enhanced_request(gaps)
        
        # Generate next generation formula
        next_gen_formula = optimizer.optimize_formulation(enhanced_request)
        
        # Track performance history
        self.performance_history.append({
            'generation': self.generation,
            'efficacy': self.formula.predicted_efficacy,
            'safety': self.formula.safety_score,
            'synergy': self.formula.synergy_score,
            'cost': self.formula.total_cost
        })
        
        # Update formula and increment generation
        self.formula = next_gen_formula
        self.generation += 1
        
        return next_gen_formula
    
    def _analyze_performance_gaps(self, targets: Dict[str, float]) -> Dict[str, Dict]:
        """Analyze gaps between current performance and targets"""
        current_metrics = {
            'efficacy': self.formula.predicted_efficacy,
            'safety': self.formula.safety_score / 10.0,  # Normalize to 0-1
            'synergy': self.formula.synergy_score,
            'cost_efficiency': 1.0 / (self.formula.total_cost / 1000.0)  # Inverse normalized cost
        }
        
        gaps = {}
        for metric, target in targets.items():
            if metric in current_metrics:
                gap = target - current_metrics[metric]
                if gap > 0:  # Only consider improvements needed
                    gaps[metric] = {
                        'current': current_metrics[metric],
                        'target': target,
                        'gap': gap,
                        'priority': gap / max(current_metrics[metric], 0.1)  # Relative gap
                    }
        
        return gaps
    
    def _create_enhanced_request(self, gaps: Dict[str, Dict]) -> FormulationRequest:
        """Create enhanced request based on performance gaps"""
        # Start with original concerns but add new ones based on gaps
        concerns = ['wrinkles', 'firmness']
        secondary_concerns = ['dryness']
        
        if 'efficacy' in gaps and gaps['efficacy']['gap'] > 0.1:
            concerns.extend(['aging', 'texture'])
        
        if 'safety' in gaps and gaps['safety']['gap'] > 0.1:
            secondary_concerns.append('sensitivity')
        
        # Adjust budget based on cost efficiency needs
        budget = 1000.0
        if 'cost_efficiency' in gaps:
            budget = max(800.0, budget - (gaps['cost_efficiency']['gap'] * 200))
        
        return FormulationRequest(
            target_concerns=concerns,
            secondary_concerns=secondary_concerns,
            skin_type='normal',
            budget=budget,
            preferences=['gentle', 'effective', 'evolved']
        )
    
    def get_evolution_report(self) -> Dict[str, Any]:
        """Generate comprehensive evolution report"""
        return {
            'current_generation': self.generation,
            'performance_history': self.performance_history,
            'market_feedback': self.market_feedback,
            'current_formula': {
                'total_cost': self.formula.total_cost,
                'predicted_efficacy': self.formula.predicted_efficacy,
                'safety_score': self.formula.safety_score,
                'synergy_score': self.formula.synergy_score,
                'stability_months': self.formula.stability_months,
                'selected_ingredients': {
                    class_name: {
                        'name': data['ingredient'].name,
                        'percentage': data['percentage'],
                        'reasoning': data['reasoning']
                    }
                    for class_name, data in self.formula.selected_hypergredients.items()
                }
            },
            'evolution_metrics': self._calculate_evolution_metrics()
        }
    
    def _calculate_evolution_metrics(self) -> Dict[str, float]:
        """Calculate evolution performance metrics"""
        if len(self.performance_history) < 2:
            return {'evolution_not_available': True}
        
        first = self.performance_history[0]
        latest = self.performance_history[-1]
        
        return {
            'efficacy_improvement': latest['efficacy'] - first['efficacy'],
            'safety_improvement': latest['safety'] - first['safety'],
            'synergy_improvement': latest['synergy'] - first['synergy'],
            'cost_change': latest['cost'] - first['cost'],
            'generations_evolved': len(self.performance_history)
        }


class HypergredientAI:
    """Machine Learning Integration for Hypergredient Prediction"""
    
    def __init__(self):
        self.model_version = "v1.0"
        self.confidence_threshold = 0.7
        self.feedback_data = []
    
    def predict_optimal_combination(self, requirements: FormulationRequest) -> Dict[str, Any]:
        """Predict best hypergredient combinations using simulated ML"""
        
        # Simulate feature extraction
        features = self._extract_features(requirements)
        
        # Simulate ML predictions (in real implementation, this would use trained models)
        predictions = self._simulate_ml_predictions(features)
        
        # Rank by confidence
        ranked_predictions = sorted(predictions, key=lambda x: x['confidence'], reverse=True)
        
        return {
            'model_version': self.model_version,
            'predictions': ranked_predictions[:5],  # Top 5 predictions
            'confidence_scores': {pred['ingredient_class']: pred['confidence'] for pred in ranked_predictions[:5]},
            'feature_importance': self._get_feature_importance(features)
        }
    
    def _extract_features(self, requirements: FormulationRequest) -> Dict[str, float]:
        """Extract features from formulation requirements"""
        
        # Concern encoding
        concern_weights = {
            'wrinkles': 1.0, 'aging': 0.9, 'firmness': 0.8, 'dryness': 0.7,
            'acne': 0.6, 'sensitivity': 0.5, 'hyperpigmentation': 0.8
        }
        
        concern_score = sum(concern_weights.get(concern, 0.3) 
                          for concern in requirements.target_concerns + requirements.secondary_concerns)
        
        # Skin type encoding
        skin_type_scores = {
            'oily': 0.2, 'dry': 0.8, 'sensitive': 0.9, 
            'normal': 0.5, 'combination': 0.6
        }
        
        features = {
            'concern_complexity': concern_score,
            'budget_normalized': min(requirements.budget / 1500.0, 1.0),
            'skin_sensitivity': skin_type_scores.get(requirements.skin_type, 0.5),
            'preference_gentleness': 1.0 if 'gentle' in requirements.preferences else 0.3,
            'preference_effectiveness': 1.0 if 'effective' in requirements.preferences else 0.7
        }
        
        return features
    
    def _simulate_ml_predictions(self, features: Dict[str, float]) -> List[Dict[str, Any]]:
        """Simulate ML model predictions"""
        import random
        
        # Simulate predictions for different hypergredient classes
        base_predictions = [
            {'ingredient_class': 'H.CT', 'base_confidence': 0.8},
            {'ingredient_class': 'H.CS', 'base_confidence': 0.9},
            {'ingredient_class': 'H.AO', 'base_confidence': 0.7},
            {'ingredient_class': 'H.ML', 'base_confidence': 0.6},
            {'ingredient_class': 'H.HY', 'base_confidence': 0.85},
            {'ingredient_class': 'H.AI', 'base_confidence': 0.75},
            {'ingredient_class': 'H.BR', 'base_confidence': 0.65}
        ]
        
        predictions = []
        for pred in base_predictions:
            # Adjust confidence based on features
            confidence_adjustment = (
                features['concern_complexity'] * 0.1 +
                features['budget_normalized'] * 0.1 +
                features['skin_sensitivity'] * 0.05 +
                features['preference_gentleness'] * 0.05
            )
            
            adjusted_confidence = min(pred['base_confidence'] + confidence_adjustment, 1.0)
            
            predictions.append({
                'ingredient_class': pred['ingredient_class'],
                'confidence': adjusted_confidence,
                'reasoning': self._generate_ml_reasoning(pred['ingredient_class'], features)
            })
        
        return predictions
    
    def _generate_ml_reasoning(self, ingredient_class: str, features: Dict[str, float]) -> str:
        """Generate reasoning for ML predictions"""
        reasons = []
        
        if features['concern_complexity'] > 0.8:
            reasons.append("High concern complexity detected")
        if features['skin_sensitivity'] > 0.7:
            reasons.append("Sensitive skin considerations")
        if features['preference_gentleness'] > 0.8:
            reasons.append("Gentleness prioritized")
        
        class_specific = {
            'H.CT': "Strong anti-aging efficacy predicted",
            'H.CS': "Collagen synthesis highly beneficial",
            'H.AO': "Antioxidant protection recommended",
            'H.ML': "Brightening effects suitable",
            'H.HY': "Hydration enhancement predicted",
            'H.AI': "Anti-inflammatory benefits expected",
            'H.BR': "Barrier repair highly recommended"
        }
        
        base_reason = class_specific.get(ingredient_class, "Standard recommendation")
        if reasons:
            return f"{base_reason}; {'; '.join(reasons)}"
        return base_reason
    
    def _get_feature_importance(self, features: Dict[str, float]) -> Dict[str, float]:
        """Get feature importance scores"""
        # Simulate feature importance (in real ML, this would come from model)
        importance = {
            'concern_complexity': 0.35,
            'skin_sensitivity': 0.25,
            'budget_normalized': 0.20,
            'preference_gentleness': 0.15,
            'preference_effectiveness': 0.05
        }
        return importance
    
    def update_from_results(self, formulation_id: str, results: Dict[str, Any]):
        """Update model from real-world results"""
        feedback_entry = {
            'formulation_id': formulation_id,
            'results': results,
            'timestamp': len(self.feedback_data),  # Simplified timestamp
            'model_version': self.model_version
        }
        
        self.feedback_data.append(feedback_entry)
        
        # Simulate model retraining trigger
        if len(self.feedback_data) >= 100:  # Retrain after 100 data points
            self._simulate_model_retraining()
    
    def _simulate_model_retraining(self):
        """Simulate model retraining process"""
        # In real implementation, this would retrain the ML model
        self.model_version = f"v{float(self.model_version[1:]) + 0.1:.1f}"
        print(f"🤖 Model retrained to version {self.model_version}")


class HypergredientVisualizer:
    """Visualization Dashboard for Hypergredient Framework"""
    
    def __init__(self, database: HypergredientDatabase):
        self.database = database
    
    def generate_formulation_report(self, formulation: FormulationResult, 
                                  request: FormulationRequest) -> Dict[str, Any]:
        """Create comprehensive visual report for formulation"""
        
        report = {
            "title": "🧬 Hypergredient Formulation Analysis Report",
            "timestamp": "Generated with Hypergredient Framework v1.0",
            "formulation_overview": self._create_formulation_overview(formulation, request),
            "performance_radar": self._create_performance_radar(formulation),
            "ingredient_breakdown": self._create_ingredient_breakdown(formulation),
            "cost_analysis": self._create_cost_analysis(formulation),
            "synergy_network": self._create_synergy_network(formulation),
            "risk_assessment": self._create_risk_assessment(formulation),
            "recommendations": self._generate_recommendations(formulation)
        }
        
        return report
    
    def _create_formulation_overview(self, formulation: FormulationResult, 
                                   request: FormulationRequest) -> Dict[str, Any]:
        """Create formulation overview section"""
        return {
            "formulation_id": f"HF-{hash(str(formulation.selected_hypergredients)) % 10000:04d}",
            "target_concerns": request.target_concerns,
            "skin_type": request.skin_type,
            "budget_allocated": f"R{formulation.total_cost:.2f} / R{request.budget:.2f}",
            "budget_utilization": f"{(formulation.total_cost / request.budget) * 100:.1f}%",
            "total_ingredients": len(formulation.selected_hypergredients),
            "predicted_outcomes": {
                "efficacy": f"{formulation.predicted_efficacy:.1%}",
                "safety_score": f"{formulation.safety_score:.1f}/10",
                "synergy_bonus": f"{formulation.synergy_score:.2f}",
                "stability": f"{formulation.stability_months} months"
            }
        }
    
    def _create_performance_radar(self, formulation: FormulationResult) -> Dict[str, Any]:
        """Create performance radar chart data"""
        metrics = {
            "Efficacy": formulation.predicted_efficacy * 100,  # Convert to percentage
            "Safety": (formulation.safety_score / 10) * 100,
            "Synergy": formulation.synergy_score * 100,
            "Stability": min((formulation.stability_months / 24) * 100, 100),  # 24 months = 100%
            "Cost Efficiency": max(100 - (formulation.total_cost / 1000 * 100), 0)  # Inverse cost
        }
        
        return {
            "chart_type": "radar",
            "data": metrics,
            "description": "Multi-dimensional performance analysis",
            "interpretation": {
                "strengths": [k for k, v in metrics.items() if v > 70],
                "areas_for_improvement": [k for k, v in metrics.items() if v < 50]
            }
        }
    
    def _create_ingredient_breakdown(self, formulation: FormulationResult) -> Dict[str, Any]:
        """Create detailed ingredient breakdown"""
        ingredients = []
        
        for class_name, data in formulation.selected_hypergredients.items():
            ingredient = data['ingredient']
            ingredients.append({
                "class": class_name,
                "name": ingredient.name,
                "inci_name": ingredient.inci_name,
                "percentage": data['percentage'],
                "cost": data['cost'],
                "cost_per_percent": data['cost'] / data['percentage'] if data['percentage'] > 0 else 0,
                "efficacy_score": ingredient.efficacy_score,
                "safety_score": ingredient.safety_score,
                "primary_function": ingredient.primary_function,
                "secondary_functions": ingredient.secondary_functions,
                "reasoning": data['reasoning']
            })
        
        # Sort by cost contribution
        ingredients.sort(key=lambda x: x['cost'], reverse=True)
        
        return {
            "ingredients": ingredients,
            "summary": {
                "total_actives_percentage": sum(ing['percentage'] for ing in ingredients),
                "most_expensive": max(ingredients, key=lambda x: x['cost'])['name'],
                "highest_efficacy": max(ingredients, key=lambda x: x['efficacy_score'])['name'],
                "safest_ingredient": max(ingredients, key=lambda x: x['safety_score'])['name']
            }
        }
    
    def _create_cost_analysis(self, formulation: FormulationResult) -> Dict[str, Any]:
        """Create cost breakdown analysis"""
        cost_breakdown = []
        total_cost = formulation.total_cost
        
        for class_name, data in formulation.selected_hypergredients.items():
            cost_breakdown.append({
                "category": class_name,
                "ingredient": data['ingredient'].name,
                "cost": data['cost'],
                "percentage_of_budget": (data['cost'] / total_cost) * 100,
                "cost_per_gram": data['ingredient'].cost_per_gram,
                "usage_amount": data['percentage']
            })
        
        cost_breakdown.sort(key=lambda x: x['cost'], reverse=True)
        
        return {
            "breakdown": cost_breakdown,
            "cost_efficiency_metrics": {
                "cost_per_efficacy_point": total_cost / (formulation.predicted_efficacy * 100) if formulation.predicted_efficacy > 0 else float('inf'),
                "cost_per_safety_point": total_cost / formulation.safety_score,
                "premium_ingredients": [item for item in cost_breakdown if item['cost_per_gram'] > 200],
                "budget_friendly": [item for item in cost_breakdown if item['cost_per_gram'] < 100]
            }
        }
    
    def _create_synergy_network(self, formulation: FormulationResult) -> Dict[str, Any]:
        """Create synergy network analysis"""
        interactions = []
        classes = list(formulation.selected_hypergredients.keys())
        
        for i, class1 in enumerate(classes):
            for class2 in classes[i+1:]:
                interaction_score = self.database.interaction_matrix.get((class1, class2), 1.0)
                
                if interaction_score != 1.0:  # Only include non-neutral interactions
                    interactions.append({
                        "source": class1,
                        "target": class2,
                        "strength": interaction_score,
                        "type": "synergy" if interaction_score > 1.0 else "antagonism",
                        "description": self._describe_interaction(class1, class2, interaction_score)
                    })
        
        return {
            "interactions": interactions,
            "network_strength": formulation.synergy_score,
            "positive_interactions": len([i for i in interactions if i['strength'] > 1.0]),
            "negative_interactions": len([i for i in interactions if i['strength'] < 1.0]),
            "network_description": self._describe_network_quality(formulation.synergy_score)
        }
    
    def _create_risk_assessment(self, formulation: FormulationResult) -> Dict[str, Any]:
        """Create comprehensive risk assessment"""
        risks = []
        warnings = []
        
        # Analyze individual ingredient risks
        for class_name, data in formulation.selected_hypergredients.items():
            ingredient = data['ingredient']
            
            # Safety score analysis
            if ingredient.safety_score < 7.0:
                risks.append({
                    "level": "moderate",
                    "ingredient": ingredient.name,
                    "concern": "Lower safety score",
                    "recommendation": "Consider patch testing and gradual introduction"
                })
            
            # Stability analysis
            if ingredient.stability_index < 0.5:
                warnings.append({
                    "ingredient": ingredient.name,
                    "issue": "Stability concerns",
                    "recommendation": "Store in cool, dark conditions. Use within stability period."
                })
            
            # pH compatibility
            ph_range_size = ingredient.ph_max - ingredient.ph_min
            if ph_range_size < 2.0:
                warnings.append({
                    "ingredient": ingredient.name,
                    "issue": f"Narrow pH range ({ingredient.ph_min}-{ingredient.ph_max})",
                    "recommendation": "Careful pH balancing required in formulation"
                })
        
        # Overall formulation risks
        overall_risk_level = "low"
        if formulation.safety_score < 7.0:
            overall_risk_level = "high" 
        elif formulation.safety_score < 8.5:
            overall_risk_level = "moderate"
        
        return {
            "overall_risk_level": overall_risk_level,
            "safety_score": formulation.safety_score,
            "individual_risks": risks,
            "formulation_warnings": warnings,
            "recommendations": self._generate_safety_recommendations(formulation, risks, warnings)
        }
    
    def _generate_recommendations(self, formulation: FormulationResult) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        if formulation.predicted_efficacy < 0.15:
            recommendations.append("⚡ Consider adding more potent actives to improve efficacy")
        
        if formulation.synergy_score < 0.3:
            recommendations.append("🔄 Review ingredient combinations to enhance synergistic effects")
        
        if formulation.stability_months < 18:
            recommendations.append("🛡️ Add stabilizing ingredients or improve packaging to extend shelf life")
        
        if formulation.total_cost > 800:
            recommendations.append("💰 Consider cost-effective alternatives to reduce formulation cost")
        
        if formulation.safety_score < 8.5:
            recommendations.append("⚠️ Conduct additional safety testing before market release")
        
        if len(formulation.selected_hypergredients) < 4:
            recommendations.append("🌟 Consider additional complementary ingredients for comprehensive benefits")
        
        return recommendations
    
    def _describe_interaction(self, class1: str, class2: str, score: float) -> str:
        """Describe interaction between two ingredient classes"""
        if score > 1.5:
            return f"{class1} and {class2} work synergistically to enhance overall performance"
        elif score > 1.0:
            return f"{class1} and {class2} have complementary benefits"
        elif score < 0.8:
            return f"{class1} and {class2} may interfere with each other's effectiveness"
        else:
            return f"{class1} and {class2} have neutral interaction"
    
    def _describe_network_quality(self, synergy_score: float) -> str:
        """Describe overall network quality"""
        if synergy_score > 0.6:
            return "Excellent synergistic network with strong ingredient interactions"
        elif synergy_score > 0.4:
            return "Good ingredient network with moderate synergistic effects"
        elif synergy_score > 0.2:
            return "Fair ingredient network with limited synergistic benefits"
        else:
            return "Weak ingredient network requiring optimization for better synergy"
    
    def _generate_safety_recommendations(self, formulation: FormulationResult, 
                                       risks: List[Dict], warnings: List[Dict]) -> List[str]:
        """Generate safety-specific recommendations"""
        recommendations = []
        
        if risks:
            recommendations.append("🧪 Conduct patch testing before full application")
            recommendations.append("📋 Provide clear usage instructions and warnings")
        
        if warnings:
            recommendations.append("📦 Implement proper storage and packaging requirements")
            recommendations.append("⏰ Establish clear expiration and stability guidelines")
        
        if formulation.safety_score < 8.0:
            recommendations.append("🔬 Consider reformulation with safer alternatives")
            recommendations.append("👥 Consult with dermatological experts for validation")
        
        return recommendations


def main():
    """Demo of hypergredient framework capabilities"""
    print("🧬 Hypergredient Framework Architecture Demo")
    print("=" * 50)
    
    # Initialize system
    database = HypergredientDatabase()
    optimizer = HypergredientOptimizer(database)
    analyzer = HypergredientAnalyzer(database)
    visualizer = HypergredientVisualizer(database)
    
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
    
    # Demo 4: AI-driven predictions
    print("\n4. AI-Driven Ingredient Predictions")
    print("-" * 40)
    
    ai_system = HypergredientAI()
    ai_predictions = ai_system.predict_optimal_combination(anti_aging_request)
    
    print(f"Model version: {ai_predictions['model_version']}")
    print("Top AI Predictions:")
    for pred in ai_predictions['predictions'][:3]:
        print(f"  • {pred['ingredient_class']}: {pred['confidence']:.1%} confidence")
        print(f"    Reasoning: {pred['reasoning']}")
    
    # Demo 5: Evolutionary formulation improvement
    print("\n5. Evolutionary Formulation Improvement")
    print("-" * 40)
    
    evolution_system = FormulationEvolution(result)
    
    # Simulate market feedback
    evolution_system.add_market_feedback({
        'efficacy_rating': 7.5,
        'safety_rating': 9.0,
        'user_satisfaction': 8.2,
        'improvement_requests': ['more moisturizing', 'faster results']
    })
    
    # Evolve the formulation
    target_improvements = {
        'efficacy': 0.25,  # Target 25% efficacy
        'safety': 0.95     # Target 95% safety
    }
    
    evolved_formula = evolution_system.evolve(database, target_improvements)
    
    print(f"✓ Evolution complete - Generation {evolution_system.generation}")
    print(f"  Evolved efficacy: {evolved_formula.predicted_efficacy:.2%}")
    print(f"  Evolved safety: {evolved_formula.safety_score:.1f}/10")
    print(f"  Evolved cost: R{evolved_formula.total_cost:.2f}")
    
    evolution_report = evolution_system.get_evolution_report()
    
    # Demo 6: Visualization dashboard
    print("\n6. Visualization Dashboard Report")
    print("-" * 40)
    
    visual_report = visualizer.generate_formulation_report(result, anti_aging_request)
    
    print(f"✓ Generated comprehensive visualization report")
    print(f"  Formulation ID: {visual_report['formulation_overview']['formulation_id']}")
    print(f"  Performance strengths: {', '.join(visual_report['performance_radar']['interpretation']['strengths'])}")
    print(f"  Most expensive ingredient: {visual_report['ingredient_breakdown']['summary']['most_expensive']}")
    print(f"  Network quality: {visual_report['synergy_network']['network_description']}")
    print(f"  Risk level: {visual_report['risk_assessment']['overall_risk_level']}")
    print(f"  Recommendations: {len(visual_report['recommendations'])} actionable items")
    
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
        "ingredient_profile": profile,
        "ai_predictions": ai_predictions,
        "evolution_report": evolution_report,
        "visualization_report": visual_report
    }
    
    with open("hypergredient_demo_results.json", "w") as f:
        json.dump(demo_results, f, indent=2, default=str)
    
    print(f"\n✓ Demo results saved to hypergredient_demo_results.json")
    print("\nHypergredient Framework successfully demonstrates:")
    print("• Multi-objective formulation optimization")
    print("• Real-time compatibility analysis") 
    print("• Ingredient profiling and scoring")
    print("• Synergy calculation and recommendations")
    print("• AI-driven ingredient predictions")
    print("• Evolutionary formulation improvement")
    print("• Comprehensive visualization dashboard")


if __name__ == "__main__":
    main()