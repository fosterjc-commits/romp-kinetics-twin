from typing import Dict, Tuple
import numpy as np
import json

def initialize_simulation_volume(
    total_target_molecules: int, 
    target_molarities: Dict[str, float],
    Na: float = 6.022e23
) -> Tuple[float, Dict[str, int]]:
    """
    Calculates the exact microscopic SSA simulation box volume (in Liters) 
    and splits the total molecule pool into individual discrete monomer counts 
    to perfectly match real-world target molarity ratios.
    
    Parameters:
        total_target_molecules (int): Total number of molecules allocated for the SSA twin (e.g., 10000)
        target_molarities (Dict[str, float]): Target concentrations in mol/L (M) for each monomer type.
        Na (float): Centralized Avogadro constant instance.
        
    Returns:
        Tuple[float, Dict[str, int]]: 
            - solvent_volume_L: The exact simulation tracking volume in Liters.
            - initial_counts: Map of monomer names to their integer particle allocations.
    """
    total_molarity = sum(target_molarities.values())
    if total_molarity <= 0:
        raise ValueError("Total target molarity must be greater than zero.")
        
    total_moles = total_target_molecules / Na
    
    solvent_volume_L = total_moles / total_molarity
    
    initial_counts = {}
    remaining_molecules = total_target_molecules
    monomer_items = list(target_molarities.items())
    
    for m_name, molarity in monomer_items[:-1]:
        mole_fraction = molarity / total_molarity
        count = int(np.round(total_target_molecules * mole_fraction))
        initial_counts[m_name] = count
        remaining_molecules -= count
        
    final_m_name, _ = monomer_items[-1]
    initial_counts[final_m_name] = remaining_molecules
    
    return float(solvent_volume_L), initial_counts

def round_dict(d, decimals=3):
    if isinstance(d, dict):
        return {k: round_dict(v, decimals) for k, v in d.items()}
    elif isinstance(d, list):
        return [round_dict(v, decimals) for v in d]
    elif isinstance(d, (float, np.floating)):
        return round(float(d), decimals)
    elif isinstance(d, (int, np.integer)):
        return int(d)
    return d
