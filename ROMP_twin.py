import sys
import numpy as np
from typing import List, Optional, Dict, Tuple
from abc import ABC, abstractmethod
from itertools import product

class ReactionChannel(ABC):
    """
    Abstract base class for a reaction channel in the ROMP simulation.
    Each channel must implement validation, calculation, and firing logic.
    """
    def __init__(self, category: str, rate_constant: float) -> None:
        """
        Initialize the reaction channel with its category and rate constant.

        Parameters:
        - category: The type of reaction channel (e.g., "initiation" or "propagation").
        - rate_constant: The rate constant associated with this reaction channel.
        """
        self.category = category
        self.rate_constant = rate_constant

    @abstractmethod
    def validate(self, twin: "ROMPTwin") -> None:
        """
        Validate the reaction channel against the current state of the ROMP twin.

        Parameters:
        - twin: The ROMP twin instance to validate against.
        """
        pass

    @abstractmethod
    def calculate(self, twin: "ROMPTwin") -> float:
        """
        Calculate the reaction rate for this channel based on the current state of the ROMP twin.

        Parameters:
        - twin: The ROMP twin instance to calculate the rate for.

        Returns:
        - The calculated reaction rate as a float.
        """
        pass

    @abstractmethod
    def fire(self, twin: "ROMPTwin") -> None:
        """
        Execute the reaction event for this channel, updating the ROMP twin's state accordingly.

        Parameters:
        - twin: The ROMP twin instance to update.
        """
        pass

class InitiationChannel(ReactionChannel):
    """
    Initiation reaction channel for the ROMP simulation.
    Handles the initiation step where an initiator reacts with a monomer to start a polymer chain.
    """
    def __init__(self, monomer_name: str, rate_constant: float) -> None:
        """
        Initialize the initiation reaction channel with the specified monomer and rate constant.

        Parameters:
        - monomer_name: The name of the monomer involved in the initiation reaction.
        - rate_constant: The rate constant for the initiation reaction.
        """
        super().__init__(category="initiation", rate_constant=rate_constant)

        self.monomer_name = monomer_name
        self.monomer_idx = -1
        self.initiator_idx = -1

    def validate(self, twin: "ROMPTwin") -> None:
        """
        Validate the initiation reaction channel against the current state of the ROMP twin.

        Parameters:
        - twin: The ROMP twin instance to validate against.
        """
        twin.validate_channel([self.monomer_name])
        self.monomer_idx = twin.token_to_idx[self.monomer_name]
        self.initiator_idx = twin.token_to_idx["initiator"]

    def calculate(self, twin: "ROMPTwin") -> float:
        """
        Calculate the reaction rate for the initiation channel based on the current state of the ROMP twin.

        Parameters:
        - twin: The ROMP twin instance to calculate the rate for.

        Returns:
        - The calculated reaction rate as a float.
        """
        ru_count = twin.pool_counts[self.initiator_idx]
        monomer_count = twin.pool_counts[self.monomer_idx]

        if ru_count <= 0 or monomer_count <= 0:
            return 0.0

        c_micro = self.rate_constant / (twin.total_volume * twin.Na) 

        return float(c_micro * ru_count * monomer_count)

    def fire(self, twin: "ROMPTwin") -> None:
        """
        Fire the initiation reaction channel, updating the state of the ROMP twin accordingly.

        Parameters:
        - twin: The ROMP twin instance to update.
        """
        twin._initiate_move(self.monomer_idx)

class PropagationChannel(ReactionChannel):
    """
    Propagation reaction channel for the ROMP twin.
    This channel represents the propagation step where a polymer chain end reacts with a monomer.
    """
    def __init__(self, chain_end_name: str, monomer_name: str, rate_constant: float) -> None:
        """
        Initialize a propagation reaction channel.

        Parameters:
        - chain_end_name: The name of the polymer chain end token.
        - monomer_name: The name of the monomer token.
        - rate_constant: The rate constant for the propagation reaction.
        """
        super().__init__(category="propagation", rate_constant=rate_constant)
    
        self.chain_end_name = f"Ru_{chain_end_name}"
        self.monomer_name = monomer_name
        self.chain_end_idx = -1
        self.monomer_idx = -1
        
    def validate(self, twin: "ROMPTwin") -> None:
        """
        Validate the propagation reaction channel against the current state of the ROMP twin.

        Parameters:
        - twin: The ROMP twin instance to validate against.
        """
        twin.validate_channel([self.chain_end_name, self.monomer_name])
        self.chain_end_idx = twin.token_to_idx[self.chain_end_name]
        self.monomer_idx = twin.token_to_idx[self.monomer_name]

    def calculate(self, twin: "ROMPTwin") -> float:
        """
        Calculate the reaction rate for the propagation channel based on the current state of the ROMP twin.

        Parameters:
        - twin: The ROMP twin instance to calculate the rate for.

        Returns:
        - The calculated reaction rate as a float.
        """
        chain_end_count = twin.pool_counts[self.chain_end_idx]
        monomer_count = twin.pool_counts[self.monomer_idx]

        if chain_end_count <= 0 or monomer_count <= 0:
            return 0.0

        c_micro = self.rate_constant / (twin.total_volume * twin.Na) 

        return float(c_micro * chain_end_count * monomer_count)

    def fire(self, twin: "ROMPTwin") -> None:
        """
        Fire the propagation reaction channel, updating the state of the ROMP twin accordingly.

        Parameters:
        - twin: The ROMP twin instance to update.
        """
        twin._propagate_move(self.chain_end_idx, self.monomer_idx)

class TerminationChannel(ReactionChannel):
    """
    Termination reaction channel class for the ROMP system.
    """
    def __init__(self, chain_end_name: str, rate_constant: float):
        """
        Initialize the termination reaction channel with the two chain ends and rate constant.

        Parameters:
        - chain_end_name: The name of the chain end for termination.
        - rate_constant: The rate constant for the termination reaction.
        """
        super().__init__(category="termination", rate_constant=rate_constant)
        self.chain_end_name = f"Ru_{chain_end_name}"
        self.chain_end_idx = -1

    def validate(self, twin: "ROMPTwin") -> None:
        """
        Validate the termination reaction channel against the ROMP twin.

        Parameters:
        - twin: The ROMP twin instance to validate against.
        """
        twin.validate_channel([self.chain_end_name])
        self.chain_end_idx = twin.token_to_idx[self.chain_end_name]

    def calculate(self, twin: "ROMPTwin") -> float:
        """
        Calculate the reaction rate for the termination reaction channel.

        Parameters:
        - twin: The ROMP twin instance to calculate the rate for.

        Returns:
        - The calculated reaction rate as a float.
        """
        chain_end_count = twin.pool_counts[self.chain_end_idx]
        if chain_end_count <= 0:
            return 0.0
    
        return float(self.rate_constant * chain_end_count)

    def fire(self, twin: "ROMPTwin") -> None:
        """
        Fire the termination reaction channel, updating the ROMP twin state.

        Parameters:
        - twin: The ROMP twin instance to update.
        """
        twin._terminate_move(self.chain_end_idx)

class ROMPTwin:
    """
    ROMP twin class representing the state and dynamics of the ROMP system.
    """
    def __init__(self, 
                 monomer_names: List[str], 
                 monomer_counts: Dict[str, int], 
                 monomer_mws: Dict[str, float], 
                 monomer_molar_volumes: Dict[str, float], 
                 polymer_molar_volumes: Dict[str, float], 
                 initiator_count: int, 
                 solvent_volume: float,
                 channels: List["ReactionChannel"]) -> None:
        """
        Initialize the ROMP twin with the given system parameters and reaction channels.

        Parameters:
        - monomer_names: List of monomer names.
        - monomer_counts: Dictionary mapping monomer names to their initial counts.
        - monomer_mws: Dictionary mapping monomer names to their molecular weights.
        - monomer_molar_volumes: Dictionary mapping monomer names to their molar volumes.
        - polymer_molar_volumes: Dictionary mapping polymer names to their molar volumes.
        - initiator_count: Count of initiator molecules.
        - solvent_volume: Volume of the solvent.
        - channels: List of reaction channels.
        """
        self.monomer_names = monomer_names
        self.monomer_counts = monomer_counts
        self.monomer_mws = monomer_mws
        self.monomer_molar_volumes = monomer_molar_volumes
        self.polymer_molar_volumes = polymer_molar_volumes
        self.initiator_count = initiator_count
        self.channels = channels
        self.num_channels = len(channels)
        self.fire_counts: np.ndarray = np.zeros_like(self.channels, dtype=np.int32)

        self.solvent_volume = solvent_volume
        self.total_volume = 0.0
        self.R = 8.314  # J/(mol*K)
        self.temperature = 298.15  # K
        self.Na = 6.022e23  # Avogadro's number
        self.time = 0.0
        self.iteration_count = 0

        self.token_to_idx: Dict[str, int] = {}
        self.idx_to_token: Dict[int, str] = {}
        self.next_row = 0
        self.next_chain_id = 0

        num_pools = 1 + 2 * len(monomer_names)
        self.max_chains = initiator_count

        self.chain_matrix = np.full((num_pools, self.max_chains), -1, dtype=np.int32)
        self.pool_counts = np.zeros(num_pools, dtype=np.int32)
        self.initial_pool_counts = np.zeros(num_pools, dtype=np.int32)
        self.injected_monomer_counts = np.zeros(num_pools, dtype=np.int32)

        self.chain_backbones: List[Optional[np.ndarray]] = [None] * self.max_chains
        self.chain_lengths = np.zeros(self.max_chains, dtype=np.int32)
        self.chain_to_matrix_row = np.full(self.max_chains, -1, dtype=np.int32)
        self.chain_to_matrix_col = np.full(self.max_chains, -1, dtype=np.int32)

        self.monomer_molar_volumes = {}
        self.polymer_molar_volumes = {}

        self.tracked_chain_rows = set()

        initiator_row = self._register_pool_row("initiator")
        self.pool_counts[initiator_row] = initiator_count
        self.initial_pool_counts[initiator_row] = initiator_count
        self.injected_monomer_counts[initiator_row] = 0

        for i, name in enumerate(monomer_names):
            row = self._register_pool_row(name)
            self.pool_counts[row] = monomer_counts[name]
            self.initial_pool_counts[row] = monomer_counts[name]
            self.injected_monomer_counts[row] = 0
            self.monomer_molar_volumes[name] = monomer_molar_volumes[name]
            self.polymer_molar_volumes[name] = polymer_molar_volumes[name]

        for name in monomer_names:
            row = self._register_pool_row(f"Ru_{name}")
            self.pool_counts[row] = 0

        self.monomer_pool_idxs = np.array([self.token_to_idx[name] for name in monomer_names])

        for i in range(self.num_channels):
            self.channels[i].validate(self)

        self._update_volume()
    
    def _register_pool_row(self, token: str) -> int:
        """
        Register a new pool row for the given token if it doesn't already exist.

        Parameters:
        - token: The name of the token to register.

        Returns:
        - The index of the row corresponding to the token.
        """
        if token in self.token_to_idx:
            return self.token_to_idx[token]
        else:
            row = self.next_row
            self.token_to_idx[token] = row
            self.idx_to_token[row] = token
            self.next_row += 1
            return row
        
    def validate_channel(self, tokens: List[str]) -> None:
        """
        Validate that all tokens in the given list are registered in the system.

        Parameters:
        - tokens: List of token names to validate.

        Raises:
        - ValueError: If any token is not recognized.
        """
        for token in tokens:
            if token not in self.token_to_idx:
                raise ValueError(f"Token '{token}' not recognized.")
            
    def _calculate_conversion(self) -> float:
        """
        Calculate the conversion of monomers in the system.

        Returns:
        - The fraction of monomers that have been converted into polymers.
        """
        current_free_monomers = 0
        total_system_throughput = 0

        for monomer_idx in self.monomer_pool_idxs:
            current_free_monomers += self.pool_counts[monomer_idx]
            total_system_throughput += self.initial_pool_counts[monomer_idx] + self.injected_monomer_counts[monomer_idx]

        if total_system_throughput == 0:
            return 0.0
        return (total_system_throughput - current_free_monomers) / total_system_throughput

    def _update_volume(self) -> None:
        """
        Update the total volume of the system based on the current free monomer and bound polymer volumes.

        This method calculates the contributions from free monomers and bound polymer units,
        and updates the total volume of the system, including the static solvent volume.
        """
        free_monomer_volume = 0.0
        for monomer in self.monomer_names:
            monomer_idx = self.token_to_idx[monomer]

            moles_free = self.pool_counts[monomer_idx] / self.Na
            free_monomer_volume += moles_free * self.monomer_molar_volumes[monomer]

        bound_polymer_volume = 0.0
        for monomer in self.monomer_names:
            monomer_idx = self.token_to_idx[monomer]
            
            total_throughput = self.initial_pool_counts[monomer_idx] + self.injected_monomer_counts[monomer_idx]
            current_count = self.pool_counts[monomer_idx]
            
            moles_consumed = (total_throughput - current_count) / self.Na
            bound_polymer_volume += moles_consumed * self.polymer_molar_volumes[monomer]

        self.total_volume = self.solvent_volume + ((free_monomer_volume + bound_polymer_volume) / 1000)

    def inject_by_tokens(self, injections: Dict[str, Dict[str, float]]) -> None:
        """
        Inject monomers into the system by specifying their concentrations and volumes.

        Parameters:
        - injections: A dictionary where keys are token names and values are dictionaries
          with "concentration" and "volume" keys.

        Raises:
        - ValueError: If any token is not recognized.
        """
        for token, injection in injections.items():
            if token not in self.token_to_idx:
                raise ValueError(f"Token '{token}' not recognized.")
            
            row = self.token_to_idx[token]
            concentration = injection.get("concentration", 0.0)
            volume = injection.get("volume", 0.0)

            self.pool_counts[row] += concentration * volume * self.Na
            self.injected_monomer_counts[row] += concentration * volume * self.Na
            self.solvent_volume += volume

        self._update_volume()
    
    def _initiate_move(self, monomer_pool: int) -> None:
        """
        Initiate a new polymer chain in the system using a monomer from the specified pool.

        Parameters:
        - monomer_pool: The index of the monomer pool from which to draw the initiating monomer.

        Raises:
        - RuntimeError: If the maximum number of chains is exceeded or if there are no available slots in the target pool matrix.
        """
        new_chain_id = self.next_chain_id
        self.next_chain_id += 1

        if new_chain_id >= self.max_chains:
            raise RuntimeError("Exceeded maximum number of chains.")

        monomer_token = self.idx_to_token[monomer_pool]
        target_pool = self.token_to_idx[f"Ru_{monomer_token}"]

        target_col = self.pool_counts[target_pool]
        if target_col >= self.chain_matrix.shape[1]:
            raise RuntimeError("Exceeded maximum number of chains in the target pool matrix column.")

        self.chain_backbones[new_chain_id] = np.full(32, -1, dtype=np.int32)
        self.chain_backbones[new_chain_id][0] = monomer_pool
        self.chain_lengths[new_chain_id] = 1

        self.chain_matrix[target_pool, target_col] = new_chain_id
        self.chain_to_matrix_row[new_chain_id] = target_pool
        self.chain_to_matrix_col[new_chain_id] = target_col
        
        self.pool_counts[target_pool] += 1
        self.pool_counts[self.token_to_idx["initiator"]] -= 1
        self.pool_counts[monomer_pool] -= 1 

    def _propagate_move(self, target_pool: int, monomer_pool: int) -> None:
        """
        Propagate an existing polymer chain by adding a monomer from the specified monomer pool.

        Parameters:
        - target_pool: The index of the polymer pool containing the chain to propagate.
        - monomer_pool: The index of the monomer pool from which to draw the monomer for propagation.

        Raises:
        - RuntimeError: If there are no chains available in the target pool or no monomers available in the monomer pool.
        """
        target_count = self.pool_counts[target_pool]
        if target_count == 0:
            raise RuntimeError("No chains available in the target pool to propagate.")
        if self.pool_counts[monomer_pool] <= 0:
            raise RuntimeError("No monomers available in the monomer pool to propagate.")

        rand_col = np.random.randint(0, target_count)
        rand_chain_id = self.chain_matrix[target_pool, rand_col]
        rand_chain_length = self.chain_lengths[rand_chain_id]
        
        if rand_chain_length >= self.chain_backbones[rand_chain_id].shape[0]:
            old_backbone = self.chain_backbones[rand_chain_id]
            new_length = max(len(old_backbone) * 2, 1)
            new_backbone = np.full(new_length, -1, dtype=np.int32)
            new_backbone[:rand_chain_length] = old_backbone
            self.chain_backbones[rand_chain_id] = new_backbone
        
        self.chain_backbones[rand_chain_id][rand_chain_length] = monomer_pool
        self.chain_lengths[rand_chain_id] += 1
        self.pool_counts[monomer_pool] -= 1

        monomer_token = self.idx_to_token[monomer_pool]
        new_pool = self.token_to_idx[f"Ru_{monomer_token}"]

        if target_pool == new_pool:
            return

        last_active_col = target_count - 1
        if rand_col != last_active_col:
            last_chain_id = self.chain_matrix[target_pool, last_active_col]
            
            self.chain_matrix[target_pool, rand_col] = last_chain_id
            self.chain_to_matrix_col[last_chain_id] = rand_col
            
        self.chain_matrix[target_pool, last_active_col] = -1
        self.pool_counts[target_pool] -= 1

        new_col = self.pool_counts[new_pool]
        if new_col >= self.chain_matrix.shape[1]:
            raise RuntimeError("Exceeded maximum number of chains in the destination pool matrix column.")
            
        self.chain_matrix[new_pool, new_col] = rand_chain_id
        self.chain_to_matrix_row[rand_chain_id] = new_pool
        self.chain_to_matrix_col[rand_chain_id] = new_col
        self.pool_counts[new_pool] += 1
    
    def _terminate_move(self, target_pool: int) -> None:
        """
        Perform a monomolecular termination move (simple chain deactivation).
        Selects and permanently deactivates a single active chain from the specified pool.

        Parameters:
        - target_pool: The index of the pool that triggered the termination event.
        """
        count = self.pool_counts[target_pool]
        if count == 0:
            raise RuntimeError(f"No chains available in pool {target_pool} to terminate.")

        rand_col = np.random.randint(0, count)
        chain_id = self.chain_matrix[target_pool, rand_col]

        last_col = count - 1
        if rand_col != last_col:
            last_chain_id = self.chain_matrix[target_pool, last_col]
            self.chain_matrix[target_pool, rand_col] = last_chain_id
            self.chain_to_matrix_col[last_chain_id] = rand_col

        self.chain_matrix[target_pool, last_col] = -1
        self.chain_to_matrix_row[chain_id] = -1
        self.chain_to_matrix_col[chain_id] = -1

        self.pool_counts[target_pool] -= 1

    def _calculate_propensities(self) -> np.ndarray:
        """
        Calculate the propensities for all reaction channels based on the current system state.

        Returns:
        - A numpy array of propensities for each channel, with non-negative values.
        """
        prop_vec = np.zeros(self.num_channels, dtype=np.float64)
        for i in range(self.num_channels):
            propensity = self.channels[i].calculate(self)

            if np.isnan(propensity):
                raise RuntimeError(f"Propensity for channel {i} is NaN.")
            
            if propensity > 0:
                prop_vec[i] = propensity
            else:
                prop_vec[i] = 0.0
        
        return prop_vec
    
    def _execute_move(self) -> None:
        """
        Execute a single reaction move based on the current propensities.

        Raises:
        - RuntimeError: If no valid moves are available or if the total propensity is non-positive.
        """
        prop_vec = self._calculate_propensities()
        if prop_vec is None or np.all(prop_vec == 0):
            raise RuntimeError("No valid moves available to execute.")

        total_propensity = np.sum(prop_vec)
        if total_propensity <= 0.0:
            raise RuntimeError("Total propensity is non-positive, cannot execute a move.")

        r1 = 1.0 - np.random.rand()
        dt = -np.log(r1) / total_propensity

        r2 = np.random.rand() * total_propensity
        cumulative_propensity = np.cumsum(prop_vec)
        
        selected_channel_index = np.searchsorted(cumulative_propensity, r2, side='right')
        selected_channel_index = min(selected_channel_index, self.num_channels - 1)
        
        if prop_vec[selected_channel_index] == 0.0:
            non_zero_indices = np.where(prop_vec > 0)[0]
            selected_channel_index = non_zero_indices[np.searchsorted(cumulative_propensity[non_zero_indices], r2)]

        winning_channel = self.channels[selected_channel_index]
        winning_channel.fire(self)
        
        self.fire_counts[selected_channel_index] += 1
        self.time += dt

    def run_to_time(self, end_time: float) -> float:
        """
        Run the simulation until the specified end time is reached.

        Parameters:
        - end_time (float): The simulation time to run until.

        Returns:
        - float: The total propensity at the end of the simulation run.

        Raises:
        - RuntimeError: If an error occurs during the execution of a reaction move.
        """
        while self.time < end_time:
            try:
                prop_vec = self._calculate_propensities()
                total_propensity = np.sum(prop_vec)
                
                if total_propensity <= 1e-12:
                    return total_propensity
                
                self._execute_move()
                self._update_volume()
                self.iteration_count += 1

                current_conversion = self._calculate_conversion()
                if current_conversion >= 1.0:
                    return total_propensity

            except RuntimeError as e:
                raise RuntimeError(f"[Simulation Terminated with Error at t={self.time:.4f}s]: {e}")

        return 1e6

    def get_report(self) -> dict:
        """
        Generate a report summarizing the current state of the simulation.

        Returns:
        - dict: A dictionary containing firing telemetry, conversion, current pool counts,
                monomer conversions, and polymer compositions.
        """
        firing_telemetry = {}
        for i, channel in enumerate(self.channels):
            category = channel.category
            if category not in firing_telemetry:
                firing_telemetry[category] = 0
            firing_telemetry[category] += int(self.fire_counts[i])

        conversion = self._calculate_conversion()

        current_counts = {
            token: int(self.pool_counts[idx]) for token, idx in self.token_to_idx.items()
        }

        monomer_conversions = {}
        polymer_compositions = []
        total_polymer_counts = 0
        for monomer in self.monomer_names:
            monomer_idx = self.token_to_idx[monomer]
            monomer_count = self.pool_counts[monomer_idx]
            initial_count = self.initial_pool_counts[monomer_idx]
            consumed_count = initial_count - monomer_count
            monomer_conversion = 1.0 - (monomer_count / initial_count) if initial_count > 0 else 0.0
            monomer_conversions[monomer] = float(monomer_conversion)
            polymer_compositions.append(float(consumed_count))
            total_polymer_counts += consumed_count

        polymer_compositions = [count / total_polymer_counts if total_polymer_counts > 0 else 0.0 for count in polymer_compositions]
        composition_by_monomer = {monomer: polymer_compositions[i] for i, monomer in enumerate(self.monomer_names)}

        pool_to_molecular_weight = {self.token_to_idx[monomer]: self.monomer_mws[monomer] for monomer in self.monomer_names}
        chain_masses = []
        for i in range(self.next_chain_id):
            backbone = self.chain_backbones[i]
            length = self.chain_lengths[i]

            if backbone is None or length == 0:
                continue

            active_monomers = backbone[:length]

            chain_mass = 0.0
            for pool_idx, molecular_weight in pool_to_molecular_weight.items():
                monomer_count = np.sum(active_monomers == pool_idx)
                chain_mass += monomer_count * molecular_weight
            chain_masses.append(chain_mass)

        chain_masses = np.array(chain_masses)
        if len(chain_masses) == 0:
            Mn, Mw, PDI = 0.0, 0.0, 1.0
        else:
            Mn = float(np.mean(chain_masses))
            Mw = float(np.sum(chain_masses ** 2) / np.sum(chain_masses))
            PDI = Mw / Mn if Mn > 0 else 1.0

        return {
            "time": float(self.time),
            "volume": float(self.total_volume),
            "iteration_count": int(self.iteration_count),
            "firing_telemetry": firing_telemetry,
            "conversion": conversion,
            "monomer_conversions": monomer_conversions,
            "composition_by_monomer": composition_by_monomer,
            "current_counts": current_counts,
            "Mn": Mn,
            "Mw": Mw,
            "PDI": PDI,
        }

    def get_counts_dict(self) -> dict:
        """
        Returns a dictionary of raw token counts for the initiator and monomers, including active chain ends.

        Returns: 
            dict: A dictionary containing raw token counts for the initiator and monomers, including active chain ends.
        """
        counts_dict = {"Ru_precatalyst": self.pool_counts[self.token_to_idx["initiator"]]}
        
        for monomer in self.monomer_names:
            label = f"M_{monomer}"
            counts_dict[label] = self.pool_counts[self.token_to_idx[monomer]]

            tip_label = f"Ru_{monomer}"
            counts_dict[tip_label] = self.pool_counts[self.token_to_idx[tip_label]]
            
        return counts_dict

    def get_xad_fractions(self, xad_length: int = 3) -> dict:
        """
        Calculates the fractions of sequence-defined monomer blocks (e.g., triads) for all active chains.

        Args:
            xad_length (int): The length of the sequence block to analyze (default is 3 for triads).

        Returns:
            dict: A dictionary containing the keys "triad_keys" (list of all possible sequence blocks),
                  "triads" (mean fractions across all chains), and "triads_std" (standard deviation across chains).
        """     
        pool_idx_to_name = {}
        for name in self.monomer_names:
            m_pool_idx = self.token_to_idx[name]
            pool_idx_to_name[m_pool_idx] = name

        all_possible_keys = ["_".join(p) for p in product(self.monomer_names, repeat=xad_length)]
        
        chain_triad_counts = []

        for i in range(self.next_chain_id):
            backbone = self.chain_backbones[i]
            length = self.chain_lengths[i]

            if backbone is None or length < xad_length:
                continue

            active_monomers = backbone[:length]
            local_counts = {k: 0 for k in all_possible_keys}
            local_total = 0
            
            for j in range(length - xad_length + 1):
                segment = active_monomers[j:j + xad_length]
                
                try:
                    names = [pool_idx_to_name[idx] for idx in segment]
                    dict_key = "_".join(names)
                    if dict_key in local_counts:
                        local_counts[dict_key] += 1
                        local_total += 1
                except KeyError:
                    continue

            if local_total > 0:
                chain_fractions = [local_counts[k] / local_total for k in all_possible_keys]
                chain_triad_counts.append(chain_fractions)

        if not chain_triad_counts:
            return {"triad_keys": [], "triads": [], "triads_std": []}

        chain_matrix = np.array(chain_triad_counts)

        global_means = np.mean(chain_matrix, axis=0)
        global_stds = np.std(chain_matrix, axis=0)

        active_indices = np.where(global_means > 0.0)[0]
        
        output_keys = [all_possible_keys[idx] for idx in active_indices]
        output_means = [float(global_means[idx]) for idx in active_indices]
        output_stds = [float(global_stds[idx]) for idx in active_indices]

        return {
            "triad_keys": output_keys,
            "triads": output_means,
            "triads_std": output_stds
        }

    def get_molecular_weight_distribution(
        self, 
        bins: int = 50
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
        """
        Calculate the molecular weight distribution (weight fraction vs Mw)
        across all tracked polymer chains.

        Parameters:
        - bins: Number of bins or an array of bin edges for the distribution histogram.

        Returns:
        - mw_bins: The center molecular weight value of each bin (X-axis).
        - weight_fractions: The weight fraction of polymer in each bin (Y-axis).
        - metrics: A dictionary containing Mn, Mw, and PDI (dispersity).
        """
        idx_to_mw = np.zeros(len(self.idx_to_token), dtype=np.float64)
        for i, name in enumerate(self.monomer_names):
            monomer_idx = self.token_to_idx[name]
            idx_to_mw[monomer_idx] = self.monomer_mws[name]

        chain_mws = []

        for i in range(self.next_chain_id):
            backbone = self.chain_backbones[i]
            length = self.chain_lengths[i]

            if backbone is None or length == 0:
                continue

            active_monomers = backbone[:length]
            chain_mw = np.sum(idx_to_mw[active_monomers])
            chain_mws.append(chain_mw)

        chain_mws = np.array(chain_mws, dtype=np.float64)

        if len(chain_mws) == 0:
            return np.zeros(bins), np.zeros(bins), {"Mn": 0.0, "Mw": 0.0, "PDI": 1.0}

        total_chains = len(chain_mws)
        total_mass = np.sum(chain_mws)
        
        mn = total_mass / total_chains
        mw = np.sum(chain_mws**2) / total_mass
        pdi = mw / mn

        mass_per_bin, bin_edges = np.histogram(chain_mws, bins=bins, weights=chain_mws)
        weight_fractions = mass_per_bin / total_mass
        
        mw_bins = (bin_edges[:-1] + bin_edges[1:]) / 2.0

        metrics = {
            "Mn": float(mn),
            "Mw": float(mw),
            "PDI": float(pdi)
        }

        return mw_bins, weight_fractions, metrics

    def get_monomer_position_frequency(self) -> Dict[str, np.ndarray]:
        """
        Sweep across all chain backbones and calculate the frequency of each monomer 
        at each individual repeat unit position (Xn).

        Returns:
        - A dictionary where keys are monomer names (e.g., "M1") and values are 
        NumPy arrays of shape (max_chain_length,). The value at index `i` represents 
        the fraction of chains that contain that monomer at position Xn = i + 1.
        """
        active_chains = []
        max_len = 0
        
        for i in range(self.next_chain_id):
            backbone = self.chain_backbones[i]
            length = self.chain_lengths[i]
            
            if backbone is not None and length > 0:
                active_chains.append(backbone[:length])
                if length > max_len:
                    max_len = length

        if max_len == 0 or len(active_chains) == 0:
            return {name: np.array([], dtype=np.float64) for name in self.monomer_names}

        token_to_monomer_idx = np.full(len(self.idx_to_token), -1, dtype=np.int32)
        for i, name in enumerate(self.monomer_names):
            monomer_token_idx = self.token_to_idx[name]
            token_to_monomer_idx[monomer_token_idx] = i

        num_monomers = len(self.monomer_names)
        counts = np.zeros((num_monomers, max_len), dtype=np.float64)
        valid_chains_at_position = np.zeros(max_len, dtype=np.float64)

        for chain in active_chains:
            chain_len = len(chain)
            monomer_indices = token_to_monomer_idx[chain]

            for pos in range(chain_len):
                m_idx = monomer_indices[pos]
                if m_idx != -1:
                    counts[m_idx, pos] += 1
            
            valid_chains_at_position[:chain_len] += 1

        frequencies = np.zeros_like(counts)
        mask = valid_chains_at_position > 0
        frequencies[:, mask] = counts[:, mask] / valid_chains_at_position[mask]

        return {
            self.monomer_names[i]: frequencies[i] 
            for i in range(num_monomers)
        }

    def get_backbone_matrix(
        self, 
        pad_value: float = -1.0
    ) -> Tuple[np.ndarray, list]:
        """
        Convert the list of variable-length chain backbones into a uniform 
        2D matrix suitable for visualization with matplotlib's imshow.

        Parameters:
        - pad_value: The value used to fill empty positions where a chain 
                    has not yet grown or has terminated (e.g., -1.0 or np.nan).

        Returns:
        - matrix: A 2D NumPy array of shape (num_tracked_chains, max_chain_length)
                where values correspond to the monomer index in self.monomer_names.
        - chain_ids: A list of the absolute chain IDs corresponding to each row.
        """
        valid_chain_indices = []
        max_len = 0

        for i in range(self.next_chain_id):
            backbone = self.chain_backbones[i]
            length = self.chain_lengths[i]
            if backbone is not None:
                valid_chain_indices.append(i)
                if length > max_len:
                    max_len = length

        num_chains = len(valid_chain_indices)

        if num_chains == 0 or max_len == 0:
            return np.empty((0, 0)), []

        token_to_monomer_idx = np.full(len(self.idx_to_token), -1, dtype=np.int32)
        for i, name in enumerate(self.monomer_names):
            monomer_token_idx = self.token_to_idx[name]
            token_to_monomer_idx[monomer_token_idx] = i

        matrix = np.full((num_chains, max_len), pad_value, dtype=np.float64)

        for row_idx, chain_id in enumerate(valid_chain_indices):
            length = self.chain_lengths[chain_id]
            if length > 0:
                mapped_backbone = token_to_monomer_idx[self.chain_backbones[chain_id][:length]]
                matrix[row_idx, :length] = mapped_backbone

        return matrix, valid_chain_indices

    def get_memory_usage_mb(self) -> float:
        """
        Calculates the exact tracking footprint of the simulation structures 
        analytically. This adds zero execution or wall-time overhead.
        """
        total_bytes = 0
        
        arrays = [
            self.chain_matrix, 
            self.chain_lengths, 
            self.chain_to_matrix_row, 
            self.chain_to_matrix_col
        ]
        for arr in arrays:
            if arr is not None:
                total_bytes += arr.nbytes
                
        for backbone in self.chain_backbones:
            if backbone is not None:
                total_bytes += backbone.nbytes
                
        if hasattr(self, 'channels'):
            total_bytes += sys.getsizeof(self.channels)
            for channel in self.channels:
                total_bytes += sys.getsizeof(channel)
                
        return total_bytes / (1024 ** 2)