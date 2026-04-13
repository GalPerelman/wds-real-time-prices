import os
import numpy as np
import pandas as pd
from epyt import epanet

import graphs


class EPANETSimulator:
    MAX_VOL_MARGIN_M = 5.0  # fixed upper margin (m); large enough to never limit max violation observation

    def __init__(self, network, inp_file, prices_col, demand_cols, min_vol_margin_m=1.0):
        self.tanks = network.tanks
        self.fsp = network.fsp
        self.prices_col = prices_col
        self.demand_cols = demand_cols
        self.demand_mapping = pd.read_csv(os.path.join(network.data_dir, 'demand_mapping.csv'))
        self.d = epanet(inp_file)

        # Pumps combination membership matrix: rows = pumps combs, columns = EPANET pump IDs (from pump_* cols)
        pump_cols = self.fsp.columns[self.fsp.columns.str.startswith('pump_')]
        self.pump_membership = self.fsp[pump_cols].copy()
        self.pump_membership.columns = self.pump_membership.columns.str.removeprefix('pump_')

        # EPANET pump ID to link index (covers all physical pumps across all combinations)
        self.epanet_pump_idx = pd.Series({pid: self.d.getLinkIndex(str(pid)) for pid in self.pump_membership.columns})

        # Tank name to EPANET node index
        self.tank_epanet_idx = pd.Series(
            [self.d.getNodeIndex(str(eid)) for eid in self.tanks['epanet_id']],
            index=self.tanks.index
        )

        # All junction indices (for zeroing demands at initialize)
        self.junc_idx = self.d.getNodeJunctionIndex()

        # demand_mapping row index to EPANET junction index
        self.mapped_junc_idx = pd.Series(
            [self.d.getNodeIndex(str(j_id)) for j_id in self.demand_mapping['junction']],
            index=self.demand_mapping.index
        )

        self.min_vol_margin_m = min_vol_margin_m
        self._is_open = False
        self._elapsed_h = 0.0
        self._recorded_schedule = pd.DataFrame(columns=self.fsp.index)

    def initialize(self, realizations):
        """
        Prepare the network - assign demands and tariffs from known reference dataset (realizations).

        Parameters
        ----------
        realizations : DataFrame
            Full realization horizon indexed by datetime.
            Must contain self.prices_col and self.demand_cols columns.
        """
        # Remove all controls so the optimizer owns the pump schedule
        self.d.deleteControls()

        # Zero all junction base demands - no silent demands
        self.d.setNodeBaseDemands(self.junc_idx, np.zeros(len(self.junc_idx)).tolist())

        # Create tariff pattern and assign to all pumps (ECost=1 so pattern values = actual price)
        self.d.addPattern('sim_tariff', realizations[self.prices_col].tolist())
        tariff_pat_idx = self.d.getPatternIndex('sim_tariff')
        for epanet_idx in self.epanet_pump_idx:
            self.d.setLinkPumpEPat(epanet_idx, tariff_pat_idx)
            self.d.setLinkPumpECost(epanet_idx, 1.0)

        # Create demand patterns from realizations and assign to mapped junctions
        for i, row in self.demand_mapping.iterrows():
            pattern_values = (realizations[row['zone']] * row['weight']).tolist()
            pattern_name = f"sim_{row['zone']}_j{row['junction']}"
            self.d.addPattern(pattern_name, pattern_values)
            pat_idx = self.d.getPatternIndex(pattern_name)
            self.d.setNodeDemandPatternIndex(self.mapped_junc_idx[i], 1, pat_idx)
            self.d.setNodeBaseDemands([self.mapped_junc_idx[i]], [1.0])

        # Set tank levels (initial, min, max) from LP volumes in self.tanks.
        # base_levels: EPANET minimum water level elevation (m) per tank - datum for level conversion.
        # base_vols: EPANET volume at minimum water level (m^3) per tank - datum for volume conversion.
        # Conversion: level = (vol - base_vol) / area + base_level
        epanet_tank_indices = self.tank_epanet_idx.tolist()
        base_levels = self.d.getNodeTankMinimumWaterLevel(epanet_tank_indices)
        base_vols = self.d.getNodeTankMinimumWaterVolume(epanet_tank_indices)
        diameters = self.d.getNodeTankDiameter(epanet_tank_indices)
        for i, tank_id in enumerate(self.tanks.index):
            area = np.pi * (diameters[i] / 2) ** 2
            idx = self.tank_epanet_idx[tank_id]
            real_min_level = (self.tanks.loc[tank_id, 'min_vol'] - base_vols[i]) / area + base_levels[i]
            real_max_level = (self.tanks.loc[tank_id, 'max_vol'] - base_vols[i]) / area + base_levels[i]
            init_vol = (self.tanks.loc[tank_id, 'init_vol'] - base_vols[i]) / area + base_levels[i]
            self.d.setNodeTankInitialLevel(idx, init_vol)
            self.d.setNodeTankMinimumWaterLevel(idx, max(0.0, real_min_level - self.min_vol_margin_m))
            self.d.setNodeTankMaximumWaterLevel(idx, real_max_level + self.MAX_VOL_MARGIN_M)

        # Open hydraulic analysis
        self.d.openHydraulicAnalysis()
        self.d.initializeHydraulicAnalysis()
        self._elapsed_h = 0.0
        self._is_open = True
        self._recorded_schedule = pd.DataFrame(columns=self.fsp.index)

    def step(self, x_fsp, wait):
        """
        Advance the simulation by `wait` hours applying the given pump schedule.

        Parameters
        ----------
        x_fsp : DataFrame, shape (wait, n_combos)
            Binary combination schedule. Columns are FSP combination names matching self.fsp.index.
        wait : int
            Number of hours to advance.

        Returns
        -------
        tank_volumes : DataFrame, shape (wait, n_tanks)
            Volume trajectory (m^3) per tank. Columns are tank names.
        energy : DataFrame, shape (wait, n_pumps)
            Energy consumed (kWh) per physical pump per hour. Columns are EPANET pump IDs.
        """
        epanet_pump_indices = self.epanet_pump_idx.values.tolist()
        tank_volumes = pd.DataFrame(index=range(wait), columns=self.tanks.index, dtype=float)
        energy = pd.DataFrame(index=range(wait), columns=self.epanet_pump_idx.index, dtype=float)
        step_start_h = self._elapsed_h

        for h in range(1, wait + 1):
            # Translate combination schedule to physical pump statuses via membership matrix
            combo_row = x_fsp.iloc[h - 1]
            pump_status = (self.pump_membership.T @ combo_row) > 0

            for pump_id, is_on in pump_status.items():
                self.d.setLinkStatus(self.epanet_pump_idx[pump_id], int(is_on))

            target_h = step_start_h + h
            hour_energy = pd.Series(0.0, index=self.epanet_pump_idx.index)

            # Advance through EPANET sub-steps until 1 hour has elapsed
            while self._elapsed_h < target_h:
                self.d.runHydraulicAnalysis()

                pump_flows = np.array(self.d.getLinkFlows(epanet_pump_indices))  # m^3/h
                pump_heads = np.abs(np.array(self.d.getLinkHeadloss(epanet_pump_indices)))  # m
                # Clamp eta away from zero: when a pump is OFF, flow=0 and an efficiency
                # curve may also return 0, causing 0/0 = NaN without this guard.
                pump_eta = np.maximum(np.array(self.d.getLinkPumpEfficiency(epanet_pump_indices)), 1e-6)
                power = pd.Series(pump_flows * pump_heads * 9.81 / 3600 / pump_eta, index=self.epanet_pump_idx.index)

                tstep = self.d.nextHydraulicAnalysisStep()
                dt_h = tstep / 3600
                hour_energy += power * dt_h
                self._elapsed_h += dt_h

                if tstep == 0:
                    break

            energy.iloc[h - 1] = hour_energy

            # Tank volumes at the end of this hour
            vols = self.d.getNodeTankVolume(self.tank_epanet_idx.tolist())
            tank_volumes.iloc[h - 1] = pd.Series(vols, index=self.tanks.index)

        self._recorded_schedule = pd.concat([self._recorded_schedule, x_fsp.iloc[:wait]], ignore_index=True)
        return tank_volumes, energy

    def close(self):
        """Close the hydraulic analysis and unload the EPANET model."""
        if self._is_open:
            self.d.closeHydraulicAnalysis()
            self._is_open = False
        self.d.unload()

    def export(self, path):
        """Save the network with the recorded pump schedule as a new .inp file.

        The combination schedule accumulated across all step() calls is translated
        to physical pump on/off patterns (one per EPANET pump) and written into the
        .inp file so the full policy is visible in the EPANET GUI.
        The output is stripped of EPANET 2.3-only fields for compatibility with
        the EPANET 2.2 GUI.
        """
        if not self._recorded_schedule.empty:
            # Set simulation duration to match recorded steps
            total_hours = len(self._recorded_schedule)
            self.d.setTimeSimulationDuration(total_hours * 3600)

            # Translate combination schedule to physical pump schedule
            physical_schedule = (self._recorded_schedule @ self.pump_membership) > 0
            for pump_id in self.epanet_pump_idx.index:
                pattern_name = f'sched_pump_{pump_id}'
                self.d.addPattern(pattern_name, physical_schedule[pump_id].astype(float).tolist())
                pat_idx = self.d.getPatternIndex(pattern_name)
                self.d.setLinkPumpPatternIndex(self.epanet_pump_idx[pump_id], pat_idx)

        self.d.saveInputFile(path)
        self._strip_v23_fields(path)

    @staticmethod
    def _strip_v23_fields(path):
        _V23_PATTERNS = ('[LEAKAGE]', 'BACKFLOW ALLOWED')
        with open(path, 'r') as f:
            lines = f.readlines()
        with open(path, 'w') as f:
            for line in lines:
                if any(p in line for p in _V23_PATTERNS):
                    continue
                f.write(line)
