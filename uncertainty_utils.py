from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass
class UncertaintyResult:
    """Holds the estimated mean and covariance."""
    mu: np.ndarray            # shape (p,)
    Sigma: np.ndarray         # shape (p, p)
    # Optional context
    vector_size: int
    variables: List[str]      # in the stacking order used
    hours_per_window: int     # usually 24


class SpatioTemporalEllipsoidUncertaintyEstimator:
    """
    Estimate mean (mu) and covariance (Sigma) of a stacked daily vector:
        y = [var1(h=0..23), var2(h=0..23), ..., varK(h=0..23)]  in R^{24*K}

    Data requirements
    -----------------
    - `df` must be hourly, indexed by pandas.DatetimeIndex (timezone-aware or naive is fine),
      with numeric columns corresponding to variables (e.g., consumers and electricity_price).
    - Rows represent timestamps; columns are variables.
    - The estimator forms DAILY windows; by default it drops days with incomplete 24 hours.

    Seasonality handling
    --------------------
    If `seasonal_demean=True`, we compute per-variable, per-hour-of-day means over the
    available days (baseline), subtract them, and estimate covariance on residuals.
    We still keep the baseline so you can reconstruct a full mean vector if desired.

    Stacking order
    --------------
    Variables are stacked by variable, then hour:
        [var1@h0, var1@h1, ..., var1@h23, var2@h0, ..., varK@h23]
    """

    def __init__(
        self,
        variables: Optional[List[str]] = None,
        hours_per_window: int = 24,
        drop_incomplete_windows: bool = True,
        seasonal_demean: bool = True,
    ):
        self.variables_in = variables
        self.hours_per_window = hours_per_window
        self.drop_incomplete_windows = drop_incomplete_windows
        self.seasonal_demean = seasonal_demean

        # Learned during fit
        self.variables_: List[str] = []
        self.baseline_: Optional[pd.DataFrame] = None  # (H x K) DataFrame
        self.mu_raw_: Optional[np.ndarray] = None
        self.Sigma_raw_: Optional[np.ndarray] = None
        self.mu_resid_: Optional[np.ndarray] = None
        self.Sigma_resid_: Optional[np.ndarray] = None

        # Internal artifacts to support re-estimation for arbitrary horizons
        self._index_map_: Dict[int, Tuple[str, int]] = {}
        self._cube_: Optional[np.ndarray] = None        # shape (D, K, H)
        self._days_: Optional[List[pd.Timestamp]] = None
        self._H_: Optional[int] = None                  # convenience alias for hours_per_window
        self._K_: Optional[int] = None                  # number of variables

    # ------------------------- Public API -------------------------

    def fit(self, df: pd.DataFrame) -> "SpatioTemporalEllipsoidUncertaintyEstimator":
        """Fit on an hourly DataFrame with a DatetimeIndex and numeric columns."""
        df = self._validate_and_prepare(df)
        vars_ = self.variables_in or [c for c in df.columns if np.issubdtype(df[c].dtype, np.number)]
        if not vars_:
            raise ValueError("No numeric variables found. Provide `variables=` or ensure numeric columns exist.")
        self.variables_ = vars_

        # Daily windows
        H = self.hours_per_window
        day_groups = df.groupby(df.index.normalize())
        complete = []
        for day, g in day_groups:
            if len(g) == H:
                complete.append((day, g.sort_index()))
        if not complete:
            raise ValueError("No complete daily windows found.")

        self._days_ = [d for d, _ in complete]
        cube = np.stack([self._day_to_array(g[self.variables_]) for _, g in complete], axis=0)  # (D,K,H)
        self._cube_ = cube
        self._K_ = cube.shape[1]
        self._H_ = H

        # Seasonality baseline (K x H)
        if self.seasonal_demean:
            baseline_KH = cube.mean(axis=0)  # (K,H)
            self.baseline_ = pd.DataFrame(
                baseline_KH.T,  # (H,K)
                index=pd.RangeIndex(H, name="hour"),
                columns=self.variables_,
            )
            resid_cube = cube - baseline_KH[None, :, :]
        else:
            self.baseline_ = None
            resid_cube = cube

        # Vectorize (D, K*H)
        Y_raw = self._vec_stack(cube)
        Y_resid = self._vec_stack(resid_cube)
        # Sample stats
        self.mu_raw_ = Y_raw.mean(axis=0)
        self.Sigma_raw_ = np.cov(Y_raw, rowvar=False)
        self.mu_resid_ = Y_resid.mean(axis=0)
        self.Sigma_resid_ = np.cov(Y_resid, rowvar=False)

        # Index map
        self._index_map_ = self._build_index_map()

        return self

    def get_uncertainty_set(
            self,
            use_residuals: bool = True,
            start_hour: int = 0,
            horizon: Optional[int] = None,
            recompute: bool = False,
    ) -> UncertaintyResult:
        """
        Return an ellipsoidal set for the specified start_hour and horizon.
        - If horizon is None: return the full fitted horizon (default H hours).
        - If horizon == H: rotate to start_hour.
        - If horizon < H: extract_horizon(horizon, start_hour, ...).
        """
        if horizon is None or horizon == self._H_:
            return self.rotate_to_start_hour(start_hour, use_residuals=use_residuals)
        else:
            return self.extract_horizon(
                horizon=horizon,
                start_hour=start_hour,
                use_residuals=use_residuals,
                recompute=recompute,
            )

    def rotate_to_start_hour(
        self,
        start_hour: int,
        use_residuals: bool = True,
    ) -> UncertaintyResult:
        """Rotate hours inside each variable-block so hour=0 becomes `start_hour`."""
        self._require_fit()
        H = self._H_
        K = self._K_
        s = start_hour % H
        mu, Sigma = self._pick_muSigma(use_residuals)

        # permutation indices for a single block of length H
        block_idx = list(range(H))
        rot = block_idx[s:] + block_idx[:s]  # [s..H-1, 0..s-1]

        # build global permutation across K blocks
        perm = []
        for k in range(K):
            base = k * H
            perm.extend([base + r for r in rot])

        mu_rot = mu[perm]
        Sigma_rot = Sigma[np.ix_(perm, perm)]

        return UncertaintyResult(
            mu=mu_rot,
            Sigma=Sigma_rot,
            vector_size=mu_rot.shape[0],
            variables=self.variables_.copy(),
            hours_per_window=H,
        )

    def extract_horizon(
        self,
        horizon: int,
        start_hour: int = 0,
        use_residuals: bool = True,
        recompute: bool = False,
    ) -> UncertaintyResult:
        """
        Produce an ellipsoid for a T-hour horizon starting at `start_hour`.

        - recompute=False: rotate then SELECT first T hours per variable.
        - recompute=True: re-estimate mean/cov from T-length windows starting at `start_hour`
                          (wrap across days if needed). Better when T << H.
        """
        self._require_fit()
        H = self._H_
        K = self._K_
        T = int(horizon)
        if not (1 <= T <= H):
            raise ValueError(f"horizon must be in [1, {H}]")

        if not recompute:
            # Rotate then select
            base = self.rotate_to_start_hour(start_hour, use_residuals=use_residuals)
            # selection indices: keep first T entries from each block
            sel = []
            for k in range(K):
                block_base = k * H
                sel.extend(range(block_base, block_base + T))
            mu_T = base.mu[sel]
            Sigma_T = base.Sigma[np.ix_(sel, sel)]
            return UncertaintyResult(
                mu=mu_T,
                Sigma=Sigma_T,
                vector_size=mu_T.shape[0],
                variables=self.variables_.copy(),
                hours_per_window=T,
            )

        # ---- recompute path: build samples (M, K, T) then estimate ----
        cube = self._cube_  # (D,K,H)
        assert cube is not None
        s = start_hour % H

        # Build windows across days with wrap handling
        # For each day i:
        #   if s+T <= H: take cube[i,:,s:s+T]
        #   else: concat cube[i,:,s:H] with cube[i+1,:,0:(s+T-H)] (needs next day)
        segments = []
        D = cube.shape[0]
        wrap = s + T - H
        if T <= H - s:
            # no wrap; use all D days
            for i in range(D):
                seg = cube[i, :, s:s+T]  # (K,T)
                segments.append(seg)
        else:
            # wrap; need pairs (i, i+1)
            for i in range(D - 1):  # last day can't wrap to next
                left = cube[i, :, s:H]              # (K, H-s)
                right = cube[i + 1, :, 0:wrap]      # (K, wrap)
                seg = np.concatenate([left, right], axis=1)  # (K,T)
                segments.append(seg)

        if len(segments) < 2:
            raise ValueError(
                "Not enough consecutive days to build recomputed windows for the requested start_hour and horizon."
            )

        sample_cube = np.stack(segments, axis=0)  # (M,K,T)

        # Seasonality demeaning for recompute path
        if self.seasonal_demean and self.baseline_ is not None:
            baseline_KH = self.baseline_.to_numpy().T  # (K,H)
            if T <= H - s:
                base_slice = baseline_KH[:, s:s+T]     # (K,T)
                resid_cube = sample_cube - base_slice[None, :, :]
            else:
                left = baseline_KH[:, s:H]
                right = baseline_KH[:, 0:wrap]
                base_slice = np.concatenate([left, right], axis=1)  # (K,T)
                resid_cube = sample_cube - base_slice[None, :, :]
        else:
            resid_cube = sample_cube

        # Vectorize and estimate
        Y = resid_cube.reshape(resid_cube.shape[0], self._K_ * T)  # (M, K*T)
        mu_T = Y.mean(axis=0)
        Sigma_T = np.cov(Y, rowvar=False)

        return UncertaintyResult(
            mu=mu_T,
            Sigma=Sigma_T,
            vector_size=mu_T.shape[0],
            variables=self.variables_.copy(),
            hours_per_window=T,
        )

    def baseline_matrix(self) -> Optional[pd.DataFrame]:
        return None if self.baseline_ is None else self.baseline_.copy()

    def index_to_var_hour(self, idx: int) -> Tuple[str, int]:
        if not self._index_map_:
            raise RuntimeError("Estimator not fit.")
        return self._index_map_[idx]

    def var_hour_to_index(self, var: str, hour: int) -> int:
        if var not in self.variables_:
            raise KeyError(f"Unknown variable '{var}'. Known: {self.variables_}")
        H = self.hours_per_window
        if not (0 <= hour < H):
            raise ValueError(f"hour must be in [0, {H-1}]")
        return self.variables_.index(var) * H + hour

    # ------------------------- Internal helpers -------------------------

    def _pick_muSigma(self, use_residuals: bool) -> Tuple[np.ndarray, np.ndarray]:
        self._require_fit()
        if use_residuals:
            return self.mu_resid_, self.Sigma_resid_
        else:
            return self.mu_raw_, self.Sigma_raw_

    def _require_fit(self):
        if self._cube_ is None or self.mu_raw_ is None or self.Sigma_raw_ is None:
            raise RuntimeError("Estimator not fit.")

    def _validate_and_prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame index must be a pandas.DatetimeIndex.")
        if df.index.freq is None:
            try:
                df = df.asfreq("h")
            except Exception:
                pass
        return df.sort_index()

    def _day_to_array(self, gvars: pd.DataFrame) -> np.ndarray:
        H = self.hours_per_window
        if len(gvars) != H:
            raise ValueError("Internal: day group not of expected length.")
        return gvars.to_numpy(dtype=float).reshape(H, -1).T  # (K,H)

    def _vec_stack(self, cube: np.ndarray) -> np.ndarray:
        D, K, H = cube.shape
        return cube.reshape(D, K * H)

    def _build_index_map(self) -> Dict[int, Tuple[str, int]]:
        K = len(self.variables_)
        H = self.hours_per_window
        return {k * H + h: (self.variables_[k], h) for k in range(K) for h in range(H)}


# ------------------------- Example usage (for reference) -------------------------
# df must be an hourly DataFrame with a DatetimeIndex and columns:
# ["consumer1_demand", ..., "consumerN_demand", "electricity_price"]

# df = pd.read_csv("data/Labs_IDC_Hourly_20250812_150341.csv", index_col=0, parse_dates=True,
#                  usecols=['Delivery Date', 'Best bid'])
# df = pd.read_csv("data/InflowData_1.csv", index_col=0, parse_dates=True)
# df.index = pd.to_datetime(df.index, dayfirst=True)
# print(df)
# est = SpatioTemporalUncertaintyEstimator(
#     seasonal_demean=True,             # remove hour-of-day baseline
#     drop_incomplete_windows=True      # require full 24h days
# )
# est.fit(df)
# #
# # # Ellipsoid on residuals (recommended if you want fluctuations around daily patterns)
# U = est.get_uncertainty_set(use_residuals=True)
# mu, Sigma = U.mu, U.Sigma
#
# plt.imshow(Sigma, cmap='viridis')
# plt.show()

# # If you prefer raw statistics without demeaning:
# U_raw = est.get_uncertainty_set(use_residuals=False)
#
# # See the hour-of-day baseline (variables x hours):
# baseline = est.baseline_matrix()
#
# # Interpret positions in the stacked vector:
# var, hour = est.index_to_var_hour(37)
# idx = est.var_hour_to_index("electricity_price", 15)
