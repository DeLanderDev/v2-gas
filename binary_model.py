"""
Binary Classification Heads for Kalshi Contract Trading

Three key components:
  1. Binary Classification Heads - XGBoost classifiers for threshold crossings
     "Will gas close > $X.XX?" for each strike price
  2. Probability Calibration - Isotonic regression for true probabilities
  3. Expected Value Framework - EV calculation for Kalshi contracts

Directly optimizes for the Kalshi contract structure rather than
deriving binary outcomes from continuous predictions.
"""

import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier

from config import XGBOOST_PARAMS

warnings.filterwarnings("ignore")


# ═══════════════════════════════════════════════════════════════════════════════
#  Strike Generation
# ═══════════════════════════════════════════════════════════════════════════════

def generate_strikes(
    current_price: float,
    step: float = 0.05,
    n_above: int = 4,
    n_below: int = 4,
) -> List[float]:
    """
    Generate Kalshi-style strike prices around the current price.
    Rounds to nearest step, then generates strikes above and below.
    """
    base = round(round(current_price / step) * step, 2)
    strikes = []
    for i in range(-n_below, n_above + 1):
        s = round(base + i * step, 2)
        if s > 0:
            strikes.append(s)
    return sorted(strikes)


# ═══════════════════════════════════════════════════════════════════════════════
#  Binary Classification Heads
# ═══════════════════════════════════════════════════════════════════════════════

class BinaryHeads:
    """
    Trains calibrated XGBoost binary classifiers for each strike price.

    For each strike S, the target is:  y = 1 if next_week_price > S else 0

    Also trains a movement classifier:
      y = 1 if |next_week_price - current_price| > move_threshold else 0
    """

    # XGBoost params tuned for classification
    _CLF_PARAMS = {
        "n_estimators": 200,
        "max_depth": 3,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.7,
        "min_child_weight": 5,
        "reg_alpha": 0.1,
        "reg_lambda": 1.5,
        "random_state": 42,
        "eval_metric": "logloss",
        "use_label_encoder": False,
    }

    def __init__(self):
        self.strike_models: Dict[float, CalibratedClassifierCV] = {}
        self.move_model: Optional[CalibratedClassifierCV] = None
        self.feature_names: List[str] = []
        self.strikes: List[float] = []
        self.move_threshold: float = 0.02
        self.is_trained: bool = False
        self._train_stats: Dict = {}

    def train(
        self,
        X: pd.DataFrame,
        y_abs: pd.Series,
        bases: pd.Series,
        feature_names: List[str],
        strikes: List[float],
        move_threshold: float = 0.02,
    ) -> Dict:
        """
        Train calibrated binary classifiers for each strike and movement.

        Args:
            X: Feature matrix
            y_abs: Actual next-week absolute prices (target)
            bases: Current-week prices (base for movement calc)
            feature_names: Feature column names
            strikes: Strike prices to train classifiers for
            move_threshold: Movement threshold in dollars (default $0.02)

        Returns:
            Training stats dict
        """
        self.feature_names = feature_names
        self.strikes = strikes
        self.move_threshold = move_threshold

        Xf = X[feature_names].values
        y_abs_arr = y_abs.values
        bases_arr = bases.values

        n_splits = min(3, max(2, len(X) // 60))
        tscv = TimeSeriesSplit(n_splits=n_splits)

        stats = {"n_samples": len(X), "n_strikes": len(strikes)}

        # ── Train strike classifiers ──────────────────────────────────────
        for strike in strikes:
            y_binary = (y_abs_arr > strike).astype(int)
            pos_rate = y_binary.mean()

            # Skip strikes with extreme class imbalance (<5% or >95%)
            if pos_rate < 0.05 or pos_rate > 0.95:
                stats[f"strike_{strike}_skipped"] = True
                stats[f"strike_{strike}_pos_rate"] = round(pos_rate, 3)
                continue

            base_clf = XGBClassifier(**self._CLF_PARAMS)

            try:
                cal_clf = CalibratedClassifierCV(
                    base_clf,
                    method="isotonic",
                    cv=tscv,
                )
                cal_clf.fit(Xf, y_binary)
                self.strike_models[strike] = cal_clf
                stats[f"strike_{strike}_pos_rate"] = round(pos_rate, 3)
            except Exception:
                # Fall back to sigmoid calibration if isotonic fails
                try:
                    cal_clf = CalibratedClassifierCV(
                        base_clf,
                        method="sigmoid",
                        cv=tscv,
                    )
                    cal_clf.fit(Xf, y_binary)
                    self.strike_models[strike] = cal_clf
                    stats[f"strike_{strike}_pos_rate"] = round(pos_rate, 3)
                    stats[f"strike_{strike}_fallback"] = "sigmoid"
                except Exception:
                    stats[f"strike_{strike}_skipped"] = True

        # ── Train movement classifier ─────────────────────────────────────
        y_move = (np.abs(y_abs_arr - bases_arr) > move_threshold).astype(int)
        move_pos_rate = y_move.mean()

        if 0.05 < move_pos_rate < 0.95:
            base_clf = XGBClassifier(**self._CLF_PARAMS)
            try:
                cal_clf = CalibratedClassifierCV(
                    base_clf,
                    method="isotonic",
                    cv=tscv,
                )
                cal_clf.fit(Xf, y_move)
                self.move_model = cal_clf
                stats["move_pos_rate"] = round(move_pos_rate, 3)
            except Exception:
                try:
                    cal_clf = CalibratedClassifierCV(
                        base_clf,
                        method="sigmoid",
                        cv=tscv,
                    )
                    cal_clf.fit(Xf, y_move)
                    self.move_model = cal_clf
                    stats["move_pos_rate"] = round(move_pos_rate, 3)
                    stats["move_fallback"] = "sigmoid"
                except Exception:
                    stats["move_skipped"] = True

        stats["n_trained_strikes"] = len(self.strike_models)
        self.is_trained = len(self.strike_models) > 0
        self._train_stats = stats
        return stats

    def predict_probs(self, X_row: pd.DataFrame) -> Dict[str, float]:
        """
        Get calibrated probabilities for each strike and movement.

        Returns dict like:
            {
                "above_3.05": 0.72,
                "above_3.10": 0.45,
                ...
                "move_gt_2c": 0.63,
            }
        """
        if not self.is_trained:
            return {}

        Xf = X_row[self.feature_names].values.reshape(1, -1)
        probs = {}

        for strike, model in self.strike_models.items():
            prob = model.predict_proba(Xf)[0, 1]
            probs[f"above_{strike:.2f}"] = round(float(prob), 4)

        if self.move_model is not None:
            prob = self.move_model.predict_proba(Xf)[0, 1]
            probs[f"move_gt_{int(self.move_threshold * 100)}c"] = round(float(prob), 4)

        return probs

    def walk_forward_validate(
        self,
        X: pd.DataFrame,
        y_abs: pd.Series,
        bases: pd.Series,
        feature_names: List[str],
        strikes: List[float],
        move_threshold: float = 0.02,
        test_weeks: int = 26,
    ) -> Dict:
        """
        Walk-forward validation for binary heads.
        Measures calibration quality (Brier score) and accuracy.
        """
        n = len(X)
        test_n = min(test_weeks, n // 3)
        test_start = n - test_n
        min_train = max(52, n // 3)

        if test_start < min_train:
            return {"error": "Not enough data for binary validation"}

        Xf = X[feature_names].values
        y_abs_arr = y_abs.values
        bases_arr = bases.values

        # Track per-strike results
        strike_results = {s: {"preds": [], "actuals": []} for s in strikes}
        move_results = {"preds": [], "actuals": []}

        n_splits = min(3, max(2, test_start // 60))
        tscv = TimeSeriesSplit(n_splits=n_splits)

        for i in range(test_start, n):
            Xtr = Xf[:i]
            y_tr_abs = y_abs_arr[:i]
            b_tr = bases_arr[:i]

            X_test = Xf[i:i + 1]
            y_test_abs = y_abs_arr[i]
            b_test = bases_arr[i]

            for strike in strikes:
                y_binary = (y_tr_abs > strike).astype(int)
                pos_rate = y_binary.mean()
                if pos_rate < 0.05 or pos_rate > 0.95:
                    continue

                try:
                    clf = XGBClassifier(**self._CLF_PARAMS)
                    cal = CalibratedClassifierCV(clf, method="isotonic", cv=tscv)
                    cal.fit(Xtr, y_binary)
                    prob = cal.predict_proba(X_test)[0, 1]
                    actual = 1 if y_test_abs > strike else 0
                    strike_results[strike]["preds"].append(prob)
                    strike_results[strike]["actuals"].append(actual)
                except Exception:
                    pass

            # Movement classifier
            y_move = (np.abs(y_tr_abs - b_tr) > move_threshold).astype(int)
            pos_rate = y_move.mean()
            if 0.05 < pos_rate < 0.95:
                try:
                    clf = XGBClassifier(**self._CLF_PARAMS)
                    cal = CalibratedClassifierCV(clf, method="isotonic", cv=tscv)
                    cal.fit(Xtr, y_move)
                    prob = cal.predict_proba(X_test)[0, 1]
                    actual = 1 if abs(y_test_abs - b_test) > move_threshold else 0
                    move_results["preds"].append(prob)
                    move_results["actuals"].append(actual)
                except Exception:
                    pass

        # Compute metrics
        metrics = {}
        all_preds, all_actuals = [], []

        for strike in strikes:
            preds = np.array(strike_results[strike]["preds"])
            actuals = np.array(strike_results[strike]["actuals"])
            if len(preds) < 3:
                continue

            brier = float(np.mean((preds - actuals) ** 2))
            accuracy = float(np.mean((preds >= 0.5) == actuals))
            metrics[f"strike_{strike:.2f}_brier"] = round(brier, 4)
            metrics[f"strike_{strike:.2f}_accuracy"] = round(accuracy, 4)
            metrics[f"strike_{strike:.2f}_n"] = len(preds)
            all_preds.extend(preds)
            all_actuals.extend(actuals)

        if len(move_results["preds"]) >= 3:
            preds = np.array(move_results["preds"])
            actuals = np.array(move_results["actuals"])
            metrics["move_brier"] = round(float(np.mean((preds - actuals) ** 2)), 4)
            metrics["move_accuracy"] = round(float(np.mean((preds >= 0.5) == actuals)), 4)
            metrics["move_n"] = len(preds)

        if all_preds:
            all_p = np.array(all_preds)
            all_a = np.array(all_actuals)
            metrics["overall_brier"] = round(float(np.mean((all_p - all_a) ** 2)), 4)
            metrics["overall_accuracy"] = round(float(np.mean((all_p >= 0.5) == all_a)), 4)
            metrics["overall_n"] = len(all_p)

        return metrics


# ═══════════════════════════════════════════════════════════════════════════════
#  Expected Value Framework
# ═══════════════════════════════════════════════════════════════════════════════

def calculate_ev(
    model_prob: float,
    market_price: float,
    contract_payout: float = 1.00,
) -> Dict[str, float]:
    """
    Calculate expected value for a Kalshi binary contract.

    Kalshi contracts pay $1.00 if the event occurs, $0 otherwise.
    You can buy YES at market_price or buy NO at (1 - market_price).

    Args:
        model_prob: Model's calibrated probability of YES outcome
        market_price: Kalshi market price for YES contract (0.01 to 0.99)
        contract_payout: Contract payout on win (default $1.00)

    Returns:
        Dict with EV calculations for both YES and NO positions
    """
    if not (0.01 <= market_price <= 0.99):
        return {"error": "Market price must be between 0.01 and 0.99"}

    # YES position: pay market_price, win contract_payout if event occurs
    yes_cost = market_price
    yes_win = contract_payout - yes_cost  # profit if YES
    yes_loss = -yes_cost                  # loss if NO
    yes_ev = model_prob * yes_win + (1 - model_prob) * yes_loss

    # NO position: pay (1 - market_price), win contract_payout if event doesn't occur
    no_cost = contract_payout - market_price
    no_win = contract_payout - no_cost   # profit if NO
    no_loss = -no_cost                   # loss if YES
    no_ev = (1 - model_prob) * no_win + model_prob * no_loss

    # Edge = model_prob - implied_prob
    yes_edge = model_prob - market_price
    no_edge = (1 - model_prob) - (1 - market_price)  # same magnitude, opposite sign

    # ROI
    yes_roi = (yes_ev / yes_cost) * 100 if yes_cost > 0 else 0
    no_roi = (no_ev / no_cost) * 100 if no_cost > 0 else 0

    # Kelly criterion for optimal bet sizing
    # f* = (bp - q) / b where b = odds, p = prob of win, q = 1-p
    yes_b = yes_win / yes_cost if yes_cost > 0 else 0
    yes_kelly = max(0, (yes_b * model_prob - (1 - model_prob)) / yes_b) if yes_b > 0 else 0

    no_b = no_win / no_cost if no_cost > 0 else 0
    no_kelly = max(0, (no_b * (1 - model_prob) - model_prob) / no_b) if no_b > 0 else 0

    return {
        "model_prob": round(model_prob, 4),
        "market_price": round(market_price, 2),
        "yes_ev": round(yes_ev, 4),
        "no_ev": round(no_ev, 4),
        "yes_edge": round(yes_edge, 4),
        "no_edge": round(no_edge, 4),
        "yes_roi_pct": round(yes_roi, 2),
        "no_roi_pct": round(no_roi, 2),
        "yes_kelly": round(yes_kelly, 4),
        "no_kelly": round(no_kelly, 4),
        "recommended": "YES" if yes_ev > no_ev and yes_ev > 0 else ("NO" if no_ev > 0 else "PASS"),
        "best_ev": round(max(yes_ev, no_ev), 4),
        "best_edge": round(max(abs(yes_edge), abs(no_edge)), 4),
    }


def build_contract_table(
    probs: Dict[str, float],
    market_prices: Optional[Dict[str, float]] = None,
    min_edge: float = 0.03,
) -> pd.DataFrame:
    """
    Build a table of all contracts with EV analysis.

    Args:
        probs: Dict of calibrated probabilities from BinaryHeads.predict_probs()
        market_prices: Dict of Kalshi market prices (same keys as probs).
                      If None, shows probabilities only without EV.
        min_edge: Minimum edge to flag as a trade opportunity

    Returns:
        DataFrame with contract analysis
    """
    rows = []
    for contract_key, model_prob in probs.items():
        row = {
            "contract": contract_key,
            "model_prob": model_prob,
            "model_prob_pct": f"{model_prob * 100:.1f}%",
        }

        if market_prices and contract_key in market_prices:
            mkt = market_prices[contract_key]
            ev = calculate_ev(model_prob, mkt)
            row.update({
                "market_price": mkt,
                "market_price_pct": f"{mkt * 100:.0f}c",
                "yes_ev": ev["yes_ev"],
                "no_ev": ev["no_ev"],
                "edge": ev["yes_edge"],
                "recommended": ev["recommended"],
                "best_ev": ev["best_ev"],
                "kelly": ev["yes_kelly"] if ev["recommended"] == "YES" else ev["no_kelly"],
                "trade_signal": abs(ev["yes_edge"]) >= min_edge,
            })
        else:
            row.update({
                "market_price": None,
                "edge": None,
                "recommended": "N/A",
                "best_ev": None,
                "trade_signal": False,
            })

        rows.append(row)

    df = pd.DataFrame(rows)
    if not df.empty and "best_ev" in df.columns:
        df = df.sort_values("best_ev", ascending=False, na_position="last")
    return df.reset_index(drop=True)
