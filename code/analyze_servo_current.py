from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATASETS = ROOT / 'datasets'

WITHOUT_SPRING_CSV = DATASETS / 'Grupo5_Sin_Resorte(in).csv'
WITH_SPRING_CSV = DATASETS / 'Grupo5_Con_Resorte(in).csv'
WITHOUT_SPRING_ACTIVE_TRIM_CSV = DATASETS / 'Grupo5_Sin_Resorte_active_trim.csv'
WITH_SPRING_ACTIVE_TRIM_CSV = DATASETS / 'Grupo5_Con_Resorte_active_trim.csv'

SUMMARY_CSV = DATASETS / 'servo_current_summary.csv'
COMPARISON_CSV = DATASETS / 'servo_current_comparison.csv'
PLOT_PNG = DATASETS / 'servo_current_comparison.png'
ACTIVE_THRESHOLD_MA = 10.0

# The notebook reports theoretical torques in the order [theta0, theta1].
# The physical robot mapping confirmed during calibration is:
#   Servo 1 -> theta1
#   Servo 2 -> theta0
THEORETICAL_PEAK_REDUCTION_PERCENT = {
    'Servo 1': 68.95494986,  # theta1
    'Servo 2': 55.43307453,  # theta0
}

THEORETICAL_RMS_REDUCTION_PERCENT = {
    'Servo 1': 85.57743519,  # theta1
    'Servo 2': 48.65977870,  # theta0
}


def load_measurement(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(
        columns={
            'Tiempo (s)': 'time_s',
            'Corriente Servo 1 (mA)': 'servo_1_mA',
            'Corriente Servo 2 (mA)': 'servo_2_mA',
        }
    )
    return df.sort_values('time_s').reset_index(drop=True)


def normalize_timebase(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy().reset_index(drop=True)
    if len(normalized) > 0:
        normalized['time_s'] = normalized['time_s'] - float(normalized['time_s'].iloc[0])
    return normalized


def compute_metrics(df: pd.DataFrame, case_name: str, window_name: str) -> list[dict]:
    if len(df) == 0:
        return []

    t = df['time_s'].to_numpy(dtype=float)
    duration = float(t[-1] - t[0]) if len(t) > 1 else 0.0
    source_start = float(df.attrs.get('source_start_time_s', t[0]))
    source_end = float(df.attrs.get('source_end_time_s', t[-1]))
    threshold_mA = df.attrs.get('threshold_mA', np.nan)

    rows: list[dict] = []
    for label, column in [('Servo 1', 'servo_1_mA'), ('Servo 2', 'servo_2_mA')]:
        signal = df[column].to_numpy(dtype=float)
        signal_abs = np.abs(signal)
        rows.append(
            {
                'case': case_name,
                'window': window_name,
                'servo': label,
                'samples': int(len(df)),
                'duration_s': duration,
                'source_start_time_s': source_start,
                'source_end_time_s': source_end,
                'threshold_mA': threshold_mA,
                'mean_abs_mA': float(np.mean(signal_abs)),
                'rms_mA': float(np.sqrt(np.mean(signal**2))),
                'peak_abs_mA': float(np.max(signal_abs)),
                'charge_mAs': float(np.trapezoid(signal_abs, t)),
                'charge_mAh': float(np.trapezoid(signal_abs, t) / 3600.0),
            }
        )

    return rows


def active_window(df: pd.DataFrame, threshold_mA: float = ACTIVE_THRESHOLD_MA) -> pd.DataFrame:
    total_current = np.abs(df['servo_1_mA']) + np.abs(df['servo_2_mA'])
    idx = np.flatnonzero(total_current.to_numpy(dtype=float) > threshold_mA)
    if len(idx) == 0:
        trimmed = normalize_timebase(df)
        trimmed.attrs['source_start_time_s'] = float(df['time_s'].iloc[0]) if len(df) > 0 else 0.0
        trimmed.attrs['source_end_time_s'] = float(df['time_s'].iloc[-1]) if len(df) > 0 else 0.0
        trimmed.attrs['threshold_mA'] = threshold_mA
        return trimmed

    start_idx = int(idx[0])
    end_idx = int(idx[-1])
    trimmed = normalize_timebase(df.iloc[start_idx : end_idx + 1])
    trimmed.attrs['source_start_time_s'] = float(df['time_s'].iloc[start_idx])
    trimmed.attrs['source_end_time_s'] = float(df['time_s'].iloc[end_idx])
    trimmed.attrs['threshold_mA'] = threshold_mA
    return trimmed


def matched_duration_window(df: pd.DataFrame, duration_s: float) -> pd.DataFrame:
    window = df[df['time_s'] <= duration_s].reset_index(drop=True)
    window.attrs = dict(df.attrs)
    source_start = float(window.attrs.get('source_start_time_s', 0.0))
    source_end = source_start + float(window['time_s'].iloc[-1]) if len(window) > 0 else source_start
    window.attrs['source_start_time_s'] = source_start
    window.attrs['source_end_time_s'] = source_end
    return window


def save_active_trimmed_measurements() -> None:
    without_active = active_window(load_measurement(WITHOUT_SPRING_CSV))
    with_active = active_window(load_measurement(WITH_SPRING_CSV))
    without_active.to_csv(WITHOUT_SPRING_ACTIVE_TRIM_CSV, index=False)
    with_active.to_csv(WITH_SPRING_ACTIVE_TRIM_CSV, index=False)


def build_summary() -> tuple[pd.DataFrame, pd.DataFrame]:
    without_spring = load_measurement(WITHOUT_SPRING_CSV)
    with_spring = load_measurement(WITH_SPRING_CSV)
    without_active = active_window(without_spring)
    with_active = active_window(with_spring)
    matched_active_duration_s = min(
        float(without_active['time_s'].iloc[-1] - without_active['time_s'].iloc[0]),
        float(with_active['time_s'].iloc[-1] - with_active['time_s'].iloc[0]),
    )

    summary_rows: list[dict] = []
    summary_rows += compute_metrics(without_spring, 'Without spring', 'full_trace')
    summary_rows += compute_metrics(with_spring, 'With spring', 'full_trace')

    summary_rows += compute_metrics(
        without_active,
        'Without spring',
        'active_trim',
    )
    summary_rows += compute_metrics(with_active, 'With spring', 'active_trim')

    summary_rows += compute_metrics(
        matched_duration_window(without_active, matched_active_duration_s),
        'Without spring',
        'matched_active_duration',
    )
    summary_rows += compute_metrics(
        matched_duration_window(with_active, matched_active_duration_s),
        'With spring',
        'matched_active_duration',
    )

    summary = pd.DataFrame(summary_rows)

    comparison_rows: list[dict] = []
    for window_name in ['active_trim', 'matched_active_duration', 'full_trace']:
        without_window = (
            summary[(summary['case'] == 'Without spring') & (summary['window'] == window_name)]
            .set_index('servo')
            .sort_index()
        )
        with_window = (
            summary[(summary['case'] == 'With spring') & (summary['window'] == window_name)]
            .set_index('servo')
            .sort_index()
        )

        for servo in without_window.index:
            row_without = without_window.loc[servo]
            row_with = with_window.loc[servo]

            comparison_rows.append(
                {
                    'window': window_name,
                    'servo': servo,
                    'experimental_mean_abs_change_percent': 100.0
                    * (row_with['mean_abs_mA'] - row_without['mean_abs_mA'])
                    / row_without['mean_abs_mA'],
                    'experimental_rms_change_percent': 100.0
                    * (row_with['rms_mA'] - row_without['rms_mA'])
                    / row_without['rms_mA'],
                    'experimental_peak_change_percent': 100.0
                    * (row_with['peak_abs_mA'] - row_without['peak_abs_mA'])
                    / row_without['peak_abs_mA'],
                    'experimental_charge_change_percent': 100.0
                    * (row_with['charge_mAs'] - row_without['charge_mAs'])
                    / row_without['charge_mAs'],
                    'theoretical_peak_reduction_percent': THEORETICAL_PEAK_REDUCTION_PERCENT[servo],
                    'theoretical_rms_reduction_percent': THEORETICAL_RMS_REDUCTION_PERCENT[servo],
                }
            )

    comparison = pd.DataFrame(comparison_rows)
    return summary, comparison


def save_plot() -> None:
    without_spring = active_window(load_measurement(WITHOUT_SPRING_CSV))
    with_spring = active_window(load_measurement(WITH_SPRING_CSV))

    fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)

    axes[0, 0].plot(without_spring['time_s'], without_spring['servo_1_mA'], label='Without spring', color='tab:red')
    axes[0, 0].plot(with_spring['time_s'], with_spring['servo_1_mA'], label='With spring', color='tab:green')
    axes[0, 0].set_title('Servo 1 current, active trim')
    axes[0, 0].set_ylabel('Current [mA]')
    axes[0, 0].set_xlabel('Trimmed time [s]')
    axes[0, 0].legend()

    axes[0, 1].plot(without_spring['time_s'], without_spring['servo_2_mA'], label='Without spring', color='tab:red')
    axes[0, 1].plot(with_spring['time_s'], with_spring['servo_2_mA'], label='With spring', color='tab:green')
    axes[0, 1].set_title('Servo 2 current, active trim')
    axes[0, 1].set_xlabel('Trimmed time [s]')
    axes[0, 1].legend()

    summary, _ = build_summary()
    trimmed = summary[summary['window'] == 'active_trim']
    pivot_mean = trimmed.pivot(index='servo', columns='case', values='mean_abs_mA')
    pivot_rms = trimmed.pivot(index='servo', columns='case', values='rms_mA')

    pivot_mean.plot(kind='bar', ax=axes[1, 0], color=['tab:red', 'tab:green'])
    axes[1, 0].set_title('Mean absolute current, active trim')
    axes[1, 0].set_ylabel('Current [mA]')
    axes[1, 0].tick_params(axis='x', rotation=0)

    pivot_rms.plot(kind='bar', ax=axes[1, 1], color=['tab:red', 'tab:green'])
    axes[1, 1].set_title('RMS current, active trim')
    axes[1, 1].set_ylabel('Current [mA]')
    axes[1, 1].tick_params(axis='x', rotation=0)

    fig.savefig(PLOT_PNG, dpi=180)
    plt.close(fig)


def main() -> int:
    save_active_trimmed_measurements()
    summary, comparison = build_summary()
    summary.to_csv(SUMMARY_CSV, index=False)
    comparison.to_csv(COMPARISON_CSV, index=False)
    save_plot()

    print('Saved active trim without spring to:', WITHOUT_SPRING_ACTIVE_TRIM_CSV)
    print('Saved active trim with spring to:', WITH_SPRING_ACTIVE_TRIM_CSV)
    print('Saved summary to:', SUMMARY_CSV)
    print('Saved comparison to:', COMPARISON_CSV)
    print('Saved plot to:', PLOT_PNG)
    print()
    print(summary.to_string(index=False))
    print()
    print(comparison.to_string(index=False))

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
