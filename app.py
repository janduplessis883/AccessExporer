from __future__ import annotations

from io import BytesIO
from typing import Iterable

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


APPOINTMENT_DATE = "Appointment date"
APPOINTMENT_STATUS = "Appointment status"
CLINICIAN = "Clinician"
ROTA_TYPE = "Rota type"
PATIENT_COUNT = "Patient Count"

REQUIRED_COLUMNS = {
    APPOINTMENT_DATE,
    APPOINTMENT_STATUS,
    CLINICIAN,
    ROTA_TYPE,
}

DEFAULT_LIST_SIZE = 13958
CURRENT_YEAR_LIST_SIZE = 13605
DEFAULT_APPOINTMENTS_PER_1000 = 85

# Editable defaults for England and Wales bank holidays that commonly affect GP access.
DEFAULT_BANK_HOLIDAYS = [
    "2025-04-18",
    "2025-04-21",
    "2025-05-05",
    "2025-05-26",
    "2025-08-25",
    "2025-12-25",
    "2025-12-26",
    "2026-01-01",
    "2026-04-03",
    "2026-04-06",
    "2026-05-04",
    "2026-05-25",
    "2026-08-31",
    "2026-12-25",
    "2026-12-28",
    "2027-01-01",
    "2027-03-26",
    "2027-03-29",
    "2027-05-03",
    "2027-05-31",
    "2027-08-30",
    "2027-12-27",
    "2027-12-28",
]


st.set_page_config(page_title="AccessExplorer", layout="wide")


@st.cache_data(show_spinner=False)
def read_csv_upload(name: str, content: bytes) -> pd.DataFrame:
    df = pd.read_csv(BytesIO(content))
    df["Source file"] = name
    return df


def load_uploads(files: Iterable) -> pd.DataFrame:
    frames = []
    for file in files or []:
        frames.append(read_csv_upload(file.name, file.getvalue()))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def prepare_data(df: pd.DataFrame, label: str) -> pd.DataFrame:
    if df.empty:
        return df

    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        missing_text = ", ".join(sorted(missing))
        st.error(f"{label} is missing required columns: {missing_text}")
        return pd.DataFrame()

    prepared = df.copy()
    prepared[APPOINTMENT_DATE] = pd.to_datetime(
        prepared[APPOINTMENT_DATE], format="mixed", dayfirst=True, errors="coerce"
    )
    prepared = prepared.dropna(subset=[APPOINTMENT_DATE])

    if PATIENT_COUNT in prepared.columns:
        prepared[PATIENT_COUNT] = pd.to_numeric(prepared[PATIENT_COUNT], errors="coerce").fillna(1)
    else:
        prepared[PATIENT_COUNT] = 1

    for column in [APPOINTMENT_STATUS, CLINICIAN, ROTA_TYPE]:
        prepared[column] = prepared[column].fillna("Unknown").astype(str)

    prepared["Week start"] = week_start_monday(prepared[APPOINTMENT_DATE])
    prepared["Financial year"] = prepared[APPOINTMENT_DATE].apply(financial_year_label)
    prepared["Financial week"] = prepared[APPOINTMENT_DATE].apply(financial_week)
    prepared["Dataset"] = label
    return prepared


def financial_year_label(date: pd.Timestamp) -> str:
    start_year = date.year if date.month >= 4 else date.year - 1
    return f"{start_year % 100:02d}/{(start_year + 1) % 100:02d}"


def financial_year_start(date: pd.Timestamp) -> pd.Timestamp:
    start_year = date.year if date.month >= 4 else date.year - 1
    return pd.Timestamp(start_year, 4, 1)


def financial_week(date: pd.Timestamp) -> int:
    fy_start = financial_year_start(date)
    first_week_start = week_start_monday(pd.Series([fy_start])).iloc[0]
    return int(((week_start_monday(pd.Series([date])).iloc[0] - first_week_start).days // 7) + 1)


def week_start_monday(dates: pd.Series) -> pd.Series:
    return dates.dt.to_period("W-SUN").dt.start_time


def options_for(df: pd.DataFrame, column: str) -> list[str]:
    if df.empty or column not in df.columns:
        return []
    return sorted(df[column].dropna().astype(str).unique().tolist())


def apply_filters(
    df: pd.DataFrame,
    clinicians: list[str],
    rota_types: list[str],
    statuses: list[str],
    date_range: tuple[pd.Timestamp, pd.Timestamp] | None,
) -> pd.DataFrame:
    if df.empty:
        return df

    filtered = df.copy()
    if clinicians:
        filtered = filtered[filtered[CLINICIAN].isin(clinicians)]
    if rota_types:
        filtered = filtered[filtered[ROTA_TYPE].isin(rota_types)]
    if statuses:
        filtered = filtered[filtered[APPOINTMENT_STATUS].isin(statuses)]
    if date_range:
        start, end = date_range
        filtered = filtered[
            (filtered[APPOINTMENT_DATE].dt.date >= start)
            & (filtered[APPOINTMENT_DATE].dt.date <= end)
        ]
    return filtered


def remove_access_exclusions(df: pd.DataFrame, exclude_hca: bool, exclude_arrs: bool) -> pd.DataFrame:
    if df.empty:
        return df

    filtered = df.copy()

    if exclude_hca:
        hca_rota_types = {"hca session", "session"}
        filtered = filtered[
            ~filtered[ROTA_TYPE].str.strip().str.lower().isin(hca_rota_types)
        ]

    if exclude_arrs:
        arrs_rota_types = {"stanhope mews pharmacist arrs"}
        filtered = filtered[
            ~filtered[ROTA_TYPE].str.strip().str.lower().isin(arrs_rota_types)
        ]

    return filtered


def access_capacity_frame(
    df: pd.DataFrame,
    previous_list_size: int,
    current_list_size: int,
    appointments_per_1000: int,
    bank_holidays: set[pd.Timestamp],
    exclude_hca: bool,
    exclude_arrs: bool,
    weekly_arrs_contribution: float,
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    capacity = remove_access_exclusions(df, exclude_hca, exclude_arrs)

    weekly = (
        capacity.groupby(["Dataset", "Financial week", "Week start"], as_index=False)[PATIENT_COUNT]
        .sum()
        .rename(columns={PATIENT_COUNT: "Appointments"})
    )
    weekly = apply_weekly_arrs_contribution(weekly, weekly_arrs_contribution)

    weekly["List size"] = weekly["Dataset"].apply(
        lambda dataset: list_size_for_dataset(dataset, previous_list_size, current_list_size)
    )
    full_week_target = appointments_per_1000 * (weekly["List size"] / 1000)
    weekly["Working days"] = weekly["Week start"].apply(
        lambda week: working_days_in_week(week, bank_holidays)
    )
    weekly["Target"] = weekly["Working days"] / 5 * full_week_target
    weekly["Appointments per 1000"] = weekly["Appointments"] / (weekly["List size"] / 1000)
    weekly["Variance to target"] = weekly["Appointments"] - weekly["Target"]
    return weekly


def list_size_for_dataset(dataset: str, previous_list_size: int, current_list_size: int) -> int:
    return current_list_size if dataset == "Current FY" else previous_list_size


def apply_weekly_arrs_contribution(
    weekly: pd.DataFrame, weekly_arrs_contribution: float
) -> pd.DataFrame:
    if weekly.empty:
        return weekly

    adjusted = weekly.copy()
    adjusted["Base appointments"] = adjusted["Appointments"]
    adjusted["ARRS contribution"] = 0.0
    current_mask = adjusted["Dataset"] == "Current FY"
    adjusted.loc[current_mask, "ARRS contribution"] = weekly_arrs_contribution
    adjusted["Appointments"] = adjusted["Base appointments"] + adjusted["ARRS contribution"]
    return adjusted


def like_for_like_periods(as_of_date: pd.Timestamp) -> dict[str, tuple[pd.Timestamp, pd.Timestamp]]:
    current_start = financial_year_start(as_of_date)
    elapsed_days = (as_of_date.normalize() - current_start).days
    previous_start = current_start - pd.DateOffset(years=1)
    return {
        "Previous FY": (previous_start, previous_start + pd.Timedelta(days=elapsed_days)),
        "Current FY": (current_start, as_of_date.normalize()),
    }


def period_label(start: pd.Timestamp, end: pd.Timestamp) -> str:
    return f"{start:%-d %b %Y} to {end:%-d %b %Y}"


def like_for_like_summary(
    previous_df: pd.DataFrame,
    current_df: pd.DataFrame,
    as_of_date: pd.Timestamp,
    previous_list_size: int,
    current_list_size: int,
    bank_holidays: set[pd.Timestamp],
    exclude_hca: bool,
    exclude_arrs: bool,
    weekly_arrs_contribution: float,
) -> pd.DataFrame:
    frames = {"Previous FY": previous_df, "Current FY": current_df}
    periods = like_for_like_periods(as_of_date)
    rows = []

    for dataset, frame in frames.items():
        start, end = periods[dataset]
        list_size = list_size_for_dataset(dataset, previous_list_size, current_list_size)
        if frame.empty or APPOINTMENT_DATE not in frame.columns:
            period_frame = pd.DataFrame()
        else:
            period_frame = frame[
                (frame[APPOINTMENT_DATE] >= start) & (frame[APPOINTMENT_DATE] <= end)
            ]
        access_frame = remove_access_exclusions(period_frame, exclude_hca, exclude_arrs)
        working_days = working_days_between(start, end, bank_holidays)
        weeks = max(((end - start).days + 1) / 7, 1 / 7)
        total_appointments = (
            period_frame[PATIENT_COUNT].sum()
            if not period_frame.empty and PATIENT_COUNT in period_frame.columns
            else 0
        )
        access_appointments = (
            access_frame[PATIENT_COUNT].sum()
            if not access_frame.empty and PATIENT_COUNT in access_frame.columns
            else 0
        )
        arrs_contribution = weekly_arrs_contribution * weeks if dataset == "Current FY" else 0
        adjusted_access_appointments = access_appointments + arrs_contribution

        rows.append(
            {
                "Dataset": dataset,
                "Period": period_label(start, end),
                "Start date": start.date(),
                "End date": end.date(),
                "List size": list_size,
                "Total appointments": total_appointments,
                "Access appointments": access_appointments,
                "ARRS contribution": arrs_contribution,
                "Adjusted access appointments": adjusted_access_appointments,
                "Working days": working_days,
                "Elapsed weeks": weeks,
                "Appointments per 1,000/week": (
                    adjusted_access_appointments / (list_size / 1000) / weeks if weeks else 0
                ),
            }
        )

    summary = pd.DataFrame(rows)
    if not summary.empty:
        previous_total = summary.loc[summary["Dataset"] == "Previous FY", "Total appointments"].sum()
        previous_rate = summary.loc[
            summary["Dataset"] == "Previous FY", "Appointments per 1,000/week"
        ].sum()
        summary["Appointment difference vs previous"] = summary["Total appointments"] - previous_total
        summary["Rate difference vs previous"] = (
            summary["Appointments per 1,000/week"] - previous_rate
        )
    return summary


def clinician_contribution_summary(
    previous_df: pd.DataFrame,
    current_df: pd.DataFrame,
    as_of_date: pd.Timestamp,
    previous_list_size: int,
    current_list_size: int,
    exclude_hca: bool,
    exclude_arrs: bool,
) -> pd.DataFrame:
    frames = {"Previous FY": previous_df, "Current FY": current_df}
    periods = like_for_like_periods(as_of_date)
    rows = []

    for dataset, frame in frames.items():
        start, end = periods[dataset]
        elapsed_weeks = max(((end - start).days + 1) / 7, 1 / 7)
        list_size = list_size_for_dataset(dataset, previous_list_size, current_list_size)
        denominator = (list_size / 1000) * elapsed_weeks

        if frame.empty or APPOINTMENT_DATE not in frame.columns:
            period_frame = pd.DataFrame()
        else:
            period_frame = frame[
                (frame[APPOINTMENT_DATE] >= start) & (frame[APPOINTMENT_DATE] <= end)
            ]

        access_frame = remove_access_exclusions(period_frame, exclude_hca, exclude_arrs)
        if access_frame.empty or CLINICIAN not in access_frame.columns:
            continue

        clinician_totals = (
            access_frame.groupby(CLINICIAN, as_index=False)[PATIENT_COUNT]
            .sum()
            .rename(columns={PATIENT_COUNT: "Appointments"})
        )
        total_rate = (
            clinician_totals["Appointments"].sum() / denominator if denominator else 0
        )

        for row in clinician_totals.to_dict("records"):
            contribution = row["Appointments"] / denominator if denominator else 0
            rows.append(
                {
                    "Dataset": dataset,
                    "Clinician": row[CLINICIAN],
                    "Period": period_label(start, end),
                    "List size": list_size,
                    "Appointments": row["Appointments"],
                    "Elapsed weeks": elapsed_weeks,
                    "Appointments per 1,000/week": contribution,
                    "Share of total": contribution / total_rate if total_rate else 0,
                }
            )

    return pd.DataFrame(rows)


def weekly_appointment_totals(df: pd.DataFrame, weekly_arrs_contribution: float) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    weekly = (
        df.groupby(["Dataset", "Financial week", "Week start"], as_index=False)[PATIENT_COUNT]
        .sum()
        .rename(columns={PATIENT_COUNT: "Appointments"})
    )
    weekly = apply_weekly_arrs_contribution(weekly, weekly_arrs_contribution)
    weekly["Week label"] = "Week " + weekly["Financial week"].astype(str)
    return weekly


def clinical_completed_rota_mask(df: pd.DataFrame) -> pd.Series:
    rota_type = df[ROTA_TYPE].str.strip().str.lower()
    return rota_type.str.contains(r"\bgp\b", na=False, regex=True) | rota_type.isin(
        {"nurse", "stanhope mews nurse", "nurse practitioner", "stanhope mews nurse practitioner"}
    )


def clinical_completed_weekday_heatmap(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    completed_clinical = df[
        (df[APPOINTMENT_STATUS].str.strip().str.lower() == "finished")
        & clinical_completed_rota_mask(df)
    ].copy()
    if completed_clinical.empty:
        return pd.DataFrame()

    weekday_order = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    completed_clinical["Day of week"] = pd.Categorical(
        completed_clinical[APPOINTMENT_DATE].dt.day_name().str[:3],
        categories=weekday_order,
        ordered=True,
    )

    heatmap = completed_clinical.pivot_table(
        index=CLINICIAN,
        columns="Day of week",
        values=PATIENT_COUNT,
        aggfunc="sum",
        fill_value=0,
        observed=False,
    ).reindex(columns=weekday_order, fill_value=0)

    clinician_order = heatmap.sum(axis=1).sort_values(ascending=False).index
    return heatmap.loc[clinician_order]


def clinical_completed_week_heatmap(df: pd.DataFrame, financial_week: int) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    completed_clinical = df[
        (df[APPOINTMENT_STATUS].str.strip().str.lower() == "finished")
        & clinical_completed_rota_mask(df)
        & (df["Financial week"] == financial_week)
    ].copy()
    if completed_clinical.empty:
        return pd.DataFrame()

    weekday_order = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    completed_clinical["Day of week"] = pd.Categorical(
        completed_clinical[APPOINTMENT_DATE].dt.day_name().str[:3],
        categories=weekday_order,
        ordered=True,
    )

    heatmap = completed_clinical.pivot_table(
        index=CLINICIAN,
        columns="Day of week",
        values=PATIENT_COUNT,
        aggfunc="sum",
        fill_value=0,
        observed=False,
    ).reindex(columns=weekday_order, fill_value=0)

    clinician_order = heatmap.sum(axis=1).sort_values(ascending=False).index
    heatmap = heatmap.loc[clinician_order]
    heatmap["Total"] = heatmap.sum(axis=1)
    totals = heatmap.sum(axis=0).to_frame().T
    totals.index = ["Total"]
    return pd.concat([heatmap, totals])


def financial_year_weeks_elapsed(as_of_date: pd.Timestamp) -> float:
    start = financial_year_start(as_of_date)
    return max(((as_of_date.normalize() - start).days + 1) / 7, 1 / 7)


def projection_summary(
    current_df: pd.DataFrame,
    as_of_date: pd.Timestamp,
    weekly_arrs_contribution: float,
    current_list_size: int,
    appointments_per_1000: int,
) -> dict[str, float | int | str]:
    current_start = financial_year_start(as_of_date)
    current_end = current_start + pd.DateOffset(years=1) - pd.Timedelta(days=1)
    as_of_date = min(as_of_date.normalize(), current_end)

    if current_df.empty or APPOINTMENT_DATE not in current_df.columns:
        observed = pd.DataFrame()
    else:
        observed = current_df[
            (current_df[APPOINTMENT_DATE] >= current_start)
            & (current_df[APPOINTMENT_DATE] <= as_of_date)
        ]

    appointments_to_date = (
        observed[PATIENT_COUNT].sum()
        if not observed.empty and PATIENT_COUNT in observed.columns
        else 0
    )
    elapsed_weeks = max(((as_of_date - current_start).days + 1) / 7, 1 / 7)
    arrs_to_date = weekly_arrs_contribution * elapsed_weeks
    adjusted_appointments_to_date = appointments_to_date + arrs_to_date
    full_year_weeks = ((current_end - current_start).days + 1) / 7
    weeks_remaining = max(full_year_weeks - elapsed_weeks, 0)
    average_per_week = adjusted_appointments_to_date / elapsed_weeks if elapsed_weeks else 0
    projected_full_year = average_per_week * full_year_weeks
    full_year_target = appointments_per_1000 * (current_list_size / 1000) * full_year_weeks
    appointments_needed = max(full_year_target - adjusted_appointments_to_date, 0)
    required_per_remaining_week = (
        appointments_needed / weeks_remaining if weeks_remaining else 0
    )

    return {
        "fy_start": period_label(current_start, as_of_date),
        "full_year": period_label(current_start, current_end),
        "appointments_to_date": appointments_to_date,
        "arrs_to_date": arrs_to_date,
        "adjusted_appointments_to_date": adjusted_appointments_to_date,
        "elapsed_weeks": elapsed_weeks,
        "average_per_week": average_per_week,
        "full_year_weeks": full_year_weeks,
        "weeks_remaining": weeks_remaining,
        "full_year_target": full_year_target,
        "appointments_needed": appointments_needed,
        "required_per_remaining_week": required_per_remaining_week,
        "projected_full_year": projected_full_year,
        "remaining_projected": max(projected_full_year - adjusted_appointments_to_date, 0),
    }


def working_days_between(
    start: pd.Timestamp, end: pd.Timestamp, bank_holidays: set[pd.Timestamp]
) -> int:
    days = pd.date_range(start, end, freq="D")
    return sum(
        day.weekday() < 5 and pd.Timestamp(day.date()) not in bank_holidays
        for day in days
    )


def working_days_in_week(week_start: pd.Timestamp, bank_holidays: set[pd.Timestamp]) -> int:
    weekdays = pd.date_range(week_start, periods=7, freq="D")
    normal_workdays = [day for day in weekdays if day.weekday() < 5]
    return sum(pd.Timestamp(day.date()) not in bank_holidays for day in normal_workdays)


def parse_bank_holidays(raw_text: str) -> set[pd.Timestamp]:
    holidays = set()
    for line in raw_text.replace(",", "\n").splitlines():
        value = line.strip()
        if not value:
            continue
        parsed = pd.to_datetime(value, dayfirst=False, errors="coerce")
        if pd.notna(parsed):
            holidays.add(pd.Timestamp(parsed.date()))
    return holidays


def format_delta(value: float) -> str:
    return f"{value:+,.0f}"


def render_metric_cards(
    like_for_like: pd.DataFrame,
    appointments_per_1000: int,
    current_list_size: int,
) -> None:
    previous_row = like_for_like[like_for_like["Dataset"] == "Previous FY"]
    current_row = like_for_like[like_for_like["Dataset"] == "Current FY"]

    previous_total = (
        previous_row["Adjusted access appointments"].sum() if not previous_row.empty else 0
    )
    current_total = current_row["Adjusted access appointments"].sum() if not current_row.empty else 0
    current_weeks_elapsed = (
        current_row["Elapsed weeks"].sum() if not current_row.empty else 0
    )
    current_target = appointments_per_1000 * (current_list_size / 1000) * current_weeks_elapsed
    current_per_1000_per_week = (
        current_row["Appointments per 1,000/week"].sum() if not current_row.empty else 0
    )

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Previous FY access appointments", f"{previous_total:,.0f}")
    col2.metric(
        "Current FY access appointments",
        f"{current_total:,.0f}",
        format_delta(current_total - previous_total),
    )
    col3.metric(
        "Current target",
        f"{current_target:,.0f}",
        format_delta(current_total - current_target),
    )
    col4.metric("Current FY weeks elapsed", f"{current_weeks_elapsed:,.1f}")
    col5.metric(
        "Current per 1,000/week",
        f"{current_per_1000_per_week:,.1f}",
        help=(
            "Current access appointments divided by "
            f"(({current_list_size:,} / 1,000) x {current_weeks_elapsed:,.1f} elapsed FY weeks)."
        ),
    )


st.markdown(
    """
    <div class="access-explorer-logo" role="img" aria-label="AccessExplorer">
        <div class="access-explorer-mark">A</div>
        <div class="access-explorer-wordmark">AccessExplorer</div>
    </div>
    <style>
        .access-explorer-logo {
            display: flex;
            align-items: center;
            gap: 1.35rem;
            margin: 0.35rem 0 0.25rem;
        }

        .access-explorer-mark {
            display: flex;
            align-items: center;
            justify-content: center;
            width: 4.75rem;
            height: 4.75rem;
            border-radius: 1rem;
            background: #0b2d42;
            color: #42d5d3;
            font-size: 2.15rem;
            font-weight: 800;
            line-height: 1;
            letter-spacing: 0;
        }

        .access-explorer-wordmark {
            color: #0b2d42;
            font-size: 2.15rem;
            font-weight: 800;
            line-height: 1;
            letter-spacing: 0;
        }

        @media (max-width: 640px) {
            .access-explorer-logo {
                gap: 1rem;
            }

            .access-explorer-mark {
                width: 3.75rem;
                height: 3.75rem;
                border-radius: 0.85rem;
                font-size: 1.75rem;
            }

            .access-explorer-wordmark {
                font-size: 1.75rem;
            }
        }
    </style>
    """,
    unsafe_allow_html=True,
)
st.caption("Compare appointment access across two financial years using uploaded CSV extracts.")

with st.sidebar:
    st.header("Upload data")
    previous_files = st.file_uploader(
        "Previous financial year CSVs",
        type=["csv"],
        accept_multiple_files=True,
        help="Upload one or more CSV files for 2025/26.",
    )
    current_files = st.file_uploader(
        "Current financial year CSVs",
        type=["csv"],
        accept_multiple_files=True,
        help="Upload one or more CSV files for 2026/27.",
    )

    with st.expander("Use sample data"):
        use_sample_as_previous = st.checkbox("Use sample as previous", value=not previous_files)
        use_sample_as_current = st.checkbox("Use sample as current", value=False)

    st.header("Access settings")
    previous_list_size = st.number_input(
        "Previous FY list size",
        min_value=1,
        value=DEFAULT_LIST_SIZE,
        step=50,
    )
    current_list_size = st.number_input(
        "Current FY list size",
        min_value=1,
        value=CURRENT_YEAR_LIST_SIZE,
        step=50,
    )
    appointments_per_1000 = st.number_input(
        "Weekly appointments per 1,000 target",
        min_value=1,
        value=DEFAULT_APPOINTMENTS_PER_1000,
        step=5,
    )
    weekly_arrs_contribution = st.number_input(
        "Weekly ARRS contribution",
        min_value=0.0,
        value=273.0,
        step=1.0,
        help="Extra ARRS appointments that can be added to Current FY totals.",
    )
    include_weekly_arrs_contribution = st.toggle(
        "Include weekly ARRS contribution in totals",
        value=False,
        help="When on, the weekly ARRS contribution is added to Current FY total appointment calculations.",
    )
    include_bank_holiday_adjustment = st.checkbox("Adjust target for bank holidays", value=True)
    bank_holiday_text = st.text_area(
        "Bank holidays",
        value="\n".join(DEFAULT_BANK_HOLIDAYS),
        height=150,
        disabled=not include_bank_holiday_adjustment,
    )
    exclude_hca = st.checkbox("Exclude HCA / healthcare assistant sessions from access target", value=True)
    exclude_arrs = st.checkbox("Exclude ARRS / pharmacist sessions from access target", value=True)
    like_for_like_as_at = st.date_input(
        "Like-for-like comparison date",
        value=pd.Timestamp.today().date(),
        help="Compares 1 April to this date against the same financial-year day last year.",
    )


load_progress = st.progress(0, text="Waiting for appointment data...")

load_progress.progress(10, text="Reading uploaded CSV files...")
previous_raw = load_uploads(previous_files)
current_raw = load_uploads(current_files)

load_progress.progress(35, text="Checking sample data options...")
if (previous_raw.empty and use_sample_as_previous) or (current_raw.empty and use_sample_as_current):
    sample_df = pd.read_csv("data/access_example.csv")
    if previous_raw.empty and use_sample_as_previous:
        previous_raw = sample_df.assign(**{"Source file": "access_example.csv"})
    if current_raw.empty and use_sample_as_current:
        current_raw = sample_df.assign(**{"Source file": "access_example.csv"})

load_progress.progress(60, text="Preparing previous financial year data...")
previous_df = prepare_data(previous_raw, "Previous FY")

load_progress.progress(80, text="Preparing current financial year data...")
current_df = prepare_data(current_raw, "Current FY")

combined_df = pd.concat([previous_df, current_df], ignore_index=True)
load_progress.progress(
    100,
    text=(
        f"Loaded {len(previous_df):,} previous FY rows and "
        f"{len(current_df):,} current FY rows."
    ),
)

if previous_df.empty and current_df.empty:
    st.info("Upload CSV extracts for the previous and current financial years to begin.")
    st.stop()

st.subheader("Filters")
filter_col1, filter_col2, filter_col3 = st.columns(3)
clinician_options = options_for(combined_df, CLINICIAN)
rota_options = options_for(combined_df, ROTA_TYPE)
status_options = options_for(combined_df, APPOINTMENT_STATUS)

selected_clinicians = filter_col1.multiselect("Clinician", clinician_options, default=clinician_options)
selected_rota_types = filter_col2.multiselect("Rota type", rota_options, default=rota_options)
selected_statuses = filter_col3.multiselect("Appointment status", status_options, default=status_options)

date_col1, date_col2 = st.columns(2)
previous_range = None
current_range = None

if not previous_df.empty:
    previous_min = previous_df[APPOINTMENT_DATE].min().date()
    previous_max = previous_df[APPOINTMENT_DATE].max().date()
    previous_range = date_col1.slider(
        "Previous FY date range",
        min_value=previous_min,
        max_value=previous_max,
        value=(previous_min, previous_max),
    )
else:
    date_col1.info("No previous FY data loaded.")

if not current_df.empty:
    current_min = current_df[APPOINTMENT_DATE].min().date()
    current_max = current_df[APPOINTMENT_DATE].max().date()
    current_range = date_col2.slider(
        "Current FY date range",
        min_value=current_min,
        max_value=current_max,
        value=(current_min, current_max),
    )
else:
    date_col2.info("No current FY data loaded.")

previous_filtered = apply_filters(
    previous_df, selected_clinicians, selected_rota_types, selected_statuses, previous_range
)
current_filtered = apply_filters(
    current_df, selected_clinicians, selected_rota_types, selected_statuses, current_range
)
filtered_combined = pd.concat([previous_filtered, current_filtered], ignore_index=True)

previous_filtered_without_date = apply_filters(
    previous_df, selected_clinicians, selected_rota_types, selected_statuses, None
)
current_filtered_without_date = apply_filters(
    current_df, selected_clinicians, selected_rota_types, selected_statuses, None
)

bank_holidays = parse_bank_holidays(bank_holiday_text) if include_bank_holiday_adjustment else set()
effective_weekly_arrs_contribution = (
    weekly_arrs_contribution if include_weekly_arrs_contribution else 0.0
)
weekly_capacity = access_capacity_frame(
    filtered_combined,
    previous_list_size=previous_list_size,
    current_list_size=current_list_size,
    appointments_per_1000=appointments_per_1000,
    bank_holidays=bank_holidays,
    exclude_hca=exclude_hca,
    exclude_arrs=exclude_arrs,
    weekly_arrs_contribution=effective_weekly_arrs_contribution,
)
previous_weekly = weekly_capacity[weekly_capacity["Dataset"] == "Previous FY"]
current_weekly = weekly_capacity[weekly_capacity["Dataset"] == "Current FY"]
like_for_like = like_for_like_summary(
    previous_filtered_without_date,
    current_filtered_without_date,
    pd.Timestamp(like_for_like_as_at),
    previous_list_size=previous_list_size,
    current_list_size=current_list_size,
    bank_holidays=bank_holidays,
    exclude_hca=exclude_hca,
    exclude_arrs=exclude_arrs,
    weekly_arrs_contribution=effective_weekly_arrs_contribution,
)
clinician_contribution = clinician_contribution_summary(
    previous_filtered_without_date,
    current_filtered_without_date,
    pd.Timestamp(like_for_like_as_at),
    previous_list_size=previous_list_size,
    current_list_size=current_list_size,
    exclude_hca=exclude_hca,
    exclude_arrs=exclude_arrs,
)
weekly_totals = weekly_appointment_totals(filtered_combined, effective_weekly_arrs_contribution)
appointment_map = pd.DataFrame()
if not filtered_combined.empty:
    appointment_map = (
        filtered_combined.groupby(
            ["Dataset", APPOINTMENT_DATE, CLINICIAN, ROTA_TYPE], as_index=False
        )[PATIENT_COUNT]
        .sum()
        .rename(columns={PATIENT_COUNT: "Appointments"})
    )
projection = projection_summary(
    current_filtered_without_date,
    pd.Timestamp(like_for_like_as_at),
    effective_weekly_arrs_contribution,
    current_list_size,
    appointments_per_1000,
)

render_metric_cards(like_for_like, appointments_per_1000, current_list_size)

tab0, tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(
    [
        "Like-for-like",
        "Weekly totals",
        "Clinician contribution",
        "Date clinician map",
        "Week clinician heatmap",
        "Weekly access",
        "Rota and clinician mix",
        "Status summary",
        "Filtered data",
    ]
)

with tab0:
    st.subheader("Like-for-like year-to-date comparison")
    st.caption(
        "Compares 1 April to the selected date with the same elapsed period in the previous financial year. "
        "Clinician, rota type, and status filters apply; HCA and ARRS exclusions apply to the per-1,000/week rate."
    )

    if like_for_like.empty:
        st.warning("No rows remain after the selected filters.")
    else:
        previous_period = like_for_like.loc[
            like_for_like["Dataset"] == "Previous FY", "Period"
        ].iloc[0]
        current_period = like_for_like.loc[
            like_for_like["Dataset"] == "Current FY", "Period"
        ].iloc[0]

        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        previous_total = like_for_like.loc[
            like_for_like["Dataset"] == "Previous FY", "Total appointments"
        ].sum()
        current_total = like_for_like.loc[
            like_for_like["Dataset"] == "Current FY", "Total appointments"
        ].sum()
        previous_rate = like_for_like.loc[
            like_for_like["Dataset"] == "Previous FY", "Appointments per 1,000/week"
        ].sum()
        current_rate = like_for_like.loc[
            like_for_like["Dataset"] == "Current FY", "Appointments per 1,000/week"
        ].sum()
        metric_col1.metric("Previous period", previous_period)
        metric_col2.metric("Current period", current_period)
        metric_col3.metric(
            "Appointment difference",
            format_delta(current_total - previous_total),
        )
        metric_col4.metric(
            "Per 1,000/week difference",
            f"{current_rate - previous_rate:+,.1f}",
        )

        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=like_for_like["Dataset"],
                y=like_for_like["Total appointments"],
                name="Total appointments",
                marker_color="#2E86AB",
                text=like_for_like["Total appointments"].round(0),
                textposition="auto",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=like_for_like["Dataset"],
                y=like_for_like["Appointments per 1,000/week"],
                name="Appointments per 1,000/week",
                mode="lines+markers+text",
                marker=dict(size=12, color="#F18F01"),
                line=dict(width=3, color="#F18F01"),
                text=like_for_like["Appointments per 1,000/week"].round(1),
                textposition="top center",
                yaxis="y2",
            )
        )
        fig.update_layout(
            barmode="group",
            hovermode="x unified",
            legend_title="",
            yaxis=dict(title="Total appointments"),
            yaxis2=dict(
                title="Appointments per 1,000/week",
                overlaying="y",
                side="right",
                rangemode="tozero",
            ),
        )
        st.plotly_chart(fig, width="stretch")

        display_summary = like_for_like[
            [
                "Dataset",
                "Period",
                "List size",
                "Total appointments",
                "Access appointments",
                "Working days",
                "Elapsed weeks",
                "Appointments per 1,000/week",
                "Appointment difference vs previous",
                "Rate difference vs previous",
            ]
        ].copy()
        display_summary["Elapsed weeks"] = display_summary["Elapsed weeks"].round(2)
        display_summary["Appointments per 1,000/week"] = display_summary[
            "Appointments per 1,000/week"
        ].round(1)
        display_summary["Rate difference vs previous"] = display_summary[
            "Rate difference vs previous"
        ].round(1)
        st.dataframe(display_summary, width="stretch", hide_index=True)

with tab1:
    st.subheader("Total appointments per week")
    st.caption(
        "Grouped by financial-year week and shown as previous versus current financial year. "
        "Clinician, rota type, appointment status, and date range filters apply."
    )

    if weekly_totals.empty:
        st.warning("No rows remain after the selected filters.")
    else:
        weekly_bar = px.bar(
            weekly_totals,
            x="Financial week",
            y="Appointments",
            color="Dataset",
            barmode="group",
            text="Appointments",
            hover_data=["Week start"],
            labels={"Financial week": "Financial-year week"},
        )
        weekly_bar.update_traces(texttemplate="%{text:,.0f}", textposition="outside")
        weekly_bar.update_layout(
            yaxis_title="Total appointments",
            legend_title="",
            uniformtext_minsize=10,
            uniformtext_mode="hide",
        )
        st.plotly_chart(weekly_bar, width="stretch")

        projection_col1, projection_col2, projection_col3, projection_col4 = st.columns(4)
        projection_col1.metric(
            "Current FY to date",
            f"{projection['adjusted_appointments_to_date']:,.0f}",
            help=projection["fy_start"],
        )
        projection_col2.metric(
            "Average appointments/week",
            f"{projection['average_per_week']:,.1f}",
            help=f"Based on {projection['elapsed_weeks']:,.1f} elapsed financial-year weeks.",
        )
        projection_col3.metric(
            "Projected full FY",
            f"{projection['projected_full_year']:,.0f}",
            help=projection["full_year"],
        )
        projection_col4.metric(
            "Projected remaining",
            f"{projection['remaining_projected']:,.0f}",
        )

        target_col1, target_col2, target_col3 = st.columns(3)
        target_col1.metric(
            "Weeks left in current FY",
            f"{projection['weeks_remaining']:,.1f}",
            help=projection["full_year"],
        )
        target_col2.metric(
            "Appointments still needed",
            f"{projection['appointments_needed']:,.0f}",
            help=f"Full-year target: {projection['full_year_target']:,.0f} appointments.",
        )
        target_col3.metric(
            "Needed per remaining week",
            f"{projection['required_per_remaining_week']:,.1f}",
            help="Average appointments needed each remaining week to meet 85 per 1,000/week by year end.",
        )

        projection_fig = go.Figure(
            data=[
                go.Bar(
                    x=["Current to date", "Projected remaining", "Projected full FY"],
                    y=[
                        projection["adjusted_appointments_to_date"],
                        projection["remaining_projected"],
                        projection["projected_full_year"],
                    ],
                    marker_color=["#2E86AB", "#A23B72", "#F18F01"],
                    text=[
                        projection["adjusted_appointments_to_date"],
                        projection["remaining_projected"],
                        projection["projected_full_year"],
                    ],
                    textposition="outside",
                )
            ]
        )
        projection_fig.update_traces(texttemplate="%{text:,.0f}")
        projection_fig.update_layout(
            title="Current-year projection at the current average weekly rate",
            yaxis_title="Appointments",
            showlegend=False,
        )
        st.plotly_chart(projection_fig, width="stretch")

        weekly_comparison = weekly_totals.pivot_table(
            index="Financial week",
            columns="Dataset",
            values="Appointments",
            aggfunc="sum",
        ).reset_index()
        if {"Previous FY", "Current FY"}.issubset(weekly_comparison.columns):
            weekly_comparison["Current minus previous"] = (
                weekly_comparison["Current FY"] - weekly_comparison["Previous FY"]
            )
            st.dataframe(weekly_comparison, width="stretch", hide_index=True)

with tab2:
    st.subheader("Clinician contribution to appointments per 1,000/week")
    st.caption(
        "Shows how much each clinician contributes to the total access rate over the like-for-like period. "
        "The calculation is clinician appointments divided by ((list size / 1,000) x elapsed FY weeks)."
    )

    if clinician_contribution.empty:
        st.warning("No clinician appointments remain after the selected filters and access exclusions.")
    else:
        current_order = (
            clinician_contribution[clinician_contribution["Dataset"] == "Current FY"]
            .groupby("Clinician")["Appointments per 1,000/week"]
            .sum()
            .sort_values(ascending=False)
            .index.tolist()
        )
        fallback_order = (
            clinician_contribution.groupby("Clinician")["Appointments per 1,000/week"]
            .sum()
            .sort_values(ascending=False)
            .index.tolist()
        )
        clinician_order = current_order or fallback_order

        clinician_bar = px.bar(
            clinician_contribution,
            x="Clinician",
            y="Appointments per 1,000/week",
            color="Dataset",
            barmode="group",
            text="Appointments per 1,000/week",
            category_orders={"Clinician": clinician_order},
            hover_data=["Appointments", "Period", "List size", "Share of total"],
        )
        clinician_bar.update_traces(texttemplate="%{text:.1f}", textposition="outside")
        clinician_bar.update_layout(
            xaxis_title="",
            yaxis_title="Appointments per 1,000/week",
            legend_title="",
            uniformtext_minsize=10,
            uniformtext_mode="hide",
        )
        st.plotly_chart(clinician_bar, width="stretch")

        stacked_contribution = px.bar(
            clinician_contribution,
            x="Dataset",
            y="Appointments per 1,000/week",
            color="Clinician",
            text="Appointments per 1,000/week",
            hover_data=["Appointments", "Period", "List size", "Share of total"],
            title="Clinician contribution stacked to total rate",
        )
        stacked_contribution.update_traces(texttemplate="%{text:.1f}", textposition="inside")
        stacked_contribution.update_layout(
            yaxis_title="Appointments per 1,000/week",
            legend_title="Clinician",
            uniformtext_minsize=9,
            uniformtext_mode="hide",
        )
        st.plotly_chart(stacked_contribution, width="stretch")

        contribution_table = clinician_contribution.copy()
        contribution_table["Appointments per 1,000/week"] = contribution_table[
            "Appointments per 1,000/week"
        ].round(1)
        contribution_table["Elapsed weeks"] = contribution_table["Elapsed weeks"].round(2)
        contribution_table["Share of total"] = (contribution_table["Share of total"] * 100).round(1)
        contribution_table = contribution_table.sort_values(
            ["Dataset", "Appointments per 1,000/week"], ascending=[True, False]
        )
        st.dataframe(contribution_table, width="stretch", hide_index=True)

with tab3:
    st.subheader("Appointments by Date and Clinician")
    st.caption("Colored by rota type. Dot size shows the number of appointments on that date.")

    if appointment_map.empty:
        st.warning("No appointment rows remain after the selected filters.")
    else:
        dataset_options = appointment_map["Dataset"].dropna().unique().tolist()
        default_dataset_index = (
            dataset_options.index("Current FY") if "Current FY" in dataset_options else 0
        )
        selected_map_dataset = st.radio(
            "Dataset",
            dataset_options,
            index=default_dataset_index,
            horizontal=True,
        )
        map_frame = appointment_map[appointment_map["Dataset"] == selected_map_dataset].copy()
        heatmap_frame = filtered_combined[
            filtered_combined["Dataset"] == selected_map_dataset
        ].copy()

        clinical_weekday_heatmap = clinical_completed_weekday_heatmap(heatmap_frame)
        st.subheader("Completed GP, nurse, and nurse practitioner appointments by day of week")
        st.caption(
            "Counts appointments with status Finished where the rota type is GP, nurse, or nurse practitioner. "
            "Clinician, rota type, status, and date range filters apply."
        )
        if clinical_weekday_heatmap.empty:
            st.info(
                "No completed GP, nurse, or nurse practitioner appointment rows remain for the selected dataset and filters."
            )
        else:
            heatmap_fig = px.imshow(
                clinical_weekday_heatmap,
                text_auto=True,
                aspect="auto",
                color_continuous_scale=[
                    [0, "#eef8f8"],
                    [0.35, "#bfe7e5"],
                    [0.7, "#57c7c2"],
                    [1, "#0f9c95"],
                ],
                labels=dict(x="Day of week", y="Clinician", color="Completed appointments"),
            )
            heatmap_fig.update_traces(
                texttemplate="%{z:,.0f}",
                hovertemplate=(
                    "Clinician: %{y}<br>"
                    "Day: %{x}<br>"
                    "Completed appointments: %{z:,.0f}<extra></extra>"
                ),
            )
            heatmap_fig.update_layout(
                height=max(360, 34 * len(clinical_weekday_heatmap.index) + 180),
                xaxis_title="",
                yaxis_title="",
                coloraxis_colorbar_title="Completed",
                margin=dict(l=10, r=10, t=20, b=20),
            )
            heatmap_fig.update_xaxes(side="top")
            st.plotly_chart(heatmap_fig, width="stretch")

            current_fy_weeks_elapsed = financial_year_weeks_elapsed(
                pd.Timestamp(like_for_like_as_at)
            )
            average_clinical_weekday_heatmap = (
                clinical_weekday_heatmap / current_fy_weeks_elapsed
            )
            st.subheader(
                "Average completed GP, nurse, and nurse practitioner appointments per week by day of week"
            )
            st.caption(
                "Each cell is the completed GP, nurse, and nurse practitioner appointment total divided by "
                f"{current_fy_weeks_elapsed:,.1f} elapsed weeks in the current financial year."
            )
            average_heatmap_fig = px.imshow(
                average_clinical_weekday_heatmap,
                text_auto=".1f",
                aspect="auto",
                color_continuous_scale=[
                    [0, "#eef8f8"],
                    [0.35, "#bfe7e5"],
                    [0.7, "#57c7c2"],
                    [1, "#0f9c95"],
                ],
                labels=dict(x="Day of week", y="Clinician", color="Average per week"),
            )
            average_heatmap_fig.update_traces(
                texttemplate="%{z:,.1f}",
                hovertemplate=(
                    "Clinician: %{y}<br>"
                    "Day: %{x}<br>"
                    "Average completed appointments/week: %{z:,.1f}<extra></extra>"
                ),
            )
            average_heatmap_fig.update_layout(
                height=max(360, 34 * len(average_clinical_weekday_heatmap.index) + 180),
                xaxis_title="",
                yaxis_title="",
                coloraxis_colorbar_title="Avg/week",
                margin=dict(l=10, r=10, t=20, b=20),
            )
            average_heatmap_fig.update_xaxes(side="top")
            st.plotly_chart(average_heatmap_fig, width="stretch")

        clinician_order = (
            map_frame.groupby(CLINICIAN)["Appointments"]
            .sum()
            .sort_values(ascending=True)
            .index.tolist()
        )
        chart_height = max(520, 28 * len(clinician_order) + 180)

        map_fig = px.scatter(
            map_frame,
            x=APPOINTMENT_DATE,
            y=CLINICIAN,
            color=ROTA_TYPE,
            size="Appointments",
            size_max=9,
            category_orders={CLINICIAN: clinician_order},
            hover_data={
                "Dataset": True,
                APPOINTMENT_DATE: "|%d %b %Y",
                CLINICIAN: True,
                ROTA_TYPE: True,
                "Appointments": True,
            },
            title="Appointments by Date and Clinician (Colored by Rota Type)",
        )
        map_fig.update_traces(marker=dict(opacity=0.85, line=dict(width=0)))
        map_fig.update_layout(
            height=chart_height,
            xaxis_title="Appointment Date",
            yaxis_title="Clinician",
            legend_title="Rota Type",
            margin=dict(l=10, r=10, t=60, b=20),
        )
        map_fig.update_xaxes(showgrid=True)
        map_fig.update_yaxes(showgrid=True)
        st.plotly_chart(map_fig, width="stretch")

with tab4:
    st.subheader("Completed GP, nurse, and nurse practitioner appointments for selected financial week")
    st.caption(
        "Select a financial-year week to see finished GP, nurse, and nurse practitioner appointment counts by clinician and day. "
        "The final column shows each clinician's weekly total; the final row shows daily totals."
    )

    if filtered_combined.empty:
        st.warning("No rows remain after the selected filters.")
    else:
        weekly_dataset_options = filtered_combined["Dataset"].dropna().unique().tolist()
        default_weekly_dataset_index = (
            weekly_dataset_options.index("Current FY")
            if "Current FY" in weekly_dataset_options
            else 0
        )
        selected_weekly_dataset = st.radio(
            "Heatmap dataset",
            weekly_dataset_options,
            index=default_weekly_dataset_index,
            horizontal=True,
            key="weekly_heatmap_dataset",
        )
        selected_financial_week = st.slider(
            "Financial-year week",
            min_value=1,
            max_value=52,
            value=1,
            step=1,
        )

        weekly_heatmap_frame = filtered_combined[
            filtered_combined["Dataset"] == selected_weekly_dataset
        ].copy()
        selected_week_heatmap = clinical_completed_week_heatmap(
            weekly_heatmap_frame,
            selected_financial_week,
        )

        if selected_week_heatmap.empty:
            st.info(
                "No completed GP, nurse, or nurse practitioner appointment rows remain for the selected dataset, week, and filters."
            )
        else:
            week_heatmap_fig = px.imshow(
                selected_week_heatmap,
                text_auto=True,
                aspect="auto",
                color_continuous_scale=[
                    [0, "#eef8f8"],
                    [0.35, "#bfe7e5"],
                    [0.7, "#57c7c2"],
                    [1, "#0f9c95"],
                ],
                labels=dict(x="Day of week", y="Clinician", color="Completed appointments"),
            )
            week_heatmap_fig.update_traces(
                texttemplate="%{z:,.0f}",
                hovertemplate=(
                    "Clinician: %{y}<br>"
                    "Column: %{x}<br>"
                    "Completed appointments: %{z:,.0f}<extra></extra>"
                ),
            )
            week_heatmap_fig.update_layout(
                height=max(380, 34 * len(selected_week_heatmap.index) + 190),
                xaxis_title="",
                yaxis_title="",
                coloraxis_colorbar_title="Completed",
                margin=dict(l=10, r=10, t=20, b=20),
            )
            week_heatmap_fig.update_xaxes(side="top")
            st.plotly_chart(week_heatmap_fig, width="stretch")
            st.dataframe(selected_week_heatmap, width="stretch")

with tab5:
    st.subheader("Weekly appointments compared with access target")
    st.caption(
        "The target is adjusted by list size and, when enabled, reduced for bank holidays in that week."
    )

    if weekly_capacity.empty:
        st.warning("No rows remain after the selected filters.")
    else:
        fig = px.line(
            weekly_capacity,
            x="Financial week",
            y="Appointments",
            color="Dataset",
            markers=True,
            hover_data=[
                "Week start",
                "Appointments per 1000",
                "Working days",
                "Target",
            ],
        )
        for dataset, frame in weekly_capacity.groupby("Dataset"):
            fig.add_trace(
                go.Scatter(
                    x=frame["Financial week"],
                    y=frame["Target"],
                    mode="lines",
                    name=f"{dataset} target",
                    line=dict(dash="dash"),
                )
            )
        fig.update_layout(yaxis_title="Appointments", legend_title="", hovermode="x unified")
        st.plotly_chart(fig, width="stretch")

        comparison = weekly_capacity.pivot_table(
            index="Financial week", columns="Dataset", values="Appointments", aggfunc="sum"
        ).reset_index()
        if {"Previous FY", "Current FY"}.issubset(comparison.columns):
            comparison["Current minus previous"] = comparison["Current FY"] - comparison["Previous FY"]
            st.dataframe(comparison, width="stretch", hide_index=True)

with tab6:
    if filtered_combined.empty:
        st.warning("No rows remain after the selected filters.")
    else:
        rota_weekly = (
            filtered_combined.groupby(["Dataset", "Financial week", ROTA_TYPE], as_index=False)[PATIENT_COUNT]
            .sum()
            .rename(columns={PATIENT_COUNT: "Appointments"})
        )
        clinician_weekly = (
            filtered_combined.groupby(["Dataset", "Financial week", CLINICIAN], as_index=False)[PATIENT_COUNT]
            .sum()
            .rename(columns={PATIENT_COUNT: "Appointments"})
        )

        rota_fig = px.scatter(
            rota_weekly,
            x="Financial week",
            y="Appointments",
            color=ROTA_TYPE,
            symbol="Dataset",
            size="Appointments",
            hover_data=["Dataset"],
            title="Appointments by rota type",
        )
        rota_fig.update_layout(height=560, legend_title="Rota type, Dataset")
        clinician_fig = px.scatter(
            clinician_weekly,
            x="Financial week",
            y="Appointments",
            color=CLINICIAN,
            symbol="Dataset",
            size="Appointments",
            hover_data=["Dataset"],
            title="Appointments by clinician",
        )
        clinician_fig.update_layout(height=640, legend_title="Clinician, Dataset")
        st.plotly_chart(rota_fig, width="stretch")
        st.plotly_chart(clinician_fig, width="stretch")

with tab7:
    if filtered_combined.empty:
        st.warning("No rows remain after the selected filters.")
    else:
        status_summary = (
            filtered_combined.groupby(["Dataset", APPOINTMENT_STATUS], as_index=False)[PATIENT_COUNT]
            .sum()
            .rename(columns={PATIENT_COUNT: "Appointments"})
        )
        fig = px.bar(
            status_summary,
            x=APPOINTMENT_STATUS,
            y="Appointments",
            color="Dataset",
            barmode="group",
            text_auto=True,
        )
        fig.update_layout(xaxis_title="", yaxis_title="Appointments", legend_title="")
        st.plotly_chart(fig, width="stretch")
        st.dataframe(status_summary, width="stretch", hide_index=True)

with tab8:
    st.subheader("Filtered appointment rows")
    st.dataframe(filtered_combined, width="stretch", hide_index=True)
    csv = filtered_combined.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download filtered data",
        data=csv,
        file_name="filtered_gp_access.csv",
        mime="text/csv",
    )
