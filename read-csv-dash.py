import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from dash import Dash, dcc, html, Input, Output
from datetime import timedelta
import xml.etree.ElementTree as ET
import numpy as np
import dash_bootstrap_components as dbc
import dash_daq as daq
import matplotlib.colors as mcolors
import statsmodels.api as sm
import matplotlib.pyplot as plt
import calendar
import time
from plotly.subplots import make_subplots
import os

# -------------------- DATA PROCESSING --------------------

def load_data(file_path='sleep_data.csv'):
    print("Loading data...")
    start_time = time.time()

    # Now load the CSV file
    data = pd.read_csv('sleep_data.csv')
    
    # Convert date columns to datetime 
    for col in ['creationDate', 'startDate', 'endDate']:
        data[col] = pd.to_datetime(data[col])
    
    # Clean value labels
    data["stage"] = data["value"].str.replace("HKCategoryValueSleepAnalysis", "")
    data["stage"] = data["stage"].str.replace("Asleep", "Asleep ").str.replace("([a-z])([A-Z])", r"\1 \2", regex=True)
    data["stage"] = data["stage"].str.strip().str.replace(" ", "").str.replace("Asleep", "Asleep")
    
    print(f"Data loaded in {time.time() - start_time:.2f} seconds")
    return data

def create_color_map():
    return {
        "Awake": "rgb(238,114,88)",
        "AsleepREM": "rgb(129,207,250)",
        "AsleepCore": "rgb(59,129,246)",
        "AsleepDeep": "rgb(54,52,157)",
        "AsleepUnspecified": "rgba(100,100,100,0.3)"
    }

# Set desired sleep stage order (for plotting and ordering in the Gantt).
# Order: Deep Sleep at bottom, then Core Sleep, then REM, and Awake at the top.
sleep_stage_order = ["AsleepDeep", "AsleepCore", "AsleepREM", "Awake"]

def compute_nightly_stats(data):
    print("Computing nightly statistics...")
    start_time = time.time()
    
    # Group by creationDate to get per-night statistics
    agg = data.groupby("creationDate", group_keys=False).apply(lambda df: pd.Series({
        # Calculate total sleep time including all sleep stages
        "total_asleep_mins": df[df["stage"].str.contains("Asleep")]["endDate"].sub(df["startDate"]).dt.total_seconds().sum() / 60,
        # Calculate awake time as sum of awake stage durations
        "total_awake_mins": df[df["stage"].isin(["Awake", "InBed"])]["endDate"].sub(df["startDate"]).dt.total_seconds().sum() / 60,
        # Count number of awake segments
        "wakeups": (df["stage"].isin(["Awake", "InBed"])).sum(),
        # Get bed and wake times
        "bed_time": df["startDate"].min(),
        "wake_time": df["endDate"].max(),
        # Calculate time in bed as difference between latest end time and earliest start time
        "in_bed_mins": (df["endDate"].max() - df["startDate"].min()).total_seconds() / 60,
        # Calculate individual stage durations
        "rem_sleep_mins": df[df["stage"] == "AsleepREM"]["endDate"].sub(df["startDate"]).dt.total_seconds().sum() / 60,
        "core_sleep_mins": df[df["stage"] == "AsleepCore"]["endDate"].sub(df["startDate"]).dt.total_seconds().sum() / 60,
        "deep_sleep_mins": df[df["stage"] == "AsleepDeep"]["endDate"].sub(df["startDate"]).dt.total_seconds().sum() / 60,
        "unspecified_sleep_mins": df[df["stage"] == "AsleepUnspecified"]["endDate"].sub(df["startDate"]).dt.total_seconds().sum() / 60,
        # Add day information
        "day_of_week": df["startDate"].min().dayofweek,
        "day_name": calendar.day_name[df["startDate"].min().dayofweek]
    })).reset_index()
    
    agg = agg.rename(columns={"creationDate": "group_date"})
    
    # Calculate sleep efficiency
    mask = agg["in_bed_mins"] > 0
    agg.loc[mask, "sleep_efficiency"] = agg.loc[mask, "total_asleep_mins"] / agg.loc[mask, "in_bed_mins"] * 100
    
    # Filter out nights with 0 sleep or 0 efficiency
    agg = agg[(agg["total_asleep_mins"] > 0) & (agg["sleep_efficiency"] > 0)]
    
    # Calculate percentages
    mask = agg["total_asleep_mins"] > 0
    agg.loc[mask, "rem_percentage"] = agg.loc[mask, "rem_sleep_mins"] / agg.loc[mask, "total_asleep_mins"] * 100
    agg.loc[mask, "core_percentage"] = agg.loc[mask, "core_sleep_mins"] / agg.loc[mask, "total_asleep_mins"] * 100
    agg.loc[mask, "deep_percentage"] = agg.loc[mask, "deep_sleep_mins"] / agg.loc[mask, "total_asleep_mins"] * 100
    agg.loc[mask, "unspecified_percentage"] = agg.loc[mask, "unspecified_sleep_mins"] / agg.loc[mask, "total_asleep_mins"] * 100
    
    # Add time components
    agg["bed_time_hour"] = agg["bed_time"].dt.hour + agg["bed_time"].dt.minute / 60
    agg["wake_time_hour"] = agg["wake_time"].dt.hour + agg["wake_time"].dt.minute / 60
    agg.loc[agg["bed_time_hour"] < 12, "bed_time_hour"] += 24
    
    # Add week and month information
    agg["iso_week"] = agg["group_date"].dt.isocalendar().week
    agg["iso_year"] = agg["group_date"].dt.isocalendar().year
    agg["month"] = agg["group_date"].dt.month
    agg["year"] = agg["group_date"].dt.year
    agg["year_month"] = agg["group_date"].dt.strftime("%Y-%m")
    agg["weekday"] = agg["group_date"].dt.dayofweek
    agg["is_weekend"] = agg["weekday"] >= 5
    
    # Weekly Aggregation
    weekly_agg = agg.groupby(["iso_year", "iso_week"]).agg({
        "total_asleep_mins": "mean",
        "total_awake_mins": "mean",
        "wakeups": "mean",
        "in_bed_mins": "mean",
        "rem_sleep_mins": "mean",
        "core_sleep_mins": "mean",
        "deep_sleep_mins": "mean",
        "unspecified_sleep_mins": "mean",
        "sleep_efficiency": "mean",
        "bed_time": "first",
        "wake_time": "first",
        "bed_time_hour": "mean",
        "wake_time_hour": "mean",
        "weekday": "first",
        "is_weekend": "first",
        "group_date": "first"
    }).reset_index()
    weekly_agg["time_period"] = weekly_agg.apply(lambda x: f"Week {int(x['iso_week'])}, {int(x['iso_year'])}", axis=1)
    
    # Monthly Aggregation
    monthly_agg = agg.groupby(["year", "month"]).agg({
        "total_asleep_mins": "mean",
        "total_awake_mins": "mean",
        "wakeups": "mean",
        "in_bed_mins": "mean",
        "rem_sleep_mins": "mean",
        "core_sleep_mins": "mean",
        "deep_sleep_mins": "mean",
        "unspecified_sleep_mins": "mean",
        "sleep_efficiency": "mean",
        "bed_time": "first",
        "wake_time": "first",
        "bed_time_hour": "mean",
        "wake_time_hour": "mean",
        "weekday": "first",
        "is_weekend": "first",
        "group_date": "first"
    }).reset_index()
    monthly_agg["time_period"] = monthly_agg.apply(lambda x: f"{calendar.month_name[int(x['month'])]} {int(x['year'])}", axis=1)
    
    print(f"Statistics computed in {time.time() - start_time:.2f} seconds")
    return agg, weekly_agg, monthly_agg

def preprocess_data(data, agg):
    print("Preprocessing data for visualization...")
    start_time = time.time()
    
    merged_data = data.merge(agg, left_on="creationDate", right_on="group_date")
    merged_data["duration"] = (merged_data["endDate"] - merged_data["startDate"]).dt.total_seconds() / 60
    merged_data["start_hour"] = merged_data["startDate"].dt.hour + merged_data["startDate"].dt.minute / 60
    merged_data["start_time"] = merged_data["startDate"].dt.strftime("%H:%M:%S")
    merged_data["end_time"] = merged_data["endDate"].dt.strftime("%H:%M:%S")
    merged_data["formatted_duration"] = merged_data.apply(
        lambda x: f"{int(x['duration'] // 60)}h {int(x['duration'] % 60)}m", axis=1
    )
    
    # Filter out rows with "InBed" for visualization purposes
    gantt_data = merged_data[~merged_data["stage"].isin(["InBed"])].copy()
    gantt_data["group_date"] = gantt_data["creationDate"].dt.normalize()
    gantt_data["hover_text"] = gantt_data.apply(
        lambda x: (
            f"<b>{x['stage']}</b><br>Start: {x['start_time']}<br>End: {x['end_time']}"
            f"<br>Duration: {x['formatted_duration']}<br>Occurrences: {x.get('stage_count', 'N/A')}"
        ), axis=1
    )
    stage_counts = gantt_data.groupby(["group_date", "stage"]).size().reset_index(name="stage_count")
    gantt_data = gantt_data.merge(stage_counts, on=["group_date", "stage"])
    
    print(f"Data preprocessing completed in {time.time() - start_time:.2f} seconds")
    return merged_data, gantt_data

def lowess_smooth(x, y, frac=0.15):
    if len(x) < 3:
        return y
    return sm.nonparametric.lowess(y, x, frac=frac, return_sorted=False)

def moving_average(data, window=7):
    return data.rolling(window=window, min_periods=1).mean()

# -------------------- HELPER FUNCTIONS --------------------

def apply_week_filter(df, week_filter):
    if week_filter == "weekday":
        return df[df["weekday"] < 5] if "weekday" in df.columns else df[~df["is_weekend"]]
    elif week_filter == "weekend":
        return df[df["weekday"] >= 5] if "weekday" in df.columns else df[df["is_weekend"]]
    return df

def get_time_period_df(period, agg, weekly_agg, monthly_agg):
    if period == 'weekly':
        return weekly_agg.copy()
    elif period == 'monthly':
        return monthly_agg.copy()
    return agg.copy()

def build_gantt_stats(selected_date, gantt_df, agg_df):
    # Filter gantt data for the selected night
    gd = gantt_df[gantt_df["creationDate"].dt.date.astype(str) == selected_date]
    if gd.empty:
        return html.Div("No Gantt statistics available for this night.")
    
    # Get the corresponding row from agg_df for this night
    night_stats = agg_df[agg_df["group_date"].dt.date.astype(str) == selected_date]
    if night_stats.empty:
        return html.Div("No statistics available for this night.")
    
    ns = night_stats.iloc[0]
    
    # Create stage summary table
    stage_stats = gd.groupby("stage").agg({
        "duration": "sum",
        "stage_count": "max"
    }).reset_index()
    
    # Calculate total in bed duration for percentages
    total_duration = stage_stats["duration"].sum()
    stage_stats["percentage"] = stage_stats["duration"].apply(lambda x: (x / total_duration * 100) if total_duration > 0 else 0)
    
    # Define stage order and filter/sort stages
    stage_order = {"Awake": 0, "AsleepREM": 1, "AsleepCore": 2, "AsleepDeep": 3, "AsleepUnspecified": 4}
    stage_stats["order"] = stage_stats["stage"].map(stage_order)
    stage_stats = stage_stats.sort_values("order")
    
    # Create stage summary table
    stage_table = html.Table([
        html.Tr([html.Th("Stage"), html.Th("Count"), html.Th("Duration"), html.Th("Percentage (%)")], style={"textAlign": "left"})] +
        [html.Tr([
            html.Td(row["stage"].replace("Asleep", "")),
            html.Td(f"{row['stage_count']:.0f}"),
            html.Td(format_duration(row['duration'])),
            html.Td(f"{row['percentage']:.1f}")
        ], style={"textAlign": "left"}) for _, row in stage_stats.iterrows()],
        className="stats-table w-100"
    )
    
    # Create overall night metrics table
    overall_table = html.Table([
        html.Tr([html.Th("Bed Time"), html.Td(format_time(ns["bed_time"]))]),
        html.Tr([html.Th("Wake Time"), html.Td(format_time(ns["wake_time"]))]),
        html.Tr([html.Th("Sleep Duration"), html.Td(format_duration(ns["total_asleep_mins"]))]),
        html.Tr([html.Th("Time in Bed"), html.Td(format_duration(ns["in_bed_mins"]))]),
        html.Tr([html.Th("Sleep Efficiency (%)"), html.Td(f"{ns['sleep_efficiency']:.1f}")])
    ], className="stats-table w-100")
    
    return dbc.Card([
        dbc.CardHeader("Sleep Stage Analysis"),
        dbc.CardBody(stage_table),
        dbc.CardHeader("Overall Night Metrics"),
        dbc.CardBody(overall_table)
    ], className="mb-3", style={"background-color": "#2a2a2a", "border-color": "#444"})

def build_night_selector(agg_df):
    req = {"total_asleep_mins", "total_awake_mins", "sleep_efficiency", "wakeups", "group_date"}
    if not req.issubset(agg_df.columns):
        print("ERROR: Aggregated data missing required columns!")
        return []
    
    # Find nights with the most and least sleep
    night_longest_sleep = agg_df.loc[agg_df["total_asleep_mins"].idxmax(), "group_date"]
    night_shortest_sleep = agg_df.loc[agg_df["total_asleep_mins"].idxmin(), "group_date"]
    
    # Find nights with the most and least awake time
    night_longest_awake = agg_df.loc[agg_df["total_awake_mins"].idxmax(), "group_date"]
    night_least_awake = agg_df.loc[agg_df["total_awake_mins"].idxmin(), "group_date"]
    
    # Find nights with best and worst efficiency
    night_best_eff = agg_df.loc[agg_df["sleep_efficiency"].idxmax(), "group_date"]
    night_worst_eff = agg_df.loc[agg_df["sleep_efficiency"].idxmin(), "group_date"]
    
    # Find nights with most and least wakeups
    night_most_awake = agg_df.loc[agg_df["wakeups"].idxmax(), "group_date"]
    night_least_awake = agg_df.loc[agg_df["wakeups"].idxmin(), "group_date"]
    
    # Debug print to verify selections
    print("\nNight Selection Debug:")
    print(f"Longest Sleep: {night_longest_sleep.date()} - {agg_df.loc[agg_df['group_date'] == night_longest_sleep, 'total_asleep_mins'].iloc[0]/60:.1f}h")
    print(f"Shortest Sleep: {night_shortest_sleep.date()} - {agg_df.loc[agg_df['group_date'] == night_shortest_sleep, 'total_asleep_mins'].iloc[0]/60:.1f}h")
    print(f"Best Efficiency: {night_best_eff.date()} - {agg_df.loc[agg_df['group_date'] == night_best_eff, 'sleep_efficiency'].iloc[0]:.1f}%")
    print(f"Worst Efficiency: {night_worst_eff.date()} - {agg_df.loc[agg_df['group_date'] == night_worst_eff, 'sleep_efficiency'].iloc[0]:.1f}%")
    
    options = [
        {"label": f"Night: Longest Sleep ({night_longest_sleep.date()})", "value": str(night_longest_sleep.date())},
        {"label": f"Night: Shortest Sleep ({night_shortest_sleep.date()})", "value": str(night_shortest_sleep.date())},
        {"label": f"Night: Longest Awake ({night_longest_awake.date()})", "value": str(night_longest_awake.date())},
        {"label": f"Night: Best Efficiency ({night_best_eff.date()})", "value": str(night_best_eff.date())},
        {"label": f"Night: Worst Efficiency ({night_worst_eff.date()})", "value": str(night_worst_eff.date())},
        {"label": f"Night: Most Awake Segments ({night_most_awake.date()})", "value": str(night_most_awake.date())},
        {"label": f"Night: Least Awake Segments ({night_least_awake.date()})", "value": str(night_least_awake.date())},
        {"label": "Debug Night (2024-10-04)", "value": "2024-10-04"}
    ]
    return options

# -------------------- STATISTICS HELPERS --------------------

def compute_summary_stats(df, metric):
    stats = {}
    col = df[metric]
    stats['Mean'] = f"{col.mean():.2f}"
    
    # Handle median with zero filter
    nonzero = col[col != 0]
    if nonzero.size > 0:
        stats['Median'] = f"{nonzero.median():.2f}*"
    else:
        stats['Median'] = "0.00"
    
    stats['Std Dev'] = f"{col.std():.2f}"
    all_min = col.min()
    if nonzero.size > 0:
        nz_min = nonzero.min()
        if all_min == 0 and nz_min != 0:
            stats['Min'] = f"{nz_min:.2f}*"
        else:
            stats['Min'] = f"{all_min:.2f}"
    else:
        stats['Min'] = "0.00"
    
    # Group values to nearest 5 for mode calculation, excluding zeros
    if 'mins' in metric or 'percentage' in metric:
        grouped_col = (nonzero / 5).round() * 5
    else:
        grouped_col = nonzero
    
    # Handle mode with zero filter
    mode_series = grouped_col.mode()
    if not mode_series.empty:
        stats['Mode'] = f"{mode_series.iloc[0]:.2f}*"
    else:
        stats['Mode'] = "N/A"
    
    note = "* indicates zero values were excluded; lowest nonzero value is shown."
    return stats, note

def format_duration(minutes):
    if minutes >= 60:
        hours = int(minutes // 60)
        mins = int(minutes % 60)
        return f"{hours}h {mins}m"
    return f"{int(minutes)}m"

def format_time(dt):
    return dt.strftime("%I:%M %p")

def make_summary_stats(df, metric, title):
    if df.empty or metric not in df.columns:
        return html.Div("No data available")
    stats, note = compute_summary_stats(df, metric)
    table = html.Table([
        html.Tr([html.Td("Mean"), html.Td(stats['Mean'])]),
        html.Tr([html.Td("Median"), html.Td(stats['Median'])]),
        html.Tr([html.Td("Min"), html.Td(stats['Min'])]),
        html.Tr([html.Td("Max"), html.Td(f"{df[metric].max():.2f}")]),
        html.Tr([html.Td("Std Dev"), html.Td(stats['Std Dev'])]),
        html.Tr([html.Td("Mode"), html.Td(stats['Mode'])])
    ], className="stats-table w-100")
    children = [dbc.CardHeader(title), dbc.CardBody([table])]
    if note:
        children.append(html.Div(note, style={"fontSize": "0.8em", "color": "red", "marginTop": "10px", "padding": "10px"}))
    return dbc.Card(children, className="mb-3", style={"background-color": "#2a2a2a", "border-color": "#444"})

def make_timing_stats(df, time_range="Overall"):
    if df.empty:
        return html.Div("No timing statistics available")
    
    bed_time_mean = df["bed_time_hour"].mean()
    bed_time_std = df["bed_time_hour"].std()
    wake_time_mean = df["wake_time_hour"].mean()
    wake_time_std = df["wake_time_hour"].std()
    
    def format_hour(hour):
        hour = hour % 24
        return f"{int(hour):02d}:{int((hour*60)%60):02d} {'AM' if hour < 12 else 'PM'}"
    
    stats_table = html.Table([
        html.Tr([html.Th("Metric"), html.Th("Mean"), html.Th("Std Dev")]),
        html.Tr([
            html.Td("Bed Time"),
            html.Td(format_hour(bed_time_mean)),
            html.Td(f"±{bed_time_std:.2f}h")
        ]),
        html.Tr([
            html.Td("Wake Time"),
            html.Td(format_hour(wake_time_mean)),
            html.Td(f"±{wake_time_std:.2f}h")
        ])
    ], className="stats-table w-100")
    
    return dbc.Card([
        dbc.CardHeader(f"Sleep Timing Statistics ({time_range})"),
        dbc.CardBody(stats_table)
    ], className="mb-3", style={"background-color": "#2a2a2a", "border-color": "#444"})

# -------------------- VISUALIZATION COMPONENTS --------------------

def create_correlation_matrix(df):
    if df.empty:
        fig = go.Figure()
        fig.update_layout(title="No data available",
                          plot_bgcolor="#1a1a1a",
                          paper_bgcolor="#1a1a1a",
                          font=dict(family="-apple-system", color="white"))
        return fig
    
    metrics = ['total_asleep_mins', 'rem_sleep_mins', 'deep_sleep_mins', 'sleep_efficiency', 'wakeups']
    available_cols = [m for m in metrics if m in df.columns]
    if len(available_cols) < 2:
        fig = go.Figure()
        fig.update_layout(title="Not enough numeric data for correlation analysis")
        return fig
        
    fig = px.scatter_matrix(
        df, dimensions=available_cols, color="is_weekend",
        color_discrete_map={True: "orange", False: "skyblue"},
        labels={m: metric_options.get(m, m) for m in available_cols},
        opacity=0.7
    )
    fig.update_layout(
        title="Sleep Metrics Correlation Matrix",
        height=500,
        plot_bgcolor="#1a1a1a",
        paper_bgcolor="#1a1a1a",
        font=dict(family="-apple-system", color="white"),
        margin=dict(l=40, r=20, t=60, b=40)
    )
    fig.update_traces(
        diagonal_visible=False,
        showupperhalf=False,
        marker=dict(size=8, line=dict(width=1, color="white"))
    )
    return fig

def create_timeseries(df, metric, time_period="daily", moving_avg_window=7):
    if df.empty or metric is None or (metric not in df.columns):
        fig = go.Figure()
        fig.update_layout(
            title="Select a metric to view the time series",
            plot_bgcolor="#1a1a1a",
            paper_bgcolor="#1a1a1a",
            font=dict(family="-apple-system", color="white")
        )
        return fig

    plot_df = df[df[metric] > 0].sort_values("group_date").copy()
    if plot_df.empty:
        fig = go.Figure()
        fig.update_layout(
            title="No nonzero data available",
            plot_bgcolor="#1a1a1a",
            paper_bgcolor="#1a1a1a",
            font=dict(family="-apple-system", color="white")
        )
        return fig
    
    x = plot_df["group_date"]
    y = plot_df[metric]
    ma = moving_average(y, window=moving_avg_window) if len(y) >= moving_avg_window else y
    smoothed = lowess_smooth(np.arange(len(y)), y) if len(y) >= 5 else y

    # Format y-axis values based on metric type
    def format_y_value(val):
        if 'percentage' in metric:
            return f"{val:.1f}%"
        elif 'mins' in metric:
            hours = int(val // 60)
            mins = int(val % 60)
            if hours == 0:
                return f"{mins}m"
            return f"{hours}h {mins}m"
        return f"{val:.1f}"
    
    # Create plasma color scale
    plasma_colors = px.colors.sequential.Plasma
    y_min, y_max = y.min(), y.max()
    y_range = y_max - y_min
    color_scale = []
    for val in y:
        if y_range > 0:
            color_idx = int((val - y_min) / y_range * (len(plasma_colors) - 1))
            color_idx = max(0, min(color_idx, len(plasma_colors) - 1))
            color_scale.append(plasma_colors[color_idx])
        else:
            color_scale.append(plasma_colors[0])
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=y,
        mode="markers",
        marker=dict(
            size=10,
            color=color_scale,
            colorscale="Plasma"
        ),
        name="Data Points",
        hovertemplate="%{x}<br>%{text}<extra></extra>",
        text=[format_y_value(val) for val in y]
    ))
    fig.add_trace(go.Scatter(
        x=x, y=ma,
        mode="lines",
        line=dict(color="rgba(255,255,255,0.7)", width=2),
        name=f"{moving_avg_window}-Day Moving Average",
        hovertemplate="%{x}<br>%{text}<extra></extra>",
        text=[format_y_value(val) for val in ma]
    ))
    fig.add_trace(go.Scatter(
        x=x, y=smoothed,
        mode="lines",
        line=dict(color="rgba(255,165,0,0.8)", width=2, dash="dot"),
        name="Trend",
        hovertemplate="%{x}<br>%{text}<extra></extra>",
        text=[format_y_value(val) for val in smoothed]
    ))
    
    # Format y-axis ticks
    def get_y_ticks():
        if 'mins' in metric:
            return np.arange(0, y.max() + 60, 60)  # Every hour
        elif 'percentage' in metric:
            return np.arange(0, 101, 20)  # Every 20%
        return None
    
    fig.update_layout(
        title=f"{metric_options.get(metric, metric)} Over Time",
        plot_bgcolor="#1a1a1a",
        paper_bgcolor="#1a1a1a",
        font=dict(family="-apple-system", color="white"),
        hovermode="closest",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=20, t=60, b=40),
        xaxis_title="Date",
        yaxis_title=metric_options.get(metric, metric),
        xaxis=dict(
            showgrid=True,
            gridcolor="rgba(255,255,255,0.1)",
            showline=True,
            linecolor="rgba(255,255,255,0.1)"
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="rgba(255,255,255,0.1)",
            showline=True,
            linecolor="rgba(255,255,255,0.1)",
            range=[0, 100] if 'percentage' in metric else None,
            tickvals=get_y_ticks(),
            ticktext=[format_y_value(tick) for tick in get_y_ticks()]
        ),
        transition_duration=500
    )
    
    return fig

def create_sleep_timing_chart(df):
    if df.empty:
        fig = go.Figure()
        fig.update_layout(
            title="No data available",
            plot_bgcolor="#1a1a1a",
            paper_bgcolor="#1a1a1a",
            font=dict(family="-apple-system", color="white")
        )
        return fig
    
    plot_df = df[(df['bed_time_hour'].notna()) & (df['wake_time_hour'].notna())].sort_values("group_date").copy()
    if plot_df.empty:
        fig = go.Figure()
        fig.update_layout(title="No sleep timing data available")
        return fig
    
    # Normalize hours to be between -6 (6pm previous day) and 18 (6pm current day)
    def normalize_hour(hour):
        if hour >= 18:  # After 6pm
            return hour - 24
        return hour
    
    plot_df["normalized_wake"] = plot_df["wake_time_hour"].apply(normalize_hour)
    plot_df["normalized_bed"] = plot_df["bed_time_hour"].apply(normalize_hour)
    
    # Create color scales for bed and wake times
    bed_colors = px.colors.sequential.Viridis
    wake_colors = px.colors.sequential.Inferno
    
    bed_min, bed_max = plot_df["normalized_bed"].min(), plot_df["normalized_bed"].max()
    wake_min, wake_max = plot_df["normalized_wake"].min(), plot_df["normalized_wake"].max()
    
    bed_range = bed_max - bed_min
    wake_range = wake_max - wake_min
    
    bed_color_scale = []
    wake_color_scale = []
    
    for bed_val, wake_val in zip(plot_df["normalized_bed"], plot_df["normalized_wake"]):
        if bed_range > 0:
            bed_idx = int((bed_val - bed_min) / bed_range * (len(bed_colors) - 1))
            bed_idx = max(0, min(bed_idx, len(bed_colors) - 1))
            bed_color_scale.append(bed_colors[bed_idx])
        else:
            bed_color_scale.append(bed_colors[0])
            
        if wake_range > 0:
            wake_idx = int((wake_val - wake_min) / wake_range * (len(wake_colors) - 1))
            wake_idx = max(0, min(wake_idx, len(wake_colors) - 1))
            wake_color_scale.append(wake_colors[wake_idx])
        else:
            wake_color_scale.append(wake_colors[0])
    
    fig = go.Figure()
    
    # Add bed time points and trend
    fig.add_trace(go.Scatter(
        x=plot_df["group_date"],
        y=plot_df["normalized_bed"],
        mode="markers",
        marker=dict(
            size=10,
            color=bed_color_scale,
            colorscale="Viridis"
        ),
        name="Bed Time",
        hovertemplate="%{x}<br>%{text}<extra></extra>",
        text=[f"{int(h if h >= 0 else h + 24):02d}:{int((h*60)%60):02d} {'AM' if h < 12 else 'PM'}" for h in plot_df["normalized_bed"]]
    ))
    
    if len(plot_df) >= 5:
        bed_smoothed = lowess_smooth(np.arange(len(plot_df)), plot_df["normalized_bed"].values)
        fig.add_trace(go.Scatter(
            x=plot_df["group_date"],
            y=bed_smoothed,
            mode="lines",
            line=dict(color="rgba(255,165,0,0.7)", width=2),
            name="Bed Time Trend"
        ))
    
    # Add wake time points and trend
    fig.add_trace(go.Scatter(
        x=plot_df["group_date"],
        y=plot_df["normalized_wake"],
        mode="markers",
        marker=dict(
            size=10,
            color=wake_color_scale,
            colorscale="Inferno"
        ),
        name="Wake Time",
        hovertemplate="%{x}<br>%{text}<extra></extra>",
        text=[f"{int(h if h >= 0 else h + 24):02d}:{int((h*60)%60):02d} {'AM' if h < 12 else 'PM'}" for h in plot_df["normalized_wake"]]
    ))
    
    if len(plot_df) >= 5:
        wake_smoothed = lowess_smooth(np.arange(len(plot_df)), plot_df["normalized_wake"].values)
        fig.add_trace(go.Scatter(
            x=plot_df["group_date"],
            y=wake_smoothed,
            mode="lines",
            line=dict(color="rgba(255,255,255,0.7)", width=2),
            name="Wake Time Trend"
        ))
    
    # Create hour ticks every 4 hours from 6pm to 6pm
    hour_ticks = list(range(-6, 19, 4))
    hour_labels = []
    for h in hour_ticks:
        hour = (h + 24) % 24
        ampm = "AM" if hour < 12 else "PM"
        if hour == 0:
            hour = 12
        elif hour > 12:
            hour -= 12
        hour_labels.append(f"{hour:02d}:00 {ampm}")
    
    fig.update_layout(
        height=400,
        plot_bgcolor="#1a1a1a",
        paper_bgcolor="#1a1a1a",
        font=dict(family="-apple-system", color="white"),
        hovermode="closest",
        margin=dict(l=40, r=20, t=80, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        yaxis=dict(
            title="Time of Day",
            tickvals=hour_ticks,
            ticktext=hour_labels,
            gridcolor="rgba(255,255,255,0.1)",
            showline=True,
            linecolor="rgba(255,255,255,0.1)",
            range=[-6, 18]  # Set y-axis range from 6pm to 6pm
        ),
        xaxis=dict(
            showgrid=True,
            gridcolor="rgba(255,255,255,0.1)",
            showline=True,
            linecolor="rgba(255,255,255,0.1)"
        ),
        transition_duration=500
    )
    return fig

def create_enhanced_gantt_one_night(df):
    if df.empty:
        fig = go.Figure()
        fig.update_layout(
            title="No Gantt data available",
            plot_bgcolor="#1a1a1a",
            paper_bgcolor="#1a1a1a",
            font=dict(color="white")
        )
        return fig

    # Check if we only have unspecified sleep, in-bed time, and awake
    unique_stages = set(df["stage"].unique())
    is_simple_night = unique_stages.issubset({"AsleepUnspecified", "InBed", "Awake"})
    
    if is_simple_night:
        # Special case for nights with only unspecified sleep, in-bed time, and awake
        stage_order = {"Asleep": 0, "Awake": 1}
        fig = go.Figure()
        
        # Create a list of all time points for the x-axis
        all_times = []
        for _, row in df.iterrows():
            all_times.extend([row["startDate"], row["endDate"]])
        all_times = sorted(set(all_times))
        
        # Generate x-axis ticks every 30 minutes
        if all_times:
            start_time = all_times[0].replace(minute=0, second=0, microsecond=0)
            end_time = all_times[-1].replace(minute=0, second=0, microsecond=0) + pd.Timedelta(hours=1)
            x_ticks = pd.date_range(start=start_time, end=end_time, freq='30min')
            x_ticktext = [t.strftime("%I:%M %p") for t in x_ticks]
        
        # Plot sleep segments
        for _, row in df[df["stage"] == "AsleepUnspecified"].iterrows():
            dur = (row["endDate"] - row["startDate"]).total_seconds() / 60
            same_stage_count = row.get("stage_count", 1)
            hover_text = (
                f"Stage: Asleep<br>"
                f"Duration: {format_duration(dur)}<br>"
                f"Start: {format_time(row['startDate'])}<br>"
                f"End: {format_time(row['endDate'])}<br>"
                f"Occurrences: {same_stage_count}<br>"
                "<extra></extra>"
            )
            y_base = stage_order["Asleep"]
            Y_BOTTOM = y_base + 0.1
            Y_TOP = y_base + 0.6  # Make the box 1.5x taller than normal (normal is 0.4)
            x_coords = [row["startDate"], row["endDate"], row["endDate"], row["startDate"], row["startDate"]]
            y_coords = [Y_BOTTOM, Y_BOTTOM, Y_TOP, Y_TOP, Y_BOTTOM]
            
            fig.add_trace(go.Scatter(
                x=x_coords,
                y=y_coords,
                fill="toself",
                fillcolor="rgb(99,230,225)",  # Brighter color for sleep
                mode="lines",
                line=dict(color="rgba(0,0,0,0)"),
                hovertemplate=hover_text,
                showlegend=False
            ))
        
        # Plot in-bed/awake segments
        for _, row in df[df["stage"].isin(["InBed", "Awake"])].iterrows():
            dur = (row["endDate"] - row["startDate"]).total_seconds() / 60
            same_stage_count = row.get("stage_count", 1)
            hover_text = (
                f"Stage: Awake<br>"
                f"Duration: {format_duration(dur)}<br>"
                f"Start: {format_time(row['startDate'])}<br>"
                f"End: {format_time(row['endDate'])}<br>"
                f"Occurrences: {same_stage_count}<br>"
                "<extra></extra>"
            )
            y_base = stage_order["Awake"]
            Y_BOTTOM = y_base + 0.1
            Y_TOP = y_base + 0.9
            x_coords = [row["startDate"], row["endDate"], row["endDate"], row["startDate"], row["startDate"]]
            y_coords = [Y_BOTTOM, Y_BOTTOM, Y_TOP, Y_TOP, Y_BOTTOM]
            
            fig.add_trace(go.Scatter(
                x=x_coords,
                y=y_coords,
                fill="toself",
                fillcolor="rgb(0,92,91)",  # Darker color for awake
                mode="lines",
                line=dict(color="rgba(0,0,0,0)"),
                hovertemplate=hover_text,
                showlegend=False
            ))
        
        # Add horizontal lines
        for i in range(3):  # Add one more line for the bottom
            fig.add_hline(y=i, line_dash="solid", line_color="rgba(255,255,255,0.2)", line_width=1)
        
        tick_vals = [val + 0.5 for key, val in sorted(stage_order.items(), key=lambda x: x[1])]
        tick_text = [key for key, val in sorted(stage_order.items(), key=lambda x: x[1])]
        
        fig.update_layout(
            title=f"Gantt for Night: {df['creationDate'].iloc[0].date()}",
            plot_bgcolor="#1a1a1a",
            paper_bgcolor="#1a1a1a",
            font=dict(family="-apple-system", color="white"),
            hovermode="x",
            spikedistance=1000,
            margin=dict(l=40, r=20, t=60, b=40),
            yaxis=dict(
                title="Sleep Stage",
                tickmode="array",
                tickvals=tick_vals,
                ticktext=tick_text,
                range=[-0.2, 2.0],
                showgrid=False
            )
        )
        fig.update_xaxes(
            showspikes=True,
            spikecolor="white",
            spikesnap="cursor",
            spikemode="across",
            spikethickness=1,
            spikedash="dot",
            title="Time",
            gridcolor="rgba(255,255,255,0.1)",
            gridwidth=1,
            griddash="dot",
            tickvals=x_ticks,
            ticktext=x_ticktext
        )
        return fig
    
    # Regular case with all sleep stages
    stage_order = {"AsleepDeep": 0, "AsleepCore": 1, "AsleepREM": 2, "Awake": 3}
    fig = go.Figure()
    
    # Create a list of all time points for the x-axis
    all_times = []
    for _, row in df.iterrows():
        all_times.extend([row["startDate"], row["endDate"]])
    all_times = sorted(set(all_times))
    
    # Generate x-axis ticks every 30 minutes
    if all_times:
        start_time = all_times[0].replace(minute=0, second=0, microsecond=0)
        end_time = all_times[-1].replace(minute=0, second=0, microsecond=0) + pd.Timedelta(hours=1)
        x_ticks = pd.date_range(start=start_time, end=end_time, freq='30min')
        x_ticktext = [t.strftime("%I:%M %p") for t in x_ticks]
    
    # Define stage colors
    stage_color_map = {
        "Awake": "rgba(238,114,88, 0.8)",
        "AsleepCore": "rgba(59,129,246,0.8)",
        "AsleepREM": "rgba(129,207,250,0.8)",
        "AsleepDeep": "rgba(54,52,157,0.8)",
        "AsleepUnspecified": "rgba(100,100,100,0.3)"
    }
    
    # Create gradient colors for unspecified sleep
    colors = [
        (54/255, 52/255, 157/255, 0.8),  # Deep
        (59/255, 129/255, 246/255, 0.8),  # Core
        (129/255, 207/255, 250/255, 0.8), # REM
        (54/255, 52/255, 157/255, 0.8)    # Back to Deep
    ]
    gradient_colormap = mcolors.LinearSegmentedColormap.from_list("", colors)
    
    # First, plot all regular sleep stages
    for _, row in df.iterrows():
        stage = row["stage"]
        if stage == "AsleepUnspecified":
            continue
            
        y_base = stage_order.get(stage, 4)
        Y_BOTTOM = y_base + 0.1  # Move boxes up slightly
        Y_TOP = y_base + 0.9     # Keep same height
        dur = (row["endDate"] - row["startDate"]).total_seconds() / 60
        same_stage_count = row.get("stage_count", 1)
        hover_text = (
            f"Stage: {stage}<br>"
            f"Duration: {format_duration(dur)}<br>"
            f"Start: {format_time(row['startDate'])}<br>"
            f"End: {format_time(row['endDate'])}<br>"
            f"Occurrences: {same_stage_count}<br>"
            "<extra></extra>"
        )
        x_coords = [row["startDate"], row["endDate"], row["endDate"], row["startDate"], row["startDate"]]
        y_coords = [Y_BOTTOM, Y_BOTTOM, Y_TOP, Y_TOP, Y_BOTTOM]
        
        color = stage_color_map.get(stage, "gray")
        fig.add_trace(go.Scatter(
            x=x_coords,
            y=y_coords,
            fill="toself",
            fillcolor=color,
            mode="lines",
            line=dict(color="rgba(0,0,0,0)"),
            hovertemplate=hover_text,
            showlegend=False
        ))
    
    # Then, plot unspecified sleep segments spanning all sleep stages
    for _, row in df[df["stage"] == "AsleepUnspecified"].iterrows():
        dur = (row["endDate"] - row["startDate"]).total_seconds() / 60
        same_stage_count = row.get("stage_count", 1)
        hover_text = (
            f"Stage: {row['stage']}<br>"
            f"Duration: {format_duration(dur)}<br>"
            f"Start: {format_time(row['startDate'])}<br>"
            f"End: {format_time(row['endDate'])}<br>"
            f"Occurrences: {same_stage_count}<br>"
            "<extra></extra>"
        )
        
        # Create vertical gradient for unspecified sleep
        num_points = 100
        y_gradient = np.linspace(0, 3, num_points)  # Vertical gradient
        x_coords = [row["startDate"], row["endDate"], row["endDate"], row["startDate"], row["startDate"]]
        
        for i in range(num_points-1):
            gradient_color = mcolors.to_hex(gradient_colormap(y_gradient[i]))
            y_bottom = y_gradient[i]
            y_top = y_gradient[i+1]
            y_coords = [y_bottom, y_bottom, y_top, y_top, y_bottom]
            
            fig.add_trace(go.Scatter(
                x=x_coords,
                y=y_coords,
                fill="toself",
                fillcolor=gradient_color,
                mode="lines",
                line=dict(color="rgba(0,0,0,0)"),
                hovertemplate=hover_text,
                showlegend=False
            ))
    
    # Add horizontal lines between stages
    for i in range(5):  # Add one more line for the bottom
        fig.add_hline(y=i, line_dash="solid", line_color="rgba(255,255,255,0.2)", line_width=1)
    
    tick_vals = [val + 0.5 for key, val in sorted(stage_order.items(), key=lambda x: x[1])]  # Center ticks
    tick_text = [key.replace("Asleep", "") for key, val in sorted(stage_order.items(), key=lambda x: x[1])]
    
    fig.update_layout(
        title=f"Gantt for Night: {df['creationDate'].iloc[0].date()}",
        plot_bgcolor="#1a1a1a",
        paper_bgcolor="#1a1a1a",
        font=dict(family="-apple-system", color="white"),
        hovermode="x",
        spikedistance=1000,
        margin=dict(l=40, r=20, t=60, b=40),
        yaxis=dict(
            title="Sleep Stage",
            tickmode="array",
            tickvals=tick_vals,
            ticktext=tick_text,
            range=[-0.2, 4.0],
            showgrid=False
        )
    )
    fig.update_xaxes(
        showspikes=True,
        spikecolor="white",
        spikesnap="cursor",
        spikemode="across",
        spikethickness=1,
        spikedash="dot",
        title="Time",
        gridcolor="rgba(255,255,255,0.1)",
        gridwidth=1,
        griddash="dot",
        tickvals=x_ticks,
        ticktext=x_ticktext
    )
    return fig

# -------------------- DASH APP --------------------

metric_options = {
    "total_asleep_mins": "Sleep Duration (mins)",
    "total_awake_mins": "Awake Duration (mins)",
    "rem_cycles": "REM Cycles",
    "wakeups": "Wakeups",
    "in_bed_mins": "Time in Bed (mins)",
    "sleep_efficiency": "Sleep Efficiency (%)",
    "rem_sleep_mins": "REM Sleep (mins)",
    "core_sleep_mins": "Core Sleep (mins)",
    "deep_sleep_mins": "Deep Sleep (mins)",
    "rem_percentage": "REM Sleep (%)",
    "core_percentage": "Core Sleep (%)",
    "deep_percentage": "Deep Sleep (%)"
}

time_period_options = [
    {"label": "Daily", "value": "daily"},
    {"label": "Weekly", "value": "weekly"},
    {"label": "Monthly", "value": "monthly"}
]

def filter_df_by_time_range(df, relayout_data):
    if not relayout_data or not any(key.startswith('xaxis.range') for key in relayout_data):
        return df
    
    start_date = None
    end_date = None
    
    if 'xaxis.range[0]' in relayout_data:
        start_date = pd.to_datetime(relayout_data['xaxis.range[0]']).tz_localize(None)
        end_date = pd.to_datetime(relayout_data['xaxis.range[1]']).tz_localize(None)
    elif 'xaxis.range' in relayout_data:
        start_date = pd.to_datetime(relayout_data['xaxis.range'][0]).tz_localize(None)
        end_date = pd.to_datetime(relayout_data['xaxis.range'][1]).tz_localize(None)
    
    if start_date and end_date:
        # Convert group_date to timezone-naive for comparison
        df = df.copy()
        df['group_date'] = df['group_date'].dt.tz_localize(None)
        return df[(df['group_date'] >= start_date) & (df['group_date'] <= end_date)]
    return df

def get_time_range_title(relayout_data):
    if not relayout_data or not any(key.startswith('xaxis.range') for key in relayout_data):
        return "Overall"
    
    start_date = None
    end_date = None
    
    if 'xaxis.range[0]' in relayout_data:
        start_date = pd.to_datetime(relayout_data['xaxis.range[0]'])
        end_date = pd.to_datetime(relayout_data['xaxis.range[1]'])
    elif 'xaxis.range' in relayout_data:
        start_date = pd.to_datetime(relayout_data['xaxis.range'][0])
        end_date = pd.to_datetime(relayout_data['xaxis.range'][1])
    
    if start_date and end_date:
        return f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
    return "Overall"

def run_dashboard():
    start_time = time.time()
    data = load_data()
    base_color_map = create_color_map()
    agg, weekly_agg, monthly_agg = compute_nightly_stats(data)
    merged_data, gantt_data = preprocess_data(data, agg)
    print(f"Total data preparation time: {time.time() - start_time:.2f} seconds")
    
    # Build special night selector options from the aggregated data
    night_dropdown_options = build_night_selector(agg)
    
    app = Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
    app.title = "Enhanced Sleep Analysis"
    
    app.layout = dbc.Container(
        id="page-content",
        fluid=True,
        style={"backgroundColor": "#121212"},
        children=[
            html.Div(style={"paddingTop": "30px"}),
            dbc.Row([
                dbc.Col(html.H2("Enhanced Sleep Analysis Dashboard", style={
                    "fontFamily": "Helvetica, sans-serif",
                    "marginBottom": "20px",
                    "textAlign": "center"
                }), width=12)
            ]),
            dbc.Row([
                dbc.Col(daq.ToggleSwitch(
                    id="dark-mode-toggle",
                    value=True,
                    label="Dark Mode",
                    labelPosition="bottom",
                    color="#5c98d6"
                ), width=2, className="offset-10", style={"textAlign": "right", "marginBottom": "20px"})
            ]),
            dbc.Row([
                dbc.Col([
                    html.Label("Time Period", style={"color": "inherit", "marginBottom": "5px"}),
                    dcc.RadioItems(
                        id='time-period-select',
                        options=time_period_options,
                        value='daily',
                        className="mb-3",
                        inputStyle={"marginRight": "5px"},
                        labelStyle={"marginRight": "15px", "display": "inline-block"}
                    )
                ], width=4),
                dbc.Col([
                    html.Label("Day Filter", style={"color": "inherit", "marginBottom": "5px"}),
                    dcc.RadioItems(
                        id='week-filter',
                        options=[
                            {"label": "All Days", "value": "all"},
                            {"label": "Weekdays", "value": "weekday"},
                            {"label": "Weekends", "value": "weekend"}
                        ],
                        value='all',
                        className="mb-3",
                        inputStyle={"marginRight": "5px"},
                        labelStyle={"marginRight": "15px", "display": "inline-block"}
                    )
                ], width=4),
                dbc.Col([
                    html.Label("Trend Smoothing", style={"color": "inherit", "marginBottom": "5px"}),
                    dcc.Slider(
                        id='smoothing-slider',
                        min=3,
                        max=14,
                        step=1,
                        value=7,
                        marks={i: str(i) for i in range(3, 15, 2)}
                    )
                ], width=4)
            ], className="mb-4"),
            dbc.Tabs([
                dbc.Tab(label="Sleep Overview", children=[
                    # Time Series Row
                    dbc.Row([
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader("Select Metric to Visualize"),
                                dbc.CardBody([
                                    dcc.Dropdown(
                                        id='metric-select',
                                        options=[{"label": v, "value": k} for k, v in metric_options.items()],
                                        value='total_asleep_mins',
                                        style={"color": "black", "backgroundColor": "#888888"}
                                    )
                                ])
                            ], className="mb-4", style={"background-color": "#2a2a2a", "border-color": "#444"}),
                            dcc.Graph(id='timeseries-graph', style={"height": "400px"}, config={"displayModeBar": False})
                        ], width=9),
                        dbc.Col(html.Div(id='timeseries-stats'), width=3)
                    ], className="mb-4"),
                    # Gantt Chart Row
                    dbc.Row([
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader("Select a Special Night"),
                                dbc.CardBody([
                                    dcc.Dropdown(
                                        id="night-select",
                                        options=night_dropdown_options,
                                        value=night_dropdown_options[0]["value"],
                                        placeholder="Choose a special night...",
                                        style={"color": "black", "backgroundColor": "#888888"}
                                    )
                                ])
                            ], className="mb-4", style={"background-color": "#2a2a2a", "border-color": "#444"}),
                            dcc.Graph(id="gantt-chart", style={"height": "400px"}, config={"displayModeBar": False})
                        ], width=9),
                        dbc.Col(html.Div(id='gantt-stats'), width=3)
                    ], className="mb-4"),
                    # Sleep Timing Row
                    dbc.Row([
                        dbc.Col(dcc.Graph(id='sleep-timing-graph', style={"height": "400px"}, config={"displayModeBar": False}), width=9),
                        dbc.Col(html.Div(id='timing-stats'), width=3)
                    ])
                ]),
                dbc.Tab(label="Correlations", children=[
                    dbc.Row([
                        dbc.Col(dcc.Graph(id='correlation-matrix', style={"height": "700px"}, config={"displayModeBar": False}), width=8),
                        dbc.Col([
                            html.Div(id='correlation-explanation', className="mt-4"),
                            html.Div(id='correlation-stats')
                        ], width=4)
                    ])
                ])
            ], className="mb-4"),
            dbc.Row([
                dbc.Col([
                    html.Hr(),
                    html.P("Enhanced Sleep Data Dashboard", 
                           className="text-center text-muted",
                           style={"fontFamily": "Helvetica, sans-serif"})
                ])
            ])
        ]
    )
    
    # -------------------- CALLBACKS --------------------
    
    @app.callback(
        Output("page-content", "style"),
        [Input("dark-mode-toggle", "value")]
    )
    def toggle_dark_mode(dark_mode):
        if dark_mode:
            return {"backgroundColor": "#121212", "color": "white", "minHeight": "100vh"}
        else:
            return {"backgroundColor": "#f8f9fa", "color": "black", "minHeight": "100vh"}
    
    @app.callback(
        Output('timeseries-graph', 'figure'),
        Input('metric-select', 'value'),
        Input('week-filter', 'value'),
        Input('time-period-select', 'value'),
        Input('smoothing-slider', 'value')
    )
    def update_timeseries_figure(metric, week_filter, time_period, smoothing_window):
        if not metric:
            metric = 'total_asleep_mins'
        df = get_time_period_df(time_period, agg, weekly_agg, monthly_agg)
        df = apply_week_filter(df, week_filter)
        fig = create_timeseries(df, metric, time_period, smoothing_window)
        return fig

    @app.callback(
        Output('timeseries-stats', 'children'),
        Input('metric-select', 'value'),
        Input('week-filter', 'value'),
        Input('time-period-select', 'value'),
        Input('timeseries-graph', 'relayoutData')
    )
    def update_timeseries_stats(metric, week_filter, time_period, relayout_data):
        if not metric:
            metric = 'total_asleep_mins'
        df = get_time_period_df(time_period, agg, weekly_agg, monthly_agg)
        df = apply_week_filter(df, week_filter)
        df = filter_df_by_time_range(df, relayout_data)
        time_range = get_time_range_title(relayout_data)
        stats = make_summary_stats(df, metric, f"{metric_options.get(metric, metric)} Statistics ({time_range})")
        return stats

    @app.callback(
        Output('sleep-timing-graph', 'figure'),
        Input('week-filter', 'value'),
        Input('time-period-select', 'value')
    )
    def update_sleep_timing_figure(week_filter, time_period):
        df = get_time_period_df(time_period, agg, weekly_agg, monthly_agg)
        df = apply_week_filter(df, week_filter)
        fig = create_sleep_timing_chart(df)
        return fig

    @app.callback(
        Output('timing-stats', 'children'),
        Input('week-filter', 'value'),
        Input('time-period-select', 'value'),
        Input('sleep-timing-graph', 'relayoutData')
    )
    def update_sleep_timing_stats(week_filter, time_period, relayout_data):
        df = get_time_period_df(time_period, agg, weekly_agg, monthly_agg)
        df = apply_week_filter(df, week_filter)
        df = filter_df_by_time_range(df, relayout_data)
        time_range = get_time_range_title(relayout_data)
        stats = make_timing_stats(df, time_range)
        return stats

    @app.callback(
        Output('correlation-matrix', 'figure'),
        Input('week-filter', 'value'),
        Input('time-period-select', 'value')
    )
    def update_correlation_figure(week_filter, time_period):
        df = get_time_period_df(time_period, agg, weekly_agg, monthly_agg)
        df = apply_week_filter(df, week_filter)
        fig = create_correlation_matrix(df)
        return fig

    @app.callback(
        Output('correlation-explanation', 'children'),
        Input('week-filter', 'value'),
        Input('time-period-select', 'value'),
        Input('correlation-matrix', 'relayoutData')
    )
    def update_correlation_explanation(week_filter, time_period, relayout_data):
        df = get_time_period_df(time_period, agg, weekly_agg, monthly_agg)
        df = apply_week_filter(df, week_filter)
        df = filter_df_by_time_range(df, relayout_data)
        time_range = get_time_range_title(relayout_data)
        explanation = dbc.Card([
            dbc.CardHeader(f"About Correlations ({time_range})"),
            dbc.CardBody([
                html.P("This matrix shows relationships between different sleep metrics:"),
                html.Ul([
                    html.Li("Blue dots represent weekdays"),
                    html.Li("Orange dots represent weekends"),
                    html.Li("Strong diagonal patterns indicate strong correlation"),
                    html.Li("Scattered patterns suggest weak or no correlation")
                ]),
                html.P("Analyze these patterns to better understand your sleep habits.")
            ])
        ], style={"background-color": "#2a2a2a", "border-color": "#444"})
        return explanation

    @app.callback(
        Output("gantt-chart", "figure"),
        Output("gantt-stats", "children"),
        Input("night-select", "value")
    )
    def update_gantt(night_value):
        if not night_value:
            fig = go.Figure()
            fig.update_layout(
                title="Select a special night to view the Gantt chart",
                plot_bgcolor="#1a1a1a",
                paper_bgcolor="#1a1a1a",
                font=dict(color="white"),
            )
            return fig, ""
        mask = gantt_data["creationDate"].dt.date.astype(str) == night_value
        this_night_df = gantt_data[mask].copy()
        if this_night_df.empty:
            fig = go.Figure()
            fig.update_layout(title="No data for that night")
            return fig, ""
        fig = create_enhanced_gantt_one_night(this_night_df)
        gantt_stats = build_gantt_stats(night_value, gantt_data, agg)
        return fig, gantt_stats

    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8050)))

if __name__ == "__main__":
    run_dashboard()
