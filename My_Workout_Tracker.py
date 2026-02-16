import streamlit as st
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, date
import calendar

# --- Configuration ---
st.set_page_config(
    page_title="Workout Tracker 2026",
    page_icon="💪",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# --- Custom CSS for mobile-friendly dark mode ---
st.markdown("""
<style>
    .stApp { max-width: 100%; }
    .section-header {
        font-size: 1.1rem; font-weight: bold; color: #4da6ff;
        border-bottom: 2px solid #4da6ff; padding-bottom: 5px;
        margin-top: 0.5rem; margin-bottom: 1rem;
    }
    .info-box { background-color: #262730; padding: 10px; border-radius: 8px; margin: 10px 0; }
    .info-label { color: #a0a0a0; font-size: 0.8rem; }
    .info-value { color: #ffffff; font-size: 1rem; font-weight: bold; }
    .target-box {
        background: linear-gradient(135deg, #8B2500 0%, #CD4F00 100%);
        padding: 12px 16px; border-radius: 10px; margin: 12px 0;
        border-left: 4px solid #FF6B35;
    }
    .target-text { color: #ffffff; font-size: 1.1rem; font-weight: 600; text-shadow: 1px 1px 2px rgba(0,0,0,0.3); }
    .stButton > button { width: 100%; padding: 0.75rem; font-size: 1rem; }
    .stNumberInput > div > div > input { font-size: 1.3rem; padding: 0.75rem; text-align: center; }
    div[data-testid="stCheckbox"] label { font-size: 1.1rem; }
    .progress-indicator { text-align: center; font-size: 1rem; color: #a0a0a0; margin-bottom: 0.5rem; }
    div[data-testid="stHorizontalBlock"] button { min-height: 60px; }
    .set-label { font-size: 1.1rem; font-weight: bold; padding-top: 0.5rem; }
    .scorecard-done { color: #00cc66; font-size: 2rem; font-weight: bold; }
    .scorecard-total { color: #a0a0a0; font-size: 1.5rem; }
</style>
""", unsafe_allow_html=True)

# --- Column Name Constants ---
COL_REST = "Rest between sets"
COL_LOAD_VAR = "Load / Variation Used"
COL_AVG_RIR = "Avg RIR (0–5)"
COL_DONE = "Done (TRUE/FALSE)"

# --- Google Sheets Setup ---
SHEET_ID = "1pkkIxVkEvcJHQ1vCzWFk-0E3GnzDgD21IUSyun5GJKs"
CREDENTIALS_FILE = r"C:\Users\CP362988\source\repos\My Workout Tracket\google_credentials.json"
SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive"
]

def get_gspread_client():
    import os
    try:
        creds_dict = st.secrets["gcp_service_account"]
        credentials = Credentials.from_service_account_info(dict(creds_dict), scopes=SCOPES)
    except (KeyError, FileNotFoundError):
        os.environ['SSL_CERT_FILE'] = r'C:\Users\CP362988\source\repos\My Workout Tracket\zscaler.pem.cer'
        os.environ['REQUESTS_CA_BUNDLE'] = r'C:\Users\CP362988\source\repos\My Workout Tracket\zscaler.pem.cer'
        credentials = Credentials.from_service_account_file(CREDENTIALS_FILE, scopes=SCOPES)
    return gspread.authorize(credentials)

def sort_mixed_column(values):
    unique_vals = list(values.dropna().unique())
    try:
        return sorted(unique_vals, key=lambda x: (isinstance(x, str), int(x) if not isinstance(x, str) else float('inf'), str(x)))
    except (ValueError, TypeError):
        return sorted(unique_vals, key=str)

@st.cache_data(ttl=300)
def load_data():
    client = get_gspread_client()
    sheet = client.open_by_key(SHEET_ID).sheet1
    records = sheet.get_all_records()
    df = pd.DataFrame(records)
    text_cols = ['Sets', 'Reps', 'RIR', COL_REST]
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).replace('', '-').replace('nan', '-')
    numeric_cols = ['Set1', 'Set2', 'Set3', 'Set4', 'Set5', COL_AVG_RIR]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    if 'Week' in df.columns:
        df['Week'] = pd.to_numeric(df['Week'], errors='coerce').fillna(0).astype(int)
    if 'Day' in df.columns:
        df['Day'] = df['Day'].astype(str).str.strip()
    for col in df.columns:
        if col not in text_cols:
            df[col] = df[col].replace('', pd.NA)
    return df

def get_exercise_key(week, day, section, exercise):
    return f"{week}_{day}_{section}_{exercise}"

def parse_set_value(val):
    if pd.isna(val):
        return 0
    try:
        return int(float(str(val).strip()))
    except (ValueError, TypeError):
        return 0

def calculate_total_reps(row):
    total = 0
    for i in range(1, 6):
        col = f'Set{i}'
        if col in row.index:
            total += parse_set_value(row[col])
    return total

def calculate_session_stats(row):
    set_values = [parse_set_value(row.get(f'Set{i}', 0)) for i in range(1, 6)]
    return sum(set_values), max(set_values) if set_values else 0

def parse_done(val):
    if pd.isna(val):
        return False
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return val == 1 or val == 1.0
    if isinstance(val, str):
        return val.strip().upper() in ['TRUE', '1', 'YES', 'Y']
    return False

def parse_last_saved_date(val):
    if pd.isna(val) or val is None:
        return None
    try:
        val_str = str(val).strip()
        if val_str in ['', 'nan', 'None']:
            return None
        return datetime.strptime(val_str[:10], "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return None

def save_data(updates: list):
    if not updates:
        return
    client = get_gspread_client()
    sheet = client.open_by_key(SHEET_ID).sheet1
    all_data = sheet.get_all_values()
    headers = all_data[0]
    col_map = {name: idx + 1 for idx, name in enumerate(headers)}
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    target_week = str(updates[0]['Week']).strip()
    target_day = str(updates[0]['Day']).strip()
    updated_rows = set()

    for update in updates:
        for row_idx, row in enumerate(all_data[1:], start=2):
            row_dict = dict(zip(headers, row))
            if (str(row_dict.get('Week', '')).strip() == str(update['Week']).strip() and
                str(row_dict.get('Day', '')).strip() == str(update['Day']).strip() and
                str(row_dict.get('Section', '')).strip() == str(update['Section']).strip() and
                str(row_dict.get('Exercise', '')).strip() == str(update['Exercise']).strip()):
                cells = []
                for i, v in enumerate(update['sets'], start=1):
                    if i <= 5 and f'Set{i}' in col_map:
                        cells.append({'row': row_idx, 'col': col_map[f'Set{i}'], 'value': v if v > 0 else ''})
                if COL_LOAD_VAR in col_map:
                    cells.append({'row': row_idx, 'col': col_map[COL_LOAD_VAR], 'value': update['load_variation']})
                if COL_AVG_RIR in col_map:
                    cells.append({'row': row_idx, 'col': col_map[COL_AVG_RIR], 'value': update['avg_rir'] if update['avg_rir'] > 0 else ''})
                if COL_DONE in col_map:
                    cells.append({'row': row_idx, 'col': col_map[COL_DONE], 'value': 'TRUE' if update['done'] else 'FALSE'})
                if 'LastSaved' in col_map:
                    cells.append({'row': row_idx, 'col': col_map['LastSaved'], 'value': timestamp})
                for cell in cells:
                    sheet.update_cell(cell['row'], cell['col'], cell['value'])
                updated_rows.add(row_idx)
                break

    if 'LastSaved' in col_map:
        for row_idx, row in enumerate(all_data[1:], start=2):
            if row_idx in updated_rows:
                continue
            row_dict = dict(zip(headers, row))
            if (str(row_dict.get('Week', '')).strip() == target_week and
                str(row_dict.get('Day', '')).strip() == target_day):
                sheet.update_cell(row_idx, col_map['LastSaved'], timestamp)

def find_next_workout(df):
    all_weeks = sorted(df['Week'].unique())
    all_workouts = []
    for week in all_weeks:
        week_df = df[df['Week'] == week]
        for day in sort_mixed_column(week_df['Day']):
            all_workouts.append((week, day))
    if not all_workouts:
        return (1, "1")
    df['IsDone'] = df[COL_DONE].apply(parse_done)
    done_df = df[df['IsDone'] == True]
    if done_df.empty:
        return all_workouts[0]
    last_done_week = done_df['Week'].max()
    sorted_days = sort_mixed_column(pd.Series(done_df[done_df['Week'] == last_done_week]['Day'].unique()))
    last_done_day = sorted_days[-1] if sorted_days else "1"
    try:
        last_idx = next((i for i, (w, d) in enumerate(all_workouts) if w == last_done_week and str(d) == str(last_done_day)), None)
        if last_idx is not None and last_idx < len(all_workouts) - 1:
            return all_workouts[last_idx + 1]
        return (last_done_week, str(last_done_day))
    except (ValueError, IndexError):
        return all_workouts[0]

def get_calendar_data(df):
    calendar_data = {}
    if 'LastSaved' not in df.columns:
        return calendar_data
    for _, row in df.iterrows():
        saved_date = parse_last_saved_date(row.get('LastSaved'))
        if saved_date is None:
            continue
        date_str = saved_date.isoformat()
        is_done = parse_done(row.get(COL_DONE))
        if date_str not in calendar_data:
            calendar_data[date_str] = {'total': 0, 'done': 0}
        calendar_data[date_str]['total'] += 1
        if is_done:
            calendar_data[date_str]['done'] += 1
    return calendar_data


# --- Main App ---
st.title("💪 Workout Tracker 2026")

app_mode = st.radio(
    "Select Mode",
    options=["🏋️ Tracker", "📊 Progress", "🏆 Records"],
    horizontal=True,
    label_visibility="collapsed"
)

st.divider()

try:
    df = load_data()

    # ========================================
    if app_mode == "🏋️ Tracker":
    # ========================================

        if 'extra_sets' not in st.session_state:
            st.session_state.extra_sets = {}
        if 'exercise_inputs' not in st.session_state:
            st.session_state.exercise_inputs = {}
        if 'current_exercise_idx' not in st.session_state:
            st.session_state.current_exercise_idx = 0
        if 'last_week_day' not in st.session_state:
            st.session_state.last_week_day = None

        default_week, default_day = find_next_workout(df)

        col1, col2 = st.columns(2)
        with col1:
            weeks = sort_mixed_column(df['Week'])
            try:
                default_week_idx = list(weeks).index(default_week)
            except ValueError:
                default_week_idx = 0
            selected_week = st.selectbox("📅 Select Week", weeks, index=default_week_idx, key="week_select")

        with col2:
            filtered_df = df[df['Week'].astype(str) == str(selected_week)]
            days = sort_mixed_column(filtered_df['Day'])
            if selected_week == default_week:
                try:
                    default_day_idx = [str(d) for d in days].index(str(default_day))
                except ValueError:
                    default_day_idx = 0
            else:
                default_day_idx = 0
            selected_day = st.selectbox("📆 Select Day", days, index=default_day_idx, key="day_select")

        current_week_day = f"{selected_week}_{selected_day}"
        if st.session_state.last_week_day != current_week_day:
            st.session_state.current_exercise_idx = 0
            st.session_state.last_week_day = current_week_day

        day_df = df[(df['Week'].astype(str) == str(selected_week)) &
                    (df['Day'].astype(str) == str(selected_day))]

        if day_df.empty:
            st.warning("No workouts found for this selection.")
        else:
            workout_type = day_df['Type'].iloc[0] if pd.notna(day_df['Type'].iloc[0]) else "Workout"
            st.markdown(f"## {workout_type}")

            exercises_list = []
            for section in day_df['Section'].unique():
                for idx, row in day_df[day_df['Section'] == section].iterrows():
                    exercises_list.append({'section': section, 'row': row, 'df_idx': idx})

            total_exercises = len(exercises_list)
            current_idx = max(0, min(st.session_state.current_exercise_idx, total_exercises - 1))
            st.session_state.current_exercise_idx = current_idx

            st.markdown(f'<div class="progress-indicator">Exercise {current_idx + 1} of {total_exercises}</div>', unsafe_allow_html=True)

            nav_col1, nav_col2, nav_col3 = st.columns([1, 2, 1])
            with nav_col1:
                if st.button("◀", key="prev_btn", disabled=(current_idx == 0), use_container_width=True):
                    st.session_state.current_exercise_idx = current_idx - 1
                    st.rerun()
            with nav_col3:
                if st.button("▶", key="next_btn", disabled=(current_idx >= total_exercises - 1), use_container_width=True):
                    st.session_state.current_exercise_idx = current_idx + 1
                    st.rerun()

            st.divider()

            current_exercise = exercises_list[current_idx]
            section = current_exercise['section']
            row = current_exercise['row']
            exercise_key = get_exercise_key(selected_week, selected_day, section, row['Exercise'])

            if exercise_key not in st.session_state.exercise_inputs:
                existing_sets = {}
                for i in range(1, 6):
                    if f'Set{i}' in row.index:
                        existing_sets[f'set_{i}'] = parse_set_value(row.get(f'Set{i}'))
                existing_load_var = ''
                if COL_LOAD_VAR in row.index:
                    lv = row.get(COL_LOAD_VAR)
                    if pd.notna(lv) and str(lv).strip() not in ['', 'nan', 'None']:
                        existing_load_var = str(lv)
                existing_avg_rir = 0
                if COL_AVG_RIR in row.index:
                    rv = row.get(COL_AVG_RIR)
                    if pd.notna(rv):
                        try:
                            existing_avg_rir = int(float(rv))
                        except (ValueError, TypeError):
                            existing_avg_rir = 0
                st.session_state.exercise_inputs[exercise_key] = {
                    'sets': existing_sets, 'load_variation': existing_load_var, 'avg_rir': existing_avg_rir
                }

            st.markdown(f"#### 📌 {section}")
            st.subheader(row['Exercise'] if pd.notna(row['Exercise']) else "Exercise")
            if pd.notna(row['Description']):
                st.caption(row['Description'])

            def safe_display(val, default="-"):
                if val is None:
                    return default
                try:
                    if pd.isna(val):
                        return default
                except (TypeError, ValueError):
                    pass
                return default if str(val).strip() in ['', 'nan', 'None', '-'] else str(val)

            target_rir = safe_display(row.get('RIR'))
            rest_val = safe_display(row.get(COL_REST))
            st.markdown(f'''
                <div class="target-box">
                    <span class="target-text">🎯 Target RIR: {target_rir}  |  ⏱ Rest: {rest_val}</span>
                </div>
            ''', unsafe_allow_html=True)

            target_sets = str(row['Sets']) if pd.notna(row['Sets']) and str(row['Sets']).strip() not in ['', 'nan', 'None', '0'] else '-'
            target_reps = str(row['Reps']) if pd.notna(row['Reps']) and str(row['Reps']).strip() not in ['', 'nan', 'None', '0'] else '-'
            info_cols = st.columns(2)
            with info_cols[0]:
                st.metric(label="🎯 Target Sets", value=target_sets)
            with info_cols[1]:
                st.metric(label="🔁 Target Reps", value=target_reps)

            if pd.notna(row.get('Escalation')) and str(row['Escalation']).strip():
                st.info(f"📈 **Escalation:** {row['Escalation']}")
            if pd.notna(row.get('Notes')) and str(row['Notes']).strip():
                st.caption(f"📝 {row['Notes']}")

            def parse_sets_for_count(val):
                val_str = str(val).strip()
                if val_str in ['', '-', 'nan', 'None']:
                    return 3
                if '-' in val_str:
                    try:
                        return int(val_str.split('-')[0])
                    except ValueError:
                        return 3
                try:
                    return int(float(val_str))
                except (ValueError, TypeError):
                    return 3

            stored_inputs = st.session_state.exercise_inputs[exercise_key]
            base_sets = parse_sets_for_count(row.get('Sets'))
            sets_with_data = len([k for k, v in stored_inputs['sets'].items() if v > 0])
            base_sets = max(base_sets, sets_with_data)
            extra = st.session_state.extra_sets.get(exercise_key, 0)
            total_sets = base_sets + extra

            st.markdown("**Log Your Sets:**")
            for set_num in range(1, total_sets + 1):
                set_key = f"set_{set_num}"
                if set_key not in stored_inputs['sets']:
                    stored_inputs['sets'][set_key] = 0
                set_cols = st.columns([1, 3])
                with set_cols[0]:
                    st.markdown(f'<div class="set-label">Set {set_num}</div>', unsafe_allow_html=True)
                with set_cols[1]:
                    reps = st.number_input(
                        f"Reps for Set {set_num}", min_value=0,
                        value=stored_inputs['sets'][set_key],
                        key=f"{exercise_key}_set{set_num}_reps",
                        label_visibility="collapsed"
                    )
                    stored_inputs['sets'][set_key] = reps

            if st.button("➕ Add Set", key=f"{exercise_key}_add_set"):
                st.session_state.extra_sets[exercise_key] = extra + 1
                st.rerun()

            st.markdown("---")

            avg_rir = st.number_input(
                "Average RIR (0-5)", min_value=0, max_value=5,
                value=stored_inputs['avg_rir'], key=f"{exercise_key}_avg_rir",
                help="Enter your average RIR across all sets"
            )
            stored_inputs['avg_rir'] = avg_rir

            load_variation = st.text_input(
                "Load / Variation Used", value=stored_inputs['load_variation'],
                key=f"{exercise_key}_load_var", placeholder="e.g., Drop set, Pause reps..."
            )
            stored_inputs['load_variation'] = load_variation

            st.markdown("---")

            if st.button("💾 SAVE ALL EXERCISES", type="primary", use_container_width=True):
                try:
                    all_exercise_updates = []
                    for ex in exercises_list:
                        ex_key = get_exercise_key(selected_week, selected_day, ex['section'], ex['row']['Exercise'])
                        if ex_key in st.session_state.exercise_inputs:
                            ex_inputs = st.session_state.exercise_inputs[ex_key]
                            sets_data = [ex_inputs['sets'][k] for k in sorted(ex_inputs['sets'].keys())]
                            all_exercise_updates.append({
                                'Week': selected_week, 'Day': selected_day,
                                'Section': ex['section'], 'Exercise': ex['row']['Exercise'],
                                'sets': sets_data, 'load_variation': ex_inputs['load_variation'],
                                'avg_rir': ex_inputs['avg_rir'], 'done': any(r > 0 for r in sets_data)
                            })
                    st.toast('Saving...', icon='⏳')
                    with st.spinner('💾 Saving your workout to Google Sheets...'):
                        save_data(all_exercise_updates)
                    st.success("✅ Workout saved successfully!")
                    st.balloons()
                    st.cache_data.clear()
                except gspread.exceptions.APIError as e:
                    st.error("❌ Could not connect to Google Sheets.")
                    st.caption(f"Technical details: {str(e)}")
                except Exception as e:
                    st.error("❌ Something went wrong while saving.")
                    st.caption(f"Error: {str(e)}")

    # ========================================
    elif app_mode == "📊 Progress":
    # ========================================

        st.header("📊 Progress Overview")

        # Filters
        filter_col1, filter_col2 = st.columns(2)
        with filter_col1:
            week_options = ["All Weeks"] + [str(w) for w in sort_mixed_column(df['Week'])]
            selected_week_filter = st.selectbox("📅 Filter by Week", week_options, key="progress_week")
        with filter_col2:
            day_options = ["All Days"] + [str(d) for d in sort_mixed_column(df['Day'])]
            selected_day_filter = st.selectbox("📆 Filter by Day", day_options, key="progress_day")

        filtered_df = df.copy()
        if selected_week_filter != "All Weeks":
            filtered_df = filtered_df[filtered_df['Week'].astype(str) == selected_week_filter]
        if selected_day_filter != "All Days":
            filtered_df = filtered_df[filtered_df['Day'].astype(str) == selected_day_filter]

        filtered_df['TotalReps'] = filtered_df.apply(calculate_total_reps, axis=1)
        filtered_df['IsDone'] = filtered_df[COL_DONE].apply(parse_done)

        # Scorecard
        total_exercises = len(filtered_df)
        done_exercises = filtered_df['IsDone'].sum()
        completion_pct = (done_exercises / total_exercises * 100) if total_exercises > 0 else 0
        done_rows = filtered_df[filtered_df['IsDone'] == True]
        days_completed = done_rows.groupby(['Week', 'Day']).ngroups if not done_rows.empty else 0
        total_days = filtered_df.groupby(['Week', 'Day']).ngroups
        weeks_active = done_rows['Week'].nunique() if not done_rows.empty else 0
        total_weeks = filtered_df['Week'].nunique()

        metric_col1, metric_col2, metric_col3 = st.columns(3)
        with metric_col1:
            st.metric("Exercises", f"{int(done_exercises)}/{total_exercises}", delta=f"{completion_pct:.0f}%")
        with metric_col2:
            days_pct = (days_completed / total_days * 100) if total_days > 0 else 0
            st.metric("Days", f"{days_completed}/{total_days}", delta=f"{days_pct:.0f}%")
        with metric_col3:
            weeks_pct = (weeks_active / total_weeks * 100) if total_weeks > 0 else 0
            st.metric("Weeks", f"{weeks_active}/{total_weeks}", delta=f"{weeks_pct:.0f}%")

        st.divider()

        # Progress by Section chart
        st.subheader("📈 Progress by Section")
        df_for_trends = df.copy()
        if selected_day_filter != "All Days":
            df_for_trends = df_for_trends[df_for_trends['Day'].astype(str) == selected_day_filter]
        df_for_trends['TotalReps'] = df_for_trends.apply(calculate_total_reps, axis=1)

        section_progress = df_for_trends[df_for_trends['TotalReps'] > 0].groupby(
            ['Week', 'Section']
        )['TotalReps'].sum().reset_index()

        if not section_progress.empty:
            section_progress['WeekSort'] = section_progress['Week'].apply(
                lambda x: (0, int(x)) if not isinstance(x, str) else (1, 0)
            )
            section_progress = section_progress.sort_values('WeekSort')
            section_progress['Week'] = section_progress['Week'].astype(str)
            fig_line = px.line(section_progress, x='Week', y='TotalReps', color='Section', markers=True, title='')
            fig_line.update_layout(
                paper_bgcolor='#ffffff', plot_bgcolor='#f8f9fa', font=dict(color='#333333'),
                xaxis=dict(gridcolor='#e0e0e0', linecolor='#cccccc', title='Week', title_font=dict(color='#333333'), tickfont=dict(color='#333333')),
                yaxis=dict(gridcolor='#e0e0e0', linecolor='#cccccc', title='Total Reps', title_font=dict(color='#333333'), tickfont=dict(color='#333333')),
                legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5, font=dict(color='#333333')),
                height=400
            )
            st.plotly_chart(fig_line, use_container_width=True)
        else:
            st.info("No weekly progress data yet.")

        st.divider()

        # RIR Trend
        st.subheader("🎯 Average RIR Trend")
        st.caption("Lower RIR = working harder 💪")

        def parse_avg_rir(val):
            if pd.isna(val):
                return None
            try:
                return float(val)
            except (ValueError, TypeError):
                return None

        df_for_trends['AvgRIR'] = df_for_trends[COL_AVG_RIR].apply(parse_avg_rir)
        rir_by_week = df_for_trends[df_for_trends['AvgRIR'].notna()].groupby('Week')['AvgRIR'].mean().reset_index()

        if not rir_by_week.empty:
            rir_by_week['WeekSort'] = rir_by_week['Week'].apply(lambda x: (0, int(x)) if not isinstance(x, str) else (1, 0))
            rir_by_week = rir_by_week.sort_values('WeekSort')
            rir_by_week['Week'] = rir_by_week['Week'].astype(str)
            fig_rir = px.line(rir_by_week, x='Week', y='AvgRIR', markers=True, title='')
            fig_rir.update_traces(line_color='#ff6b35', marker_color='#ff6b35')
            fig_rir.update_layout(
                paper_bgcolor='#ffffff', plot_bgcolor='#f8f9fa', font=dict(color='#333333'),
                xaxis=dict(gridcolor='#e0e0e0', linecolor='#cccccc', title='Week', title_font=dict(color='#333333'), tickfont=dict(color='#333333')),
                yaxis=dict(gridcolor='#e0e0e0', linecolor='#cccccc', title='Average RIR', range=[0, 5], title_font=dict(color='#333333'), tickfont=dict(color='#333333')),
                height=300
            )
            st.plotly_chart(fig_rir, use_container_width=True)
        else:
            st.info("No RIR data recorded yet.")

        st.divider()

        # Exercise Summary Table
        st.subheader("📋 Exercise Summary")

        def count_sets_logged(row):
            return sum(1 for i in range(1, 6) if parse_set_value(row.get(f'Set{i}', 0)) > 0)

        filtered_df['SetsLogged'] = filtered_df.apply(count_sets_logged, axis=1)
        summary_df = filtered_df.groupby(['Exercise', 'Section']).agg({'IsDone': 'sum', 'SetsLogged': 'sum'}).reset_index()
        summary_df.columns = ['Exercise', 'Section', 'Times Completed', 'Total Sets Logged']
        summary_df = summary_df.sort_values('Times Completed', ascending=False)
        st.dataframe(summary_df, use_container_width=True, hide_index=True,
            column_config={
                "Exercise": st.column_config.TextColumn("Exercise", width="large"),
                "Section": st.column_config.TextColumn("Section", width="medium"),
                "Times Completed": st.column_config.NumberColumn("✅ Done", format="%d"),
                "Total Sets Logged": st.column_config.NumberColumn("🔢 Sets", format="%d")
            }
        )

        st.divider()

        # --- CALENDAR (BELOW PROGRESS) ---
        st.subheader("📅 Workout Calendar")

        if 'calendar_year' not in st.session_state:
            st.session_state.calendar_year = date.today().year
        if 'calendar_month' not in st.session_state:
            st.session_state.calendar_month = date.today().month
        if 'selected_calendar_date' not in st.session_state:
            st.session_state.selected_calendar_date = None

        calendar_data = get_calendar_data(df)

        nav_col1, nav_col2, nav_col3 = st.columns([1, 3, 1])
        with nav_col1:
            if st.button("◀", key="prev_month", use_container_width=True):
                if st.session_state.calendar_month == 1:
                    st.session_state.calendar_month = 12
                    st.session_state.calendar_year -= 1
                else:
                    st.session_state.calendar_month -= 1
                st.rerun()
        with nav_col2:
            month_name = calendar.month_name[st.session_state.calendar_month]
            st.markdown(f"<h3 style='text-align:center; margin:0;'>{month_name} {st.session_state.calendar_year}</h3>", unsafe_allow_html=True)
        with nav_col3:
            if st.button("▶", key="next_month", use_container_width=True):
                if st.session_state.calendar_month == 12:
                    st.session_state.calendar_month = 1
                    st.session_state.calendar_year += 1
                else:
                    st.session_state.calendar_month += 1
                st.rerun()

        cal = calendar.Calendar(firstweekday=0)
        month_days = cal.monthdayscalendar(st.session_state.calendar_year, st.session_state.calendar_month)
        today = date.today()

        html_rows = []
        day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        header_cells = ''.join([f"<th style='padding:8px; color:#a0a0a0; font-weight:bold; text-align:center;'>{d}</th>" for d in day_names])
        html_rows.append(f"<tr>{header_cells}</tr>")

        for week in month_days:
            row_cells = []
            for day in week:
                if day == 0:
                    row_cells.append("<td style='padding:8px; min-height:50px;'></td>")
                else:
                    current_date = date(st.session_state.calendar_year, st.session_state.calendar_month, day)
                    date_str = current_date.isoformat()
                    status_dot = ""
                    if date_str in calendar_data:
                        d = calendar_data[date_str]
                        if d['total'] > 0 and d['done'] == d['total']:
                            status_dot = "<br><span style='font-size:0.8rem;'>🟢</span>"
                        elif d['total'] > 0:
                            status_dot = "<br><span style='font-size:0.8rem;'>🔴</span>"
                    is_selected = st.session_state.selected_calendar_date == date_str
                    is_today = current_date == today
                    has_data = date_str in calendar_data
                    cell_style = "padding:8px; min-height:50px; text-align:center; border-radius:8px; vertical-align:middle;"
                    if has_data:
                        cell_style += " background-color:#262730;"
                    if is_selected:
                        cell_style += " border:2px solid #4da6ff; background-color:#1e3a5f;"
                    elif is_today:
                        cell_style += " border:2px solid #ffaa00;"
                    fw = 'bold' if is_selected else 'normal'
                    row_cells.append(f"<td style='{cell_style}'><span style='color:#ffffff; font-weight:{fw};'>{day}</span>{status_dot}</td>")
            html_rows.append(f"<tr>{''.join(row_cells)}</tr>")

        st.markdown(f"""
        <table style='width:100%; border-collapse:separate; border-spacing:4px; table-layout:fixed;'>
            {''.join(html_rows)}
        </table>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div style='display:flex; gap:20px; justify-content:center; margin:10px 0; font-size:0.8rem;'>
            <span>🟢 All completed</span><span>🔴 Partial</span>
            <span style='color:#ffaa00;'>▢ Today</span><span style='color:#4da6ff;'>▢ Selected</span>
        </div>
        """, unsafe_allow_html=True)

        dates_with_data = sorted([d for d in calendar_data.keys()
                                  if d.startswith(f"{st.session_state.calendar_year}-{st.session_state.calendar_month:02d}")])
        if dates_with_data:
            date_options = ["Select a date..."] + dates_with_data
            selected_idx = 0
            if st.session_state.selected_calendar_date in dates_with_data:
                selected_idx = dates_with_data.index(st.session_state.selected_calendar_date) + 1
            selected_date_option = st.selectbox("📅 Select workout date to view details:", date_options, index=selected_idx, key="calendar_date_select")
            if selected_date_option != "Select a date...":
                st.session_state.selected_calendar_date = selected_date_option
        else:
            st.info("No workout data recorded this month.")

        st.divider()

        # Day Detail Panel
        st.subheader("📋 Day Details")
        if st.session_state.selected_calendar_date:
            selected_date = st.session_state.selected_calendar_date
            st.success(f"📅 Showing workouts for: **{selected_date}**")
            if 'LastSaved' in df.columns:
                try:
                    target_date = date.fromisoformat(selected_date)
                    day_exercises = df[df['LastSaved'].apply(
                        lambda x: parse_last_saved_date(x) == target_date if pd.notna(x) else False
                    )]
                except ValueError:
                    day_exercises = pd.DataFrame()
            else:
                day_exercises = pd.DataFrame()

            if day_exercises.empty:
                st.warning("No workout data for this date.")
            else:
                for section in day_exercises['Section'].unique():
                    section_df = day_exercises[day_exercises['Section'] == section]
                    with st.expander(f"📌 {section} ({len(section_df)} exercises)", expanded=True):
                        for _, ex_row in section_df.iterrows():
                            done_icon = "✅" if parse_done(ex_row.get(COL_DONE)) else "⬜"
                            st.markdown(f"**{ex_row.get('Exercise', 'Unknown')}** {done_icon}")
                            sets_display = [f"Set {i}: {parse_set_value(ex_row.get(f'Set{i}', 0))}"
                                          for i in range(1, 6) if parse_set_value(ex_row.get(f'Set{i}', 0)) > 0]
                            if sets_display:
                                st.caption(" | ".join(sets_display))
                            lv = ex_row.get(COL_LOAD_VAR)
                            if pd.notna(lv) and str(lv).strip() not in ['', 'nan', 'None']:
                                st.caption(f"🔧 Load/Variation: {lv}")
                            ar = ex_row.get(COL_AVG_RIR)
                            if pd.notna(ar):
                                try:
                                    st.caption(f"🎯 Avg RIR: {int(float(ar))}")
                                except Exception:
                                    pass
                            st.markdown("---")
        else:
            st.info("👆 Select a date from the dropdown above to see workout details.")

    # ========================================
    elif app_mode == "🏆 Records":
    # ========================================

        st.header("🏆 Personal Records")

        records_df = df.copy()
        records_df['TotalReps'] = records_df.apply(lambda r: calculate_session_stats(r)[0], axis=1)
        records_df['BestSingleSet'] = records_df.apply(lambda r: calculate_session_stats(r)[1], axis=1)
        records_df = records_df[records_df['TotalReps'] > 0]

        if records_df.empty:
            st.info("🏋️ No workout data recorded yet. Complete some exercises to see your Personal Records!")
        else:
            st.subheader("🔥 Recent PRs")
            max_week = records_df['Week'].max()
            current_week_df = records_df[records_df['Week'] == max_week]
            previous_weeks_df = records_df[records_df['Week'] < max_week]
            new_prs = []

            if not previous_weeks_df.empty:
                current_bests = current_week_df.groupby('Exercise').agg({'TotalReps': 'max', 'BestSingleSet': 'max'}).reset_index()
                previous_bests = previous_weeks_df.groupby('Exercise').agg({'TotalReps': 'max', 'BestSingleSet': 'max'}).reset_index()
                for _, curr_row in current_bests.iterrows():
                    exercise = curr_row['Exercise']
                    curr_total = curr_row['TotalReps']
                    curr_best_set = curr_row['BestSingleSet']
                    prev_row = previous_bests[previous_bests['Exercise'] == exercise]
                    if not prev_row.empty:
                        prev_total = prev_row['TotalReps'].values[0]
                        prev_best_set = prev_row['BestSingleSet'].values[0]
                        is_total_pr = curr_total > prev_total
                        is_set_pr = curr_best_set > prev_best_set
                        if is_total_pr or is_set_pr:
                            new_prs.append({'exercise': exercise, 'curr_total': curr_total, 'prev_total': prev_total,
                                          'curr_best_set': curr_best_set, 'prev_best_set': prev_best_set,
                                          'is_total_pr': is_total_pr, 'is_set_pr': is_set_pr})
                    else:
                        new_prs.append({'exercise': exercise, 'curr_total': curr_total, 'prev_total': 0,
                                      'curr_best_set': curr_best_set, 'prev_best_set': 0, 'is_total_pr': True, 'is_set_pr': True})

            if new_prs:
                for pr in new_prs:
                    pr_details = []
                    if pr['is_total_pr']:
                        pr_details.append(f"Total: {pr['curr_total']} reps (+{pr['curr_total'] - pr['prev_total']})")
                    if pr['is_set_pr']:
                        pr_details.append(f"Best Set: {pr['curr_best_set']} reps")
                    st.success(f"🏆 **NEW PR!** {pr['exercise']} — {' | '.join(pr_details)}")
            else:
                st.info("💪 No new PRs this week yet. Keep pushing!")

            st.divider()
            st.subheader("📋 All-Time Records")

            all_time_records = []
            for exercise in records_df['Exercise'].unique():
                ex_df = records_df[records_df['Exercise'] == exercise]
                best_total_row = ex_df.loc[ex_df['TotalReps'].idxmax()]
                best_single_set = ex_df['BestSingleSet'].max()
                all_time_records.append({
                    'Exercise': exercise,
                    'Section': best_total_row.get('Section', '-'),
                    '🏆 Best Total Reps': int(best_total_row['TotalReps']),
                    '💪 Best Single Set': int(best_single_set),
                    '📅 Week Achieved': int(best_total_row['Week']),
                    '📆 Day Achieved': str(best_total_row['Day'])
                })

            records_table = pd.DataFrame(all_time_records).sort_values(['Section', 'Exercise'])
            st.dataframe(records_table, use_container_width=True, hide_index=True,
                column_config={
                    "Exercise": st.column_config.TextColumn("Exercise", width="large"),
                    "Section": st.column_config.TextColumn("Section", width="medium"),
                    "🏆 Best Total Reps": st.column_config.NumberColumn("🏆 Best Total", format="%d"),
                    "💪 Best Single Set": st.column_config.NumberColumn("💪 Best Set", format="%d"),
                    "📅 Week Achieved": st.column_config.NumberColumn("📅 Week", format="%d"),
                    "📆 Day Achieved": st.column_config.TextColumn("📆 Day", width="small")
                }
            )

            st.divider()
            st.subheader("📊 Best Total Reps by Exercise")

            pr_chart_data = records_table[['Exercise', '🏆 Best Total Reps']].copy()
            pr_chart_data = pr_chart_data.sort_values('🏆 Best Total Reps', ascending=True)
            fig_pr = px.bar(pr_chart_data, x='🏆 Best Total Reps', y='Exercise', orientation='h', title='',
                           color_discrete_sequence=['#00cc66'])
            fig_pr.update_layout(
                paper_bgcolor='#ffffff', plot_bgcolor='#f8f9fa', font=dict(color='#333333'),
                xaxis=dict(gridcolor='#e0e0e0', linecolor='#cccccc', title='Best Total Reps', title_font=dict(color='#333333'), tickfont=dict(color='#333333')),
                yaxis=dict(gridcolor='#e0e0e0', linecolor='#cccccc', title='', tickfont=dict(color='#333333')),
                height=max(400, len(pr_chart_data) * 40)
            )
            st.plotly_chart(fig_pr, use_container_width=True)

except FileNotFoundError:
    st.error("❌ Credentials file not found!")
    st.info(f"Please ensure 'google_credentials.json' exists at:\n{CREDENTIALS_FILE}")
except gspread.exceptions.SpreadsheetNotFound:
    st.error("❌ Google Sheet not found!")
    st.info(f"Please check the Sheet ID: {SHEET_ID}")
except gspread.exceptions.APIError as e:
    st.error(f"❌ Google Sheets API error: {str(e)}")
except Exception as e:
    st.error(f"❌ Error: {str(e)}")
    st.exception(e)