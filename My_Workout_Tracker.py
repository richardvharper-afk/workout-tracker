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
    .stApp {
        max-width: 100%;
    }
    .section-header {
        font-size: 1.1rem;
        font-weight: bold;
        color: #4da6ff;
        border-bottom: 2px solid #4da6ff;
        padding-bottom: 5px;
        margin-top: 0.5rem;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #262730;
        padding: 10px;
        border-radius: 8px;
        margin: 10px 0;
    }
    .info-label {
        color: #a0a0a0;
        font-size: 0.8rem;
    }
    .info-value {
        color: #ffffff;
        font-size: 1rem;
        font-weight: bold;
    }
    .target-box {
        background: linear-gradient(135deg, #8B2500 0%, #CD4F00 100%);
        padding: 12px 16px;
        border-radius: 10px;
        margin: 12px 0;
        border-left: 4px solid #FF6B35;
    }
    .target-text {
        color: #ffffff;
        font-size: 1.1rem;
        font-weight: 600;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.3);
    }
    .stButton > button {
        width: 100%;
        padding: 0.75rem;
        font-size: 1rem;
    }
    .stNumberInput > div > div > input {
        font-size: 1.3rem;
        padding: 0.75rem;
        text-align: center;
    }
    div[data-testid="stCheckbox"] label {
        font-size: 1.1rem;
    }
    .progress-indicator {
        text-align: center;
        font-size: 1rem;
        color: #a0a0a0;
        margin-bottom: 0.5rem;
    }
    div[data-testid="stHorizontalBlock"] button {
        min-height: 60px;
    }
    .set-label {
        font-size: 1.1rem;
        font-weight: bold;
        padding-top: 0.5rem;
    }
    .scorecard-done {
        color: #00cc66;
        font-size: 2rem;
        font-weight: bold;
    }
    .scorecard-total {
        color: #a0a0a0;
        font-size: 1.5rem;
    }
    .calendar-grid {
        display: grid;
        grid-template-columns: repeat(7, 1fr);
        gap: 4px;
        margin: 10px 0;
    }
    .calendar-header {
        text-align: center;
        font-weight: bold;
        color: #a0a0a0;
        padding: 8px 4px;
        font-size: 0.8rem;
    }
    .calendar-day {
        text-align: center;
        padding: 8px 4px;
        border-radius: 8px;
        background-color: #262730;
        min-height: 50px;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
    }
    .calendar-day-empty {
        background-color: transparent;
    }
    .calendar-day-number {
        font-size: 0.9rem;
        color: #ffffff;
    }
    .calendar-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        margin-top: 4px;
    }
    .calendar-dot-green { background-color: #00cc66; }
    .calendar-dot-red { background-color: #ff4444; }
    .calendar-day-selected {
        border: 2px solid #4da6ff;
        background-color: #1e3a5f;
    }
    .calendar-day-today {
        border: 1px solid #ffaa00;
    }
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
    """Create and return a gspread client."""
    import os
    
    try:
        creds_dict = st.secrets["gcp_service_account"]
        credentials = Credentials.from_service_account_info(
            dict(creds_dict), 
            scopes=SCOPES
        )
    except (KeyError, FileNotFoundError):
        os.environ['SSL_CERT_FILE'] = r'C:\Users\CP362988\source\repos\My Workout Tracket\zscaler.pem.cer'
        os.environ['REQUESTS_CA_BUNDLE'] = r'C:\Users\CP362988\source\repos\My Workout Tracket\zscaler.pem.cer'
        
        credentials = Credentials.from_service_account_file(
            CREDENTIALS_FILE, 
            scopes=SCOPES
        )
    
    return gspread.authorize(credentials)

def sort_mixed_column(values):
    """Sort a column with mixed int/str types."""
    unique_vals = list(values.dropna().unique())
    try:
        return sorted(unique_vals, key=lambda x: (isinstance(x, str), int(x) if not isinstance(x, str) else float('inf'), str(x)))
    except (ValueError, TypeError):
        return sorted(unique_vals, key=str)

@st.cache_data(ttl=300)
def load_data():
    """Load workout data from Google Sheets."""
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
    """Generate a unique key for an exercise."""
    return f"{week}_{day}_{section}_{exercise}"

def parse_set_value(val):
    """Parse a set value to integer."""
    if pd.isna(val):
        return 0
    try:
        return int(float(str(val).strip()))
    except (ValueError, TypeError):
        return 0

def calculate_total_reps(row):
    """Calculate total reps from Set1-Set5 columns."""
    total = 0
    for i in range(1, 6):
        col = f'Set{i}'
        if col in row.index:
            total += parse_set_value(row[col])
    return total

def calculate_session_stats(row):
    """Calculate total reps and best single set for a session."""
    set_values = []
    for i in range(1, 6):
        val = parse_set_value(row.get(f'Set{i}', 0))
        set_values.append(val)
    
    total_reps = sum(set_values)
    best_single_set = max(set_values) if set_values else 0
    
    return total_reps, best_single_set

def parse_done(val):
    """Parse Done column value to boolean."""
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
    """Parse LastSaved column to date object."""
    if pd.isna(val) or val is None:
        return None
    try:
        val_str = str(val).strip()
        if val_str in ['', 'nan', 'None']:
            return None
        # Parse "YYYY-MM-DD HH:MM:SS" format
        dt = datetime.strptime(val_str[:10], "%Y-%m-%d")
        return dt.date()
    except (ValueError, TypeError):
        return None

def save_data(updates: list):
    """Save updates back to Google Sheets with timestamp for ALL exercises that day."""
    if not updates:
        return
    
    client = get_gspread_client()
    sheet = client.open_by_key(SHEET_ID).sheet1
    
    all_data = sheet.get_all_values()
    headers = all_data[0]
    col_map = {name: idx + 1 for idx, name in enumerate(headers)}
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Get Week and Day from first update for the second pass
    target_week = str(updates[0]['Week']).strip()
    target_day = str(updates[0]['Day']).strip()
    
    # Track which rows we've already updated
    updated_rows = set()
    
    # First pass: update exercises with data
    for update in updates:
        for row_idx, row in enumerate(all_data[1:], start=2):
            row_dict = dict(zip(headers, row))
            
            week_match = str(row_dict.get('Week', '')).strip() == str(update['Week']).strip()
            day_match = str(row_dict.get('Day', '')).strip() == str(update['Day']).strip()
            section_match = str(row_dict.get('Section', '')).strip() == str(update['Section']).strip()
            exercise_match = str(row_dict.get('Exercise', '')).strip() == str(update['Exercise']).strip()
            
            if week_match and day_match and section_match and exercise_match:
                cells_to_update = []
                
                for i, reps_value in enumerate(update['sets'], start=1):
                    if i <= 5:
                        col_name = f'Set{i}'
                        if col_name in col_map:
                            cell_value = reps_value if reps_value > 0 else ''
                            cells_to_update.append({'row': row_idx, 'col': col_map[col_name], 'value': cell_value})
                
                if COL_LOAD_VAR in col_map:
                    cells_to_update.append({'row': row_idx, 'col': col_map[COL_LOAD_VAR], 'value': update['load_variation']})
                
                if COL_AVG_RIR in col_map:
                    cells_to_update.append({'row': row_idx, 'col': col_map[COL_AVG_RIR], 'value': update['avg_rir'] if update['avg_rir'] > 0 else ''})
                
                if COL_DONE in col_map:
                    cells_to_update.append({'row': row_idx, 'col': col_map[COL_DONE], 'value': 'TRUE' if update['done'] else 'FALSE'})
                
                if 'LastSaved' in col_map:
                    cells_to_update.append({'row': row_idx, 'col': col_map['LastSaved'], 'value': timestamp})
                
                for cell in cells_to_update:
                    sheet.update_cell(cell['row'], cell['col'], cell['value'])
                
                updated_rows.add(row_idx)
                break
    
    # Second pass: update LastSaved for ALL remaining rows on the same Week/Day
    if 'LastSaved' in col_map:
        for row_idx, row in enumerate(all_data[1:], start=2):
            if row_idx in updated_rows:
                continue  # Already updated
            
            row_dict = dict(zip(headers, row))
            week_match = str(row_dict.get('Week', '')).strip() == target_week
            day_match = str(row_dict.get('Day', '')).strip() == target_day
            
            if week_match and day_match:
                sheet.update_cell(row_idx, col_map['LastSaved'], timestamp)

def find_next_workout(df):
    """Find the next workout to do based on completed data."""
    all_weeks = sorted(df['Week'].unique())
    
    all_workouts = []
    for week in all_weeks:
        week_df = df[df['Week'] == week]
        days = sort_mixed_column(week_df['Day'])
        for day in days:
            all_workouts.append((week, day))
    
    if not all_workouts:
        return (1, "1")
    
    df['IsDone'] = df[COL_DONE].apply(parse_done)
    done_df = df[df['IsDone'] == True]
    
    if done_df.empty:
        return all_workouts[0]
    
    last_done_week = done_df['Week'].max()
    last_week_done_df = done_df[done_df['Week'] == last_done_week]
    last_done_days = last_week_done_df['Day'].unique()
    sorted_days = sort_mixed_column(pd.Series(last_done_days))
    last_done_day = sorted_days[-1] if sorted_days else "1"
    last_done_tuple = (last_done_week, str(last_done_day))
    
    try:
        last_idx = None
        for i, (w, d) in enumerate(all_workouts):
            if w == last_done_week and str(d) == str(last_done_day):
                last_idx = i
                break
        
        if last_idx is not None and last_idx < len(all_workouts) - 1:
            return all_workouts[last_idx + 1]
        else:
            return last_done_tuple
    except (ValueError, IndexError):
        return all_workouts[0]


def get_calendar_data(df):
    """Extract calendar data from LastSaved column."""
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

# --- Mode Toggle ---
app_mode = st.radio(
    "Select Mode",
    options=["🏋️ Tracker", "📅 Calendar", "🏆 Records"],
    horizontal=True,
    label_visibility="collapsed"
)

st.divider()

try:
    df = load_data()
    
    if app_mode == "🏋️ Tracker":
        # ========================================
        # TRACKER VIEW
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
                section_df = day_df[day_df['Section'] == section]
                for idx, row in section_df.iterrows():
                    exercises_list.append({'section': section, 'row': row, 'df_idx': idx})
            
            total_exercises = len(exercises_list)
            current_idx = st.session_state.current_exercise_idx
            
            if current_idx >= total_exercises:
                current_idx = total_exercises - 1
                st.session_state.current_exercise_idx = current_idx
            if current_idx < 0:
                current_idx = 0
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
                    set_col = f'Set{i}'
                    if set_col in row.index:
                        set_val = parse_set_value(row.get(set_col))
                        existing_sets[f'set_{i}'] = set_val
                
                existing_load_var = ''
                if COL_LOAD_VAR in row.index:
                    load_val = row.get(COL_LOAD_VAR)
                    if pd.notna(load_val) and str(load_val).strip() not in ['', 'nan', 'None']:
                        existing_load_var = str(load_val)
                
                existing_avg_rir = 0
                if COL_AVG_RIR in row.index:
                    rir_val = row.get(COL_AVG_RIR)
                    if pd.notna(rir_val):
                        try:
                            existing_avg_rir = int(float(rir_val))
                        except (ValueError, TypeError):
                            existing_avg_rir = 0
                
                st.session_state.exercise_inputs[exercise_key] = {
                    'sets': existing_sets,
                    'load_variation': existing_load_var,
                    'avg_rir': existing_avg_rir
                }
            
            st.markdown(f"#### 📌 {section}")
            
            exercise_name = row['Exercise'] if pd.notna(row['Exercise']) else "Exercise"
            st.subheader(exercise_name)
            
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
                if str(val).strip() in ['', 'nan', 'None', '-']:
                    return default
                return str(val)
            
            # --- Target RIR and Rest Info Box ---
            target_rir = safe_display(row.get('RIR'))
            rest_val = safe_display(row.get(COL_REST))
            st.markdown(f'''
                <div class="target-box">
                    <span class="target-text">🎯 Target RIR: {target_rir}  |  ⏱ Rest: {rest_val}</span>
                </div>
            ''', unsafe_allow_html=True)
            
            # Target Sets/Reps using native Streamlit metrics
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
            
            base_sets = parse_sets_for_count(row.get('Sets'))
            stored_inputs = st.session_state.exercise_inputs[exercise_key]
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
                        f"Reps for Set {set_num}", 
                        min_value=0, 
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
                "Average RIR (0-5)",
                min_value=0,
                max_value=5,
                value=stored_inputs['avg_rir'],
                key=f"{exercise_key}_avg_rir",
                help="Enter your average RIR across all sets"
            )
            stored_inputs['avg_rir'] = avg_rir
            
            load_variation = st.text_input(
                "Load / Variation Used",
                value=stored_inputs['load_variation'],
                key=f"{exercise_key}_load_var",
                placeholder="e.g., Drop set, Pause reps..."
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
                            sets_data = []
                            for set_key in sorted(ex_inputs['sets'].keys()):
                                sets_data.append(ex_inputs['sets'][set_key])
                            
                            is_done = any(reps > 0 for reps in sets_data)
                            
                            all_exercise_updates.append({
                                'Week': selected_week,
                                'Day': selected_day,
                                'Section': ex['section'],
                                'Exercise': ex['row']['Exercise'],
                                'sets': sets_data,
                                'load_variation': ex_inputs['load_variation'],
                                'avg_rir': ex_inputs['avg_rir'],
                                'done': is_done
                            })
                    
                    st.toast('Saving...', icon='⏳')
                    
                    with st.spinner('💾 Saving your workout to Google Sheets...'):
                        save_data(all_exercise_updates)
                    
                    st.success("✅ Workout saved successfully!")
                    st.balloons()
                    st.cache_data.clear()
                    
                except gspread.exceptions.APIError as e:
                    st.error("❌ Could not connect to Google Sheets. Please check your internet connection and try again.")
                    st.caption(f"Technical details: {str(e)}")
                except Exception as e:
                    st.error("❌ Something went wrong while saving. Please try again.")
                    st.caption(f"Error: {str(e)}")
    
    elif app_mode == "📅 Calendar":
        # ========================================
        # CALENDAR VIEW
        # ========================================
        
        st.header("📅 Workout Calendar")
        
        # Initialize session state for calendar
        if 'calendar_year' not in st.session_state:
            st.session_state.calendar_year = date.today().year
        if 'calendar_month' not in st.session_state:
            st.session_state.calendar_month = date.today().month
        if 'selected_calendar_date' not in st.session_state:
            st.session_state.selected_calendar_date = None
        
        # Get calendar data from LastSaved column
        calendar_data = get_calendar_data(df)
        
        # --- PART A: CALENDAR GRID ---
        st.subheader("📆 Monthly View")
        
        # Month navigation
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
            st.markdown(f"<h3 style='text-align: center;'>{month_name} {st.session_state.calendar_year}</h3>", unsafe_allow_html=True)
        with nav_col3:
            if st.button("▶", key="next_month", use_container_width=True):
                if st.session_state.calendar_month == 12:
                    st.session_state.calendar_month = 1
                    st.session_state.calendar_year += 1
                else:
                    st.session_state.calendar_month += 1
                st.rerun()
        
        # Day headers
        day_headers = st.columns(7)
        day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        for i, header in enumerate(day_headers):
            with header:
                st.markdown(f"<div style='text-align:center; color:#a0a0a0; font-weight:bold;'>{day_names[i]}</div>", unsafe_allow_html=True)
        
        # Get calendar for current month
        cal = calendar.Calendar(firstweekday=0)
        month_days = cal.monthdayscalendar(st.session_state.calendar_year, st.session_state.calendar_month)
        today = date.today()
        
        # Render calendar grid
        for week in month_days:
            cols = st.columns(7)
            for i, day in enumerate(week):
                with cols[i]:
                    if day == 0:
                        st.markdown("<div style='height:50px;'></div>", unsafe_allow_html=True)
                    else:
                        current_date = date(st.session_state.calendar_year, st.session_state.calendar_month, day)
                        date_str = current_date.isoformat()
                        
                        # Determine status indicator
                        status_indicator = ""
                        if date_str in calendar_data:
                            data = calendar_data[date_str]
                            if data['total'] > 0 and data['done'] == data['total']:
                                status_indicator = "\n🟢"
                            elif data['total'] > 0:
                                status_indicator = "\n🔴"
                        
                        # Check if selected
                        is_selected = st.session_state.selected_calendar_date == date_str
                        
                        # Build button label with indicator inside
                        if is_selected:
                            button_label = f"*{day}*{status_indicator}"
                        else:
                            button_label = f"{day}{status_indicator}"
                        
                        if st.button(button_label, key=f"cal_{date_str}", use_container_width=True):
                            st.session_state.selected_calendar_date = date_str
                            st.rerun()
        
        # Legend
        st.markdown("""
        <div style='display:flex; gap:20px; justify-content:center; margin-top:10px; font-size:0.8rem;'>
            <span>🟢 All exercises completed</span>
            <span>🔴 Partially completed</span>
        </div>
        """, unsafe_allow_html=True)
        
        st.divider()
        
        # --- PART B: DAY DETAIL PANEL ---
        st.subheader("📋 Day Details")
        
        if st.session_state.selected_calendar_date:
            selected_date = st.session_state.selected_calendar_date
            st.info(f"📅 Showing workouts for: **{selected_date}**")
            
            # Filter exercises for selected date
            if 'LastSaved' in df.columns:
                day_exercises = df[df['LastSaved'].apply(lambda x: parse_last_saved_date(x) == date.fromisoformat(selected_date) if pd.notna(x) else False)]
            else:
                day_exercises = pd.DataFrame()
            
            if day_exercises.empty:
                st.warning("No workout data for this date.")
            else:
                # Group by Section
                for section in day_exercises['Section'].unique():
                    section_df = day_exercises[day_exercises['Section'] == section]
                    
                    with st.expander(f"📌 {section} ({len(section_df)} exercises)", expanded=True):
                        for _, ex_row in section_df.iterrows():
                            exercise_name = ex_row.get('Exercise', 'Unknown')
                            is_done = parse_done(ex_row.get(COL_DONE))
                            done_icon = "✅" if is_done else "⬜"
                            
                            st.markdown(f"**{exercise_name}** {done_icon}")
                            
                            # Show sets
                            sets_display = []
                            for i in range(1, 6):
                                set_val = parse_set_value(ex_row.get(f'Set{i}', 0))
                                if set_val > 0:
                                    sets_display.append(f"Set {i}: {set_val}")
                            
                            if sets_display:
                                st.caption(" | ".join(sets_display))
                            
                            # Show Load/Variation
                            load_var = ex_row.get(COL_LOAD_VAR)
                            if pd.notna(load_var) and str(load_var).strip() not in ['', 'nan', 'None']:
                                st.caption(f"🔧 Load/Variation: {load_var}")
                            
                            # Show Avg RIR
                            avg_rir = ex_row.get(COL_AVG_RIR)
                            if pd.notna(avg_rir):
                                try:
                                    st.caption(f"🎯 Avg RIR: {int(float(avg_rir))}")
                                except:
                                    pass
                            
                            st.markdown("---")
        else:
            st.info("👆 Click on a date above to see workout details.")
        
        st.divider()
        
        # --- PART C: PROGRESS CHARTS ---
        st.subheader("📊 Progress Overview")
        
        # Filters
        filter_col1, filter_col2 = st.columns(2)
        with filter_col1:
            week_options = ["All Weeks"] + [str(w) for w in sort_mixed_column(df['Week'])]
            selected_week_filter = st.selectbox("📅 Filter by Week", week_options, key="calendar_progress_week")
        with filter_col2:
            day_options = ["All Days"] + [str(d) for d in sort_mixed_column(df['Day'])]
            selected_day_filter = st.selectbox("📆 Filter by Day", day_options, key="calendar_progress_day")
        
        filtered_df = df.copy()
        if selected_week_filter != "All Weeks":
            filtered_df = filtered_df[filtered_df['Week'].astype(str) == selected_week_filter]
        if selected_day_filter != "All Days":
            filtered_df = filtered_df[filtered_df['Day'].astype(str) == selected_day_filter]
        
        filtered_df['TotalReps'] = filtered_df.apply(calculate_total_reps, axis=1)
        filtered_df['IsDone'] = filtered_df[COL_DONE].apply(parse_done)
        
        # Completion Scorecard
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
            st.metric(label="Exercises", value=f"{int(done_exercises)}/{total_exercises}", delta=f"{completion_pct:.0f}%")
        with metric_col2:
            days_pct = (days_completed / total_days * 100) if total_days > 0 else 0
            st.metric(label="Days", value=f"{days_completed}/{total_days}", delta=f"{days_pct:.0f}%")
        with metric_col3:
            weeks_pct = (weeks_active / total_weeks * 100) if total_weeks > 0 else 0
            st.metric(label="Weeks", value=f"{weeks_active}/{total_weeks}", delta=f"{weeks_pct:.0f}%")
        
        st.divider()
        
        # Progress Over Weeks by SECTION
        st.subheader("📈 Progress by Section")
        
        df_for_trends = df.copy()
        if selected_day_filter != "All Days":
            df_for_trends = df_for_trends[df_for_trends['Day'].astype(str) == selected_day_filter]
        
        df_for_trends['TotalReps'] = df_for_trends.apply(calculate_total_reps, axis=1)
        
        # Group by Week and SECTION instead of Exercise
        section_progress = df_for_trends[df_for_trends['TotalReps'] > 0].groupby(
            ['Week', 'Section']
        )['TotalReps'].sum().reset_index()
        
        if not section_progress.empty:
            section_progress['WeekSort'] = section_progress['Week'].apply(
                lambda x: (0, int(x)) if not isinstance(x, str) else (1, 0)
            )
            section_progress = section_progress.sort_values('WeekSort')
            section_progress['Week'] = section_progress['Week'].astype(str)
            
            fig_line = px.line(
                section_progress,
                x='Week',
                y='TotalReps',
                color='Section',
                markers=True,
                title=''
            )
            fig_line.update_layout(
                paper_bgcolor='#ffffff',
                plot_bgcolor='#f8f9fa',
                font=dict(color='#333333'),
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
            rir_by_week['WeekSort'] = rir_by_week['Week'].apply(
                lambda x: (0, int(x)) if not isinstance(x, str) else (1, 0)
            )
            rir_by_week = rir_by_week.sort_values('WeekSort')
            rir_by_week['Week'] = rir_by_week['Week'].astype(str)
            
            fig_rir = px.line(rir_by_week, x='Week', y='AvgRIR', markers=True, title='')
            fig_rir.update_traces(line_color='#ff6b35', marker_color='#ff6b35')
            fig_rir.update_layout(
                paper_bgcolor='#ffffff',
                plot_bgcolor='#f8f9fa',
                font=dict(color='#333333'),
                xaxis=dict(gridcolor='#e0e0e0', linecolor='#cccccc', title='Week', title_font=dict(color='#333333'), tickfont=dict(color='#333333')),
                yaxis=dict(gridcolor='#e0e0e0', linecolor='#cccccc', title='Average RIR', range=[0, 5], title_font=dict(color='#333333'), tickfont=dict(color='#333333')),
                height=300
            )
            st.plotly_chart(fig_rir, use_container_width=True)
        else:
            st.info("No RIR data recorded yet.")
        
        st.divider()
        
        # Completion Table
        st.subheader("📋 Exercise Summary")
        
        def count_sets_logged(row):
            count = 0
            for i in range(1, 6):
                if parse_set_value(row.get(f'Set{i}', 0)) > 0:
                    count += 1
            return count
        
        filtered_df['SetsLogged'] = filtered_df.apply(count_sets_logged, axis=1)
        
        summary_df = filtered_df.groupby(['Exercise', 'Section']).agg({
            'IsDone': 'sum',
            'SetsLogged': 'sum'
        }).reset_index()
        summary_df.columns = ['Exercise', 'Section', 'Times Completed', 'Total Sets Logged']
        summary_df = summary_df.sort_values('Times Completed', ascending=False)
        
        st.dataframe(
            summary_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Exercise": st.column_config.TextColumn("Exercise", width="large"),
                "Section": st.column_config.TextColumn("Section", width="medium"),
                "Times Completed": st.column_config.NumberColumn("✅ Done", format="%d"),
                "Total Sets Logged": st.column_config.NumberColumn("🔢 Sets", format="%d")
            }
        )
    
    elif app_mode == "🏆 Records":
        # ========================================
        # PERSONAL RECORDS VIEW
        # ========================================
        
        st.header("🏆 Personal Records")
        
        records_df = df.copy()
        records_df['TotalReps'] = records_df.apply(lambda row: calculate_session_stats(row)[0], axis=1)
        records_df['BestSingleSet'] = records_df.apply(lambda row: calculate_session_stats(row)[1], axis=1)
        records_df = records_df[records_df['TotalReps'] > 0]
        
        if records_df.empty:
            st.info("🏋️ No workout data recorded yet. Complete some exercises to see your Personal Records!")
        else:
            # SECTION 1: RECENT PRs
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
                            new_prs.append({
                                'exercise': exercise,
                                'curr_total': curr_total,
                                'prev_total': prev_total,
                                'curr_best_set': curr_best_set,
                                'prev_best_set': prev_best_set,
                                'is_total_pr': is_total_pr,
                                'is_set_pr': is_set_pr
                            })
                    else:
                        new_prs.append({
                            'exercise': exercise,
                            'curr_total': curr_total,
                            'prev_total': 0,
                            'curr_best_set': curr_best_set,
                            'prev_best_set': 0,
                            'is_total_pr': True,
                            'is_set_pr': True
                        })
            
            if new_prs:
                for pr in new_prs:
                    pr_details = []
                    if pr['is_total_pr']:
                        pr_details.append(f"Total: {pr['curr_total']} reps (+{pr['curr_total'] - pr['prev_total']})")
                    if pr['is_set_pr']:
                        pr_details.append(f"Best Set: {pr['curr_best_set']} reps")
                    st.success(f"🏆 **NEW PR!** {pr['exercise']} — {' | '.join(pr_details)}")
            else:
                st.info("💪 No new PRs this week yet. Keep pushing! Your next PR is just around the corner.")
            
            st.divider()
            
            # SECTION 2: ALL TIME RECORDS TABLE
            st.subheader("📋 All-Time Records")
            
            all_time_records = []
            for exercise in records_df['Exercise'].unique():
                ex_df = records_df[records_df['Exercise'] == exercise]
                best_total_idx = ex_df['TotalReps'].idxmax()
                best_total_row = ex_df.loc[best_total_idx]
                best_single_set = ex_df['BestSingleSet'].max()
                
                all_time_records.append({
                    'Exercise': exercise,
                    'Section': best_total_row.get('Section', '-'),
                    '🏆 Best Total Reps': int(best_total_row['TotalReps']),
                    '💪 Best Single Set': int(best_single_set),
                    '📅 Week Achieved': int(best_total_row['Week']),
                    '📆 Day Achieved': str(best_total_row['Day'])
                })
            
            records_table = pd.DataFrame(all_time_records)
            records_table = records_table.sort_values(['Section', 'Exercise'])
            
            st.dataframe(
                records_table,
                use_container_width=True,
                hide_index=True,
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
            
            # SECTION 3: PR BAR CHART
            st.subheader("📊 Best Total Reps by Exercise")
            
            pr_chart_data = records_table[['Exercise', '🏆 Best Total Reps']].copy()
            pr_chart_data = pr_chart_data.sort_values('🏆 Best Total Reps', ascending=True)
            
            fig_pr = px.bar(
                pr_chart_data,
                x='🏆 Best Total Reps',
                y='Exercise',
                orientation='h',
                title='',
                color_discrete_sequence=['#00cc66']
            )
            fig_pr.update_layout(
                paper_bgcolor='#ffffff',
                plot_bgcolor='#f8f9fa',
                font=dict(color='#333333'),
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