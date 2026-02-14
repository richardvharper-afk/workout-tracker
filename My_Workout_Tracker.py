import streamlit as st
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

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
    """Create and return a gspread client. 
    Works both on Streamlit Cloud (using secrets) and locally (using JSON file).
    """
    import os
    
    try:
        # First try Streamlit Cloud secrets
        creds_dict = st.secrets["gcp_service_account"]
        credentials = Credentials.from_service_account_info(
            dict(creds_dict), 
            scopes=SCOPES
        )
    except (KeyError, FileNotFoundError):
        # Fallback to local credentials file
        # Set SSL certs for Zscaler proxy (local environment)
        os.environ['SSL_CERT_FILE'] = r'C:\Users\CP362988\source\repos\My Workout Tracket\zscaler.pem.cer'
        os.environ['REQUESTS_CA_BUNDLE'] = r'C:\Users\CP362988\source\repos\My Workout Tracket\zscaler.pem.cer'
        
        credentials = Credentials.from_service_account_file(
            CREDENTIALS_FILE, 
            scopes=SCOPES
        )
    
    return gspread.authorize(credentials)

def sort_mixed_column(values):
    """Sort a column with mixed int/str types by converting all to strings."""
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
    
    # Keep Sets, Reps, RIR as strings (they may contain "8-10", "2-3" etc)
    # Do NOT convert these to numeric
    text_cols = ['Sets', 'Reps', 'RIR', COL_REST]
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).replace('', '-').replace('nan', '-')
    
    # Convert only the actual numeric columns
    numeric_cols = ['Set1', 'Set2', 'Set3', 'Set4', 'Set5', COL_AVG_RIR]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Week should be integer not float
    if 'Week' in df.columns:
        df['Week'] = pd.to_numeric(df['Week'], errors='coerce').fillna(0).astype(int)

    # Day should stay as text string
    if 'Day' in df.columns:
        df['Day'] = df['Day'].astype(str).str.strip()
    
    # Handle empty strings as NaN for non-text columns only
    for col in df.columns:
        if col not in text_cols:
            df[col] = df[col].replace('', pd.NA)
    
    return df

def get_exercise_key(week, day, section, exercise):
    """Generate a unique key for an exercise."""
    return f"{week}_{day}_{section}_{exercise}"

def parse_set_value(val):
    """Parse a set value to integer, handling various formats."""
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

def parse_done(val):
    """Parse Done column value to boolean, handling various formats."""
    if pd.isna(val):
        return False
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return val == 1 or val == 1.0
    if isinstance(val, str):
        return val.strip().upper() in ['TRUE', '1', 'YES', 'Y']
    return False

def save_data(updates: list):
    """Save updates back to Google Sheets with timestamp."""
    client = get_gspread_client()
    sheet = client.open_by_key(SHEET_ID).sheet1
    
    # Get all data including headers
    all_data = sheet.get_all_values()
    headers = all_data[0]
    
    # Create column name to index mapping (1-based for gspread)
    col_map = {name: idx + 1 for idx, name in enumerate(headers)}
    
    # Get current timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    for update in updates:
        # Find the matching row
        for row_idx, row in enumerate(all_data[1:], start=2):  # Start from 2 (1-based, skip header)
            row_dict = dict(zip(headers, row))
            
            week_match = str(row_dict.get('Week', '')).strip() == str(update['Week']).strip()
            day_match = str(row_dict.get('Day', '')).strip() == str(update['Day']).strip()
            section_match = str(row_dict.get('Section', '')).strip() == str(update['Section']).strip()
            exercise_match = str(row_dict.get('Exercise', '')).strip() == str(update['Exercise']).strip()
            
            if week_match and day_match and section_match and exercise_match:
                # Build list of cell updates for this row
                cells_to_update = []
                
                # Update Set columns
                for i, reps_value in enumerate(update['sets'], start=1):
                    if i <= 5:
                        col_name = f'Set{i}'
                        if col_name in col_map:
                            cell_value = reps_value if reps_value > 0 else ''
                            cells_to_update.append({
                                'row': row_idx,
                                'col': col_map[col_name],
                                'value': cell_value
                            })
                
                # Update Load / Variation Used
                if COL_LOAD_VAR in col_map:
                    cells_to_update.append({
                        'row': row_idx,
                        'col': col_map[COL_LOAD_VAR],
                        'value': update['load_variation']
                    })
                
                # Update Avg RIR
                if COL_AVG_RIR in col_map:
                    cells_to_update.append({
                        'row': row_idx,
                        'col': col_map[COL_AVG_RIR],
                        'value': update['avg_rir'] if update['avg_rir'] > 0 else ''
                    })
                
                # Update Done column
                if COL_DONE in col_map:
                    cells_to_update.append({
                        'row': row_idx,
                        'col': col_map[COL_DONE],
                        'value': 'TRUE' if update['done'] else 'FALSE'
                    })
                
                # Update LastSaved timestamp (if column exists)
                if 'LastSaved' in col_map:
                    cells_to_update.append({
                        'row': row_idx,
                        'col': col_map['LastSaved'],
                        'value': timestamp
                    })
                
                # Batch update cells for this row
                for cell in cells_to_update:
                    sheet.update_cell(cell['row'], cell['col'], cell['value'])
                
                break  # Found and updated the row, move to next update


# --- Find Next Workout ---
def find_next_workout(df):
    """Find the next workout to do based on completed data.
    Returns (default_week, default_day) tuple.
    """
    # Get all unique weeks and days, sorted
    all_weeks = sorted(df['Week'].unique())
    
    # Build ordered list of all (week, day) combinations
    all_workouts = []
    for week in all_weeks:
        week_df = df[df['Week'] == week]
        days = sort_mixed_column(week_df['Day'])
        for day in days:
            all_workouts.append((week, day))
    
    if not all_workouts:
        return (1, "1")  # Fallback
    
    # Find the last completed workout
    df['IsDone'] = df[COL_DONE].apply(parse_done)
    done_df = df[df['IsDone'] == True]
    
    if done_df.empty:
        # No workouts done yet, return first workout
        return all_workouts[0]
    
    # Find the highest week with completed exercises
    last_done_week = done_df['Week'].max()
    last_week_done_df = done_df[done_df['Week'] == last_done_week]
    
    # Find the highest day in that week with completed exercises
    last_done_days = last_week_done_df['Day'].unique()
    # Sort days and get the last one
    sorted_days = sort_mixed_column(pd.Series(last_done_days))
    last_done_day = sorted_days[-1] if sorted_days else "1"
    
    # Find index of last completed workout
    last_done_tuple = (last_done_week, str(last_done_day))
    
    try:
        # Find position in ordered list
        last_idx = None
        for i, (w, d) in enumerate(all_workouts):
            if w == last_done_week and str(d) == str(last_done_day):
                last_idx = i
                break
        
        if last_idx is not None and last_idx < len(all_workouts) - 1:
            # Return next workout
            return all_workouts[last_idx + 1]
        else:
            # Already at the end, return the last workout (current)
            return last_done_tuple
    except (ValueError, IndexError):
        return all_workouts[0]


# --- Main App ---
st.title("💪 Workout Tracker 2026")

# --- Mode Toggle ---
app_mode = st.radio(
    "Select Mode",
    options=["🏋️ Tracker", "📊 Progress"],
    horizontal=True,
    label_visibility="collapsed"
)

st.divider()

try:
    df = load_data()
    
    if app_mode == "🏋️ Tracker":
        # ========================================
        # TRACKER VIEW (existing functionality)
        # ========================================
        
        # Initialize session state
        if 'extra_sets' not in st.session_state:
            st.session_state.extra_sets = {}
        if 'exercise_inputs' not in st.session_state:
            st.session_state.exercise_inputs = {}
        if 'current_exercise_idx' not in st.session_state:
            st.session_state.current_exercise_idx = 0
        if 'last_week_day' not in st.session_state:
            st.session_state.last_week_day = None
        
        # --- Week and Day Selection with Smart Defaults ---
        # Find the next workout to do
        default_week, default_day = find_next_workout(df)
        
        col1, col2 = st.columns(2)
        with col1:
            weeks = sort_mixed_column(df['Week'])
            # Find index of default week
            try:
                default_week_idx = list(weeks).index(default_week)
            except ValueError:
                default_week_idx = 0
            
            selected_week = st.selectbox(
                "📅 Select Week", 
                weeks, 
                index=default_week_idx,
                key="week_select"
            )
        
        with col2:
            filtered_df = df[df['Week'].astype(str) == str(selected_week)]
            days = sort_mixed_column(filtered_df['Day'])
            
            # Find index of default day (only if we're on the default week)
            if selected_week == default_week:
                try:
                    default_day_idx = [str(d) for d in days].index(str(default_day))
                except ValueError:
                    default_day_idx = 0
            else:
                default_day_idx = 0
            
            selected_day = st.selectbox(
                "📆 Select Day", 
                days, 
                index=default_day_idx,
                key="day_select"
            )
        
        # Reset exercise index when week/day changes
        current_week_day = f"{selected_week}_{selected_day}"
        if st.session_state.last_week_day != current_week_day:
            st.session_state.current_exercise_idx = 0
            st.session_state.last_week_day = current_week_day
        
        # Filter data for selected week and day
        day_df = df[(df['Week'].astype(str) == str(selected_week)) & 
                    (df['Day'].astype(str) == str(selected_day))]
        
        if day_df.empty:
            st.warning("No workouts found for this selection.")
        else:
            # --- Workout Type Header ---
            workout_type = day_df['Type'].iloc[0] if pd.notna(day_df['Type'].iloc[0]) else "Workout"
            st.markdown(f"## {workout_type}")
            
            # Build list of all exercises for this day
            exercises_list = []
            for section in day_df['Section'].unique():
                section_df = day_df[day_df['Section'] == section]
                for idx, row in section_df.iterrows():
                    exercises_list.append({
                        'section': section,
                        'row': row,
                        'df_idx': idx
                    })
            
            total_exercises = len(exercises_list)
            current_idx = st.session_state.current_exercise_idx
            
            # Clamp index to valid range
            if current_idx >= total_exercises:
                current_idx = total_exercises - 1
                st.session_state.current_exercise_idx = current_idx
            if current_idx < 0:
                current_idx = 0
                st.session_state.current_exercise_idx = current_idx
            
            # --- Progress Indicator ---
            st.markdown(f'<div class="progress-indicator">Exercise {current_idx + 1} of {total_exercises}</div>', unsafe_allow_html=True)
            
            # --- Navigation Arrows ---
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
            
            # --- Current Exercise ---
            current_exercise = exercises_list[current_idx]
            section = current_exercise['section']
            row = current_exercise['row']
            exercise_key = get_exercise_key(selected_week, selected_day, section, row['Exercise'])
            
            # Initialize inputs for this exercise if not exists
            # Pre-populate with existing values from Google Sheets
            if exercise_key not in st.session_state.exercise_inputs:
                # Load existing Set values from the row
                existing_sets = {}
                for i in range(1, 6):
                    set_col = f'Set{i}'
                    if set_col in row.index:
                        set_val = parse_set_value(row.get(set_col))
                        existing_sets[f'set_{i}'] = set_val
                
                # Load existing Load/Variation
                existing_load_var = ''
                if COL_LOAD_VAR in row.index:
                    load_val = row.get(COL_LOAD_VAR)
                    if pd.notna(load_val) and str(load_val).strip() not in ['', 'nan', 'None']:
                        existing_load_var = str(load_val)
                
                # Load existing Avg RIR
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
            
            # Section Header
            st.markdown(f"#### 📌 {section}")
            
            # Exercise Name (prominent)
            exercise_name = row['Exercise'] if pd.notna(row['Exercise']) else "Exercise"
            st.subheader(exercise_name)

            
            # Description (smaller text)
            if pd.notna(row['Description']):
                st.caption(row['Description'])
            
            # Helper function to safely display values
            def safe_display(val, default="-"):
                """Return display-safe value, replacing NaN/empty with default."""
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
            
            # --- Target RIR and Rest Info Box (Red/Orange) ---
            target_rir = safe_display(row.get('RIR'))
            rest_val = safe_display(row.get(COL_REST))
            st.markdown(f'''
                <div class="target-box">
                    <span class="target-text">🎯 Target RIR: {target_rir}  |  ⏱ Rest: {rest_val}</span>
                </div>
            ''', unsafe_allow_html=True)
            
            
            # Target Info Box (Sets/Reps)
            target_sets = str(row['Sets']) if pd.notna(row['Sets']) and str(row['Sets']).strip() not in ['', 'nan', 'None', '0'] else '-'
            target_reps = str(row['Reps']) if pd.notna(row['Reps']) and str(row['Reps']).strip() not in ['', 'nan', 'None', '0'] else '-'

            
            # Target Info Box (Sets/Reps) - using native Streamlit
            info_cols = st.columns(2)
            with info_cols[0]:
                st.metric(label="🎯 Target Sets", value=target_sets)
            with info_cols[1]:
                st.metric(label="🔁 Target Reps", value=target_reps)

            # Escalation and Notes
            if pd.notna(row.get('Escalation')) and str(row['Escalation']).strip():
                st.info(f"📈 **Escalation:** {row['Escalation']}")
            if pd.notna(row.get('Notes')) and str(row['Notes']).strip():
                st.caption(f"📝 {row['Notes']}")
            
            # Determine number of sets (handle ranges like "3" or "3-4")
            def parse_sets_for_count(val):
                """Parse sets value to get a count for input fields."""
                val_str = str(val).strip()
                if val_str in ['', '-', 'nan', 'None']:
                    return 3  # Default
                # If it's a range like "3-4", take the first number
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
            
            # Also consider how many sets already have data
            stored_inputs = st.session_state.exercise_inputs[exercise_key]
            sets_with_data = len([k for k, v in stored_inputs['sets'].items() if v > 0])
            base_sets = max(base_sets, sets_with_data)
            
            extra = st.session_state.extra_sets.get(exercise_key, 0)
            total_sets = base_sets + extra
            
            # Set Input Rows - SIMPLIFIED (only Reps)
            st.markdown("**Log Your Sets:**")
            
            for set_num in range(1, total_sets + 1):
                set_key = f"set_{set_num}"
                
                # Initialize set data if not exists
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
            
            # Add Set Button
            if st.button("➕ Add Set", key=f"{exercise_key}_add_set"):
                st.session_state.extra_sets[exercise_key] = extra + 1
                st.rerun()
            
            st.markdown("---")
            
            # Average RIR Input (single input for whole exercise)
            avg_rir = st.number_input(
                "Average RIR (0-5)",
                min_value=0,
                max_value=5,
                value=stored_inputs['avg_rir'],
                key=f"{exercise_key}_avg_rir",
                help="Enter your average RIR across all sets"
            )
            stored_inputs['avg_rir'] = avg_rir
            
            # Load Variation Input
            load_variation = st.text_input(
                "Load / Variation Used",
                value=stored_inputs['load_variation'],
                key=f"{exercise_key}_load_var",
                placeholder="e.g., Drop set, Pause reps..."
            )
            stored_inputs['load_variation'] = load_variation
            
            # --- Save Button ---
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
                    
                    # Show progress feedback
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
    
    else:
        # ========================================
        # PROGRESS DASHBOARD VIEW
        # ========================================
        
        st.header("📊 Progress Dashboard")
        
        # --- Filters ---
        filter_col1, filter_col2 = st.columns(2)
        with filter_col1:
            week_options = ["All Weeks"] + [str(w) for w in sort_mixed_column(df['Week'])]
            selected_week_filter = st.selectbox("📅 Filter by Week", week_options, key="progress_week")
        with filter_col2:
            day_options = ["All Days"] + [str(d) for d in sort_mixed_column(df['Day'])]
            selected_day_filter = st.selectbox("📆 Filter by Day", day_options, key="progress_day")
        
        # Apply filters
        filtered_df = df.copy()
        if selected_week_filter != "All Weeks":
            filtered_df = filtered_df[filtered_df['Week'].astype(str) == selected_week_filter]
        if selected_day_filter != "All Days":
            filtered_df = filtered_df[filtered_df['Day'].astype(str) == selected_day_filter]
        
        # Calculate total reps for each row
        filtered_df['TotalReps'] = filtered_df.apply(calculate_total_reps, axis=1)
        
        # Apply parse_done to the Done column
        filtered_df['IsDone'] = filtered_df[COL_DONE].apply(parse_done)
        
        st.divider()
        
        # --- 1. COMPLETION SCORECARD ---
        st.subheader("✅ Completion Status")
        
        total_exercises = len(filtered_df)
        done_exercises = filtered_df['IsDone'].sum()
        completion_pct = (done_exercises / total_exercises * 100) if total_exercises > 0 else 0
        
        score_col1, score_col2 = st.columns(2)
        with score_col1:
            st.metric(
                label="Exercises Completed",
                value=f"{done_exercises} / {total_exercises}",
                delta=f"{completion_pct:.1f}%"
            )
        with score_col2:
            # Visual completion bar
            if completion_pct >= 80:
                bar_color = "#00cc66"
            elif completion_pct >= 50:
                bar_color = "#ffaa00"
            else:
                bar_color = "#ff4444"
            
            st.markdown(f'''
                <div style="background-color: #333; border-radius: 10px; padding: 3px; margin-top: 1rem;">
                    <div style="background-color: {bar_color}; width: {completion_pct}%; height: 30px; border-radius: 8px; 
                         display: flex; align-items: center; justify-content: center; color: white; font-weight: bold;">
                        {completion_pct:.0f}%
                    </div>
                </div>
            ''', unsafe_allow_html=True)
        
        st.divider()
        
        # --- 2. TOTAL REPS PER EXERCISE - BAR CHART ---
        st.subheader("📊 Total Reps per Exercise")
        
        # Group by exercise and sum reps
        reps_by_exercise = filtered_df[filtered_df['TotalReps'] > 0].groupby('Exercise')['TotalReps'].sum().reset_index()
        reps_by_exercise = reps_by_exercise.sort_values('TotalReps', ascending=True)
        
        if not reps_by_exercise.empty:
            fig_bar = px.bar(
                reps_by_exercise,
                x='TotalReps',
                y='Exercise',
                orientation='h',
                title='',
                color='TotalReps',
                color_continuous_scale='Blues'
            )
            fig_bar.update_layout(
                paper_bgcolor='#ffffff',
                plot_bgcolor='#f8f9fa',
                font=dict(color='#333333'),
                xaxis=dict(gridcolor='#e0e0e0', linecolor='#cccccc', title='Total Reps', title_font=dict(color='#333333'), tickfont=dict(color='#333333')),
                yaxis=dict(gridcolor='#e0e0e0', linecolor='#cccccc', title='', tickfont=dict(color='#333333')),
                coloraxis_showscale=False,
                height=max(400, len(reps_by_exercise) * 35)
            )
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("No rep data recorded yet.")
        
        st.divider()
        
        # --- 3. PROGRESS OVER WEEKS - LINE CHART ---
        st.subheader("📈 Progress Over Weeks")
        
        # Use full dataset for week trends (ignore week filter for this chart)
        df_for_trends = df.copy()
        if selected_day_filter != "All Days":
            df_for_trends = df_for_trends[df_for_trends['Day'].astype(str) == selected_day_filter]
        
        df_for_trends['TotalReps'] = df_for_trends.apply(calculate_total_reps, axis=1)
        
        # Group by week and exercise
        weekly_progress = df_for_trends[df_for_trends['TotalReps'] > 0].groupby(
            ['Week', 'Exercise']
        )['TotalReps'].sum().reset_index()
        
        if not weekly_progress.empty:
            # Sort weeks
            weekly_progress['WeekSort'] = weekly_progress['Week'].apply(
                lambda x: (0, int(x)) if not isinstance(x, str) else (1, 0)
            )
            weekly_progress = weekly_progress.sort_values('WeekSort')
            weekly_progress['Week'] = weekly_progress['Week'].astype(str)
            
            fig_line = px.line(
                weekly_progress,
                x='Week',
                y='TotalReps',
                color='Exercise',
                markers=True,
                title=''
            )
            fig_line.update_layout(
                paper_bgcolor='#ffffff',
                plot_bgcolor='#f8f9fa',
                font=dict(color='#333333'),
                xaxis=dict(gridcolor='#e0e0e0', linecolor='#cccccc', title='Week', title_font=dict(color='#333333'), tickfont=dict(color='#333333')),
                yaxis=dict(gridcolor='#e0e0e0', linecolor='#cccccc', title='Total Reps', title_font=dict(color='#333333'), tickfont=dict(color='#333333')),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=-0.5,
                    xanchor="center",
                    x=0.5,
                    font=dict(color='#333333')
                ),
                height=450
            )
            st.plotly_chart(fig_line, use_container_width=True)
        else:
            st.info("No weekly progress data yet.")
        
        st.divider()
        
        # --- 4. AVERAGE RIR TREND - LINE CHART ---
        st.subheader("🎯 Average RIR Trend")
        st.caption("Lower RIR = working harder 💪")
        
        # Parse Avg RIR column
        def parse_avg_rir(val):
            if pd.isna(val):
                return None
            try:
                return float(val)
            except (ValueError, TypeError):
                return None
        
        df_for_trends['AvgRIR'] = df_for_trends[COL_AVG_RIR].apply(parse_avg_rir)
        
        # Group by week and calculate mean RIR
        rir_by_week = df_for_trends[df_for_trends['AvgRIR'].notna()].groupby('Week')['AvgRIR'].mean().reset_index()
        
        if not rir_by_week.empty:
            rir_by_week['WeekSort'] = rir_by_week['Week'].apply(
                lambda x: (0, int(x)) if not isinstance(x, str) else (1, 0)
            )
            rir_by_week = rir_by_week.sort_values('WeekSort')
            rir_by_week['Week'] = rir_by_week['Week'].astype(str)
            
            fig_rir = px.line(
                rir_by_week,
                x='Week',
                y='AvgRIR',
                markers=True,
                title=''
            )
            fig_rir.update_traces(line_color='#ff6b35', marker_color='#ff6b35')
            fig_rir.update_layout(
                paper_bgcolor='#ffffff',
                plot_bgcolor='#f8f9fa',
                font=dict(color='#333333'),
                xaxis=dict(gridcolor='#e0e0e0', linecolor='#cccccc', title='Week', title_font=dict(color='#333333'), tickfont=dict(color='#333333')),
                yaxis=dict(gridcolor='#e0e0e0', linecolor='#cccccc', title='Average RIR', range=[0, 5], title_font=dict(color='#333333'), tickfont=dict(color='#333333')),
                height=350
            )
            st.plotly_chart(fig_rir, use_container_width=True)
        else:
            st.info("No RIR data recorded yet.")
        
        st.divider()
        
        # --- 5. COMPLETION TABLE ---
        st.subheader("📋 Exercise Completion Summary")
        
        # Calculate stats per exercise
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
        
        # Style the dataframe
        st.dataframe(
            summary_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Exercise": st.column_config.TextColumn("Exercise", width="large"),
                "Section": st.column_config.TextColumn("Section", width="medium"),
                "Times Completed": st.column_config.NumberColumn("✅ Completed", format="%d"),
                "Total Sets Logged": st.column_config.NumberColumn("🔢 Sets", format="%d")
            }
        )

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