from google import genai
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import numpy as np
import os
import re
import io
from collections import deque
from math import radians, sin, cos, sqrt, atan2

# --- Helper Function for Haversine Distance ---
def haversine_distance(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    r_metres = 6371000 # metres radius of the Earth
    distance = r_metres * c
    return distance

# ---  Function to Convert Negative Values in Specific Columns ---
def convert_negatives_to_half_positive(df: pd.DataFrame) -> pd.DataFrame:

    for col in df.columns:
        # Check if column is numeric AND its name ends with 'ppm' or 'pct' (case-insensitive)
        if pd.api.types.is_numeric_dtype(df[col]) and \
           (col.lower().endswith('ppm') or col.lower().endswith('pct')):
            # Apply the transformation: where value is negative, replace with abs(value)/2, else keep original
            df[col] = np.where((pd.notnull(df[col])) & (df[col] < 0), abs(df[col]) / 2, df[col])
    return df

def get_data_summary(df: pd.DataFrame) -> str:
    summary_stream = io.StringIO()
    df.info(buf=summary_stream)
    info_str = summary_stream.getvalue()
    lat_col_name = None
    lon_col_name = None
    lat_lon_info = ""
    if 'lat94' in df.columns: lat_col_name = 'lat94'
    elif any('lat' in col.lower() or col.lower() == 'y' for col in df.columns):
        possible_lat_cols = [col for col in df.columns if 'lat' in col.lower() or col.lower() == 'northing']
        lat_lon_info += f"\nPotential Latitude Columns (if 'lat94' not used): {possible_lat_cols}"
    if 'lng94' in df.columns: lon_col_name = 'lng94'
    elif any('lon' in col.lower() or 'long' in col.lower() or col.lower() == 'easting' for col in df.columns):
        possible_lon_cols = [col for col in df.columns if 'lon' in col.lower() or 'lng' in col.lower() or col.lower() == 'x']
        lat_lon_info += f"\nPotential Longitude Columns (if 'lng94' not used): {possible_lon_cols}"
    if lat_col_name and lon_col_name: lat_lon_info = f"\nConfirmed Latitude Column: '{lat_col_name}'\nConfirmed Longitude Column: '{lon_col_name}'"
    elif lat_col_name: lat_lon_info = f"\nConfirmed Latitude Column: '{lat_col_name}' (Longitude 'lng94' not found, please check other potential names if needed)."
    elif lon_col_name: lat_lon_info = f"\nConfirmed Longitude Column: '{lon_col_name}' (Latitude 'lat94' not found, please check other potential names if needed)."
    else: lat_lon_info += "\nSpecific 'lat94'/'lng94' columns not found. If other lat/lon columns exist, please specify their names in your query if needed."
    summary = f"""
DATA SUMMARY:
DataFrame Shape: {df.shape}
Column Information:
{info_str}{lat_lon_info}
First 5 rows:
{df.head().to_string()}
Basic Descriptive Statistics:
{df.describe(include='all').to_string()}
"""
    for col in df.select_dtypes(include=['object', 'category']).columns:
        if df[col].nunique() < 20: summary += f"\nUnique values in '{col}': {df[col].unique().tolist()}"
    return summary

def get_gemini_generated_code(
    user_question: str, data_summary_for_llm: str, conversation_history_str: str,
    api_key: str, model_name: str = "gemini-2.5-flash-preview-05-20"
) -> str:
    try: client = genai.Client(api_key=api_key)
    except Exception as e: print(f"# Error initialising Gemini Client: {e}"); return None
    prompt = f"""
You are a data analysis assistant. Please use AU/UK English spellings (e.g., colour, analyse) in your responses and generated plot titles and labels.
A pandas DataFrame named 'df' has been loaded with the following data:
{data_summary_for_llm}
CONVERSATION HISTORY (Previous questions and generated code):
{conversation_history_str if conversation_history_str else "No previous conversation."}
CURRENT USER QUESTION: "{user_question}"
Based on the data, conversation history, and the current user's question, generate Python code to perform the requested analysis or create the described plot.
The code should use the 'df' DataFrame.
Available Python libraries: pandas (pd), matplotlib.pyplot (plt), matplotlib.ticker (ticker), matplotlib.colors (as mcolors), seaborn (sns), numpy (np).
Helper functions `haversine_distance(lat1, lon1, lat2, lon2)` (returns distance in metres) and `convert_negatives_to_half_positive(df)` (modifies df to convert negative numbers in columns ending with 'ppm' or 'pct') are available.

Instructions for the generated code:
1.  If the question asks for a calculation or data subset, print the result or the head of the resulting DataFrame. If you create a new DataFrame for the subset (e.g., for outliers), assign it to a descriptive variable name like `outlier_df`.
2.  If the question asks for a plot, generate it. Ensure plots have titles and axis labels using UK English.
    IMPORTANT FOR PLOT AXES (NON-LOG): To ensure X and Y axes (like latitude and longitude) display full numerical values without offset notation,
    get the current axes using `ax = plt.gca()` and then apply formatters:
    `ax.xaxis.set_major_formatter(ticker.ScalarFormatter(useOffset=False))`
    `ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useOffset=False))`
    IMPORTANT FOR LOGARITHMIC SCALES (Plot Axes and Colour Bars):
    - When a logarithmic scale is requested for a plot axis (e.g., for 'Au' concentrations) or for a colour bar:
        1. **Prepare Data for Log Scale**:
           - Before applying the log scale, the data for the specific column being plotted (let's call it `target_column_name`) needs special handling.
           - Create a *temporary working copy* of the data from `df[target_column_name]` for plotting. Do NOT modify the original `df` DataFrame unless explicitly asked by the user to do so.
           - Example for creating and preparing the working copy (e.g., `data_for_log`):
             `data_for_log = df['target_column_name'].copy()`
           - **Step 1.1: Handle Zeros**: In `data_for_log`, replace all values equal to `0` with `0.05`.
             `data_for_log[data_for_log == 0] = 0.05`
           - **Step 1.2: Handle Specific Negatives**: In `data_for_log`, for values `x` where `-10 <= x < 0`, replace them with `abs(x) / 2`.
             `neg_condition = (data_for_log >= -10) & (data_for_log < 0)`
             `data_for_log[neg_condition] = data_for_log[neg_condition].abs() / 2`
           - **Step 1.3: Filter for Positivity**: After these transformations, the `data_for_log` used for the log scale *must* be strictly positive. Filter out any remaining non-positive values.
             `data_for_log = data_for_log[data_for_log > 0]`
           - If `data_for_log` is empty after these steps (e.g., all original values were <= -10 or became non-positive after transformation), inform the user (e.g., print a message) and consider not generating the log-scaled plot or that part of it.
        2. **Apply Logarithmic Transformation**:
           - For an axis: `ax.set_xscale('log')` or `ax.set_yscale('log')` (applied to the axis object, using the prepared `data_for_log`).
           - For a colour map: use `norm=mcolors.LogNorm(...)` (ensure `vmin` and `vmax` for `LogNorm` are derived from the prepared `data_for_log`).
        3. **Format Tick Labels**: Ensure tick labels display numerical values in a readable decimal format.
           - For plot axes (after setting scale to log):
             `ax.xaxis.set_major_formatter(ticker.ScalarFormatter())`
             `ax.yaxis.set_major_formatter(ticker.ScalarFormatter())`
             If `ScalarFormatter` results in scientific notation where decimal is preferred (e.g., for 0.001), use `ticker.FormatStrFormatter('%g')`:
             `ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%g'))`
           - For colour bar tick labels (after `cbar = plt.colorbar(...)`):
             `cbar.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2g'))` (Vertical bar)
             `cbar.ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2g'))` (Horizontal bar)
        4. **Labelling**: Ensure axis and colour bar labels clearly indicate the log scale and original units (e.g., "Gold (Au) (ppm, Log Scale)"). Use UK spelling.
3.  Consider the conversation history for context if the user asks a follow-up question.
4.  LOCATION-BASED QUERIES:
    - Parse target latitude, longitude, and radius (in metres) from the question.
    - Assume DataFrame has 'lat94' and 'lng94' if 'Confirmed' in DATA SUMMARY.
    - Create 'distance_to_target' column using `haversine_distance`.
    - Filter by radius. Example:
      target_lat = 140.7; target_lon = -28.9; radius_m = 300
      df['distance_to_target'] = df.apply(lambda row: haversine_distance(row['lat94'], row['lng94'], target_lat, target_lon), axis=1)
      nearby_samples = df[df['distance_to_target'] <= radius_m]
      print(f"Found {{len(nearby_samples)}} samples within {{radius_m}}m of ({{target_lat}}, {{target_lon}}):"); print(nearby_samples.head())
5.  Return ONLY the Python code, without any surrounding text, explanations, or markdown formatting.
"""
    try:
        response = client.models.generate_content(model=model_name, contents=prompt)
        generated_text = response.text
        code_match = re.search(r"```python\n(.*?)```", generated_text, re.DOTALL)
        if code_match: code = code_match.group(1).strip()
        else:
            cleaned_text = generated_text.strip()
            if cleaned_text.lower().startswith("python\n"): cleaned_text = cleaned_text[len("python\n"):].strip()
            code = cleaned_text
        # Remove haversine_distance or convert_negatives_to_half_positive if Gemini tries to redefine them
        code_lines = code.splitlines(); in_helper_def = False; new_code_lines = []
        helper_defs = ["def haversine_distance(", "def convert_negatives_to_half_positive("]
        for line in code_lines:
            stripped_line = line.strip()
            if any(stripped_line.startswith(h_def) for h_def in helper_defs):
                in_helper_def = True
                continue
            if in_helper_def:
                if not stripped_line or line.startswith(" ") or line.startswith("\t"):
                    continue # still inside the helper function body
                else:
                    in_helper_def = False # exited helper function body
            if not in_helper_def:
                new_code_lines.append(line)
        code = "\n".join(new_code_lines)
        return code
    except Exception as e: print(f"# Error communicating with Gemini API: {e}"); return None

def execute_generated_code(code_to_execute: str, df_data: pd.DataFrame):
    if not code_to_execute or code_to_execute.strip().startswith("#") or not code_to_execute.strip():
        print("No valid executable code to run.")
        return False

    execution_globals = {
        'pd': pd, 'plt': plt, 'ticker': ticker, 'sns': sns, 'np': np, 'df': df_data,
        'haversine_distance': haversine_distance,
        'convert_negatives_to_half_positive': convert_negatives_to_half_positive,
        'mcolors': plt.matplotlib.colors
    }

    ### MODIFICATION START: Capture variable names before execution
    initial_vars = set(execution_globals.keys())
    ### MODIFICATION END

    print("\n▶️ Executing Generated Code...\n--------------------------------")
    try:
        exec(code_to_execute, execution_globals)
        if plt.get_fignums():
            print("\n🖼️ Displaying generated plot(s)...")
            plt.show()
        else:
            print("\nℹ️ No plots were generated by the code.")
        print("--------------------------------\n✅ Code Execution Finished.")

        ### MODIFICATION START: Check for new DataFrames and offer to save them
        final_vars = set(execution_globals.keys())
        new_vars = final_vars - initial_vars
        new_dfs = {
            name: var for name, var in execution_globals.items()
            if name in new_vars and isinstance(var, pd.DataFrame) and not var.empty
        }

        if new_dfs:
            print("\n💾 New DataFrames were created by the code.")
            for name, new_df in new_dfs.items():
                print(f"\n--- Found DataFrame: '{name}' ---")
                print(f"Shape: {new_df.shape}")
                print("Head:")
                print(new_df.head())

                while True:
                    save_choice = input(f"Do you want to save the '{name}' DataFrame to a CSV file? (yes/no): ").lower().strip()
                    if save_choice in ['yes', 'y']:
                        default_filename = f"{name}.csv"
                        filename_prompt = f"Enter filename (press Enter for '{default_filename}'): "
                        filename = input(filename_prompt).strip()
                        if not filename:
                            filename = default_filename

                        try:
                            new_df.to_csv(filename, index=False)
                            print(f"✅ Successfully saved to '{os.path.abspath(filename)}'")
                        except Exception as e:
                            print(f"❌ Error saving file: {e}")
                        break
                    elif save_choice in ['no', 'n']:
                        print(f"Skipping save for '{name}'.")
                        break
                    else:
                        print("Invalid input. Please enter 'yes' or 'no'.")
        ### MODIFICATION END

        return True
    except Exception as e:
        print(f"\n❌ Error during execution of generated code:\n{e}\n--------------------------------")
        return False


if __name__ == "__main__":
    print("Conversational Geochemistry Data Analyser with Location Awareness (Google Gemini)")
    print("===============================================================================")
    print("Type 'exit' to end the conversation.")
    csv_filepath = input("Enter the path to your geochemistry CSV file: ").strip()
    if not os.path.exists(csv_filepath): print(f"Error: File not found at '{csv_filepath}'"); exit()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key: api_key = input("Enter your Google API Key (or set GOOGLE_API_KEY environment variable): ").strip()
    if not api_key: print("Error: Google API Key is required."); exit()

    default_model_name = "gemini-1.5-flash-preview-0514"
    if get_gemini_generated_code.__defaults__: default_model_name = get_gemini_generated_code.__defaults__[0]

    gemini_model_name = input(f"Enter Gemini model name (e.g., {default_model_name}): ").strip()
    if not gemini_model_name: gemini_model_name = default_model_name; print(f"Using default model: {gemini_model_name}")

    try:
        print(f"\n🔄 Loading data from '{csv_filepath}'...")
        main_df = pd.read_csv(csv_filepath)
        current_df_state = main_df.copy()
        print(f"✅ Data loaded successfully. Shape: {current_df_state.shape}")
    except Exception as e: print(f"Error loading CSV file: {e}"); exit()

    data_summary = get_data_summary(main_df)
    conversation_history = deque(maxlen=5)
    while True:
        print("\n" + "-"*50)
        user_question = input("Ask your geochemistry question (or type 'exit'): ").strip()
        if user_question.lower() == 'exit': print("Exiting analyser. Goodbye!"); break
        if not user_question: continue

        history_str_for_prompt = "\n\n".join(list(conversation_history))
        print("\n🤖 Requesting analysis code from Gemini...")

        generated_code = get_gemini_generated_code(
            user_question, data_summary, history_str_for_prompt, api_key, model_name=gemini_model_name
        )

        if generated_code:
            print("\n📝 Gemini generated the following Python code:\n----------------------------------------------------")
            print(generated_code)
            print("----------------------------------------------------")
            if execute_generated_code(generated_code, current_df_state):
                turn_summary = f"User asked: \"{user_question}\"\nAssistant generated code:\n```python\n{generated_code}\n```"
                conversation_history.append(turn_summary)
        else: print("Could not generate code from Gemini for this question.")
    print("\n👋 Analysis session ended.")
