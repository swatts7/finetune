import streamlit as st
import pandas as pd
import json
import io
import hashlib
from openai_utils import chat_with_model

st.set_page_config(layout="wide")

# Password hashing function
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# Password checking function
def check_password():
    """Returns `True` if the user had the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if hash_password(st.session_state["password"]) == st.secrets["general"]["password"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # don't store password
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # First run, show input for password.
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        return False
    elif not st.session_state["password_correct"]:
        # Password incorrect, show input + error.
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        st.error("😕 Password incorrect")
        return False
    else:
        # Password correct.
        return True

# Ensure secrets are loaded correctly
try:
    secret_password = st.secrets["general"]["password"]
except KeyError:
    st.error("No password found in secrets. Please configure the secrets.toml file correctly.")
    st.stop()

# Check password
if not check_password():
    st.stop()

# If password is correct, proceed with the rest of the app
st.success("Password correct. Access granted.")

def generate_jsonl_data():
    data = []
    summary_data = load_summary_data()  # Load the summary data
    outputs_df = load_outputs_data()  # Load the outputs (generated summaries)
    system_prompt = load_system_prompt("master/system_prompt.txt")  # Load system prompt

    for _, row in summary_data.iterrows():
        operator_id = row['operator_id']
        operator_name = row['operator_name']
        review_meta_summary = row['review_meta_summary']
        comment_meta_summary = row['comment_meta_summary']

        # Fetch the generated summary (master_summary) for this operator
        generated_summary = outputs_df[outputs_df['operator_id'] == operator_id]['master_summary'].values[0] if not outputs_df[outputs_df['operator_id'] == operator_id].empty else ""

        # Create the user prompt
        user_prompt = {
            "operator_name": operator_name,
            "review_meta_summary": review_meta_summary,
            "comment_meta_summary": comment_meta_summary
        }

        # Add the conversation structure for JSONL
        conversation = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(user_prompt, indent=2)},
                {"role": "assistant", "content": generated_summary}
            ]
        }

        data.append(conversation)

    return data

# Utility functions
def load_outputs_data():
    try:
        df = pd.read_csv("master/outputs.csv")
        return df
    except FileNotFoundError:
        return pd.DataFrame(columns=['operator_name', 'operator_id', 'master_summary', 'is_approved'])

def save_output(operator_id, master_summary, is_approved):
    df = load_outputs_data()
    df = df[df['operator_id'] != operator_id]  # Remove existing entry if any
    operator_name = df[df['operator_id'] == operator_id]['operator_name'].values[0] if not df[df['operator_id'] == operator_id].empty else ""
    new_row = pd.DataFrame({
        'operator_name': [operator_name],
        'operator_id': [operator_id],
        'master_summary': [master_summary],
        'is_approved': [is_approved]
    })
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv("master/outputs.csv", index=False)

def load_summary_data():
    try:
        df = pd.read_csv("master/master_summary_data.csv")
        return df
    except FileNotFoundError:
        st.error("Summary data file not found.")
        return pd.DataFrame()

def load_system_prompt(file_path):
    try:
        with open(file_path, 'r') as f:
            return f.read()
    except FileNotFoundError:
        st.error(f"System prompt file not found: {file_path}")
        return ""

def save_system_prompt(file_path, content):
    with open(file_path, 'w') as f:
        f.write(content)

def generate_summary(operator_id, user_prompt, system_prompt):
    try:
        api_key = st.secrets["openai_api_key"]
        summary, _, _ = chat_with_model(
            api_key,
            user_message_content=json.dumps(user_prompt),
            system_message_content=system_prompt,
            model="gpt-4o-mini"
        )
        return summary
    except Exception as e:
        st.error(f"Error generating summary: {str(e)}")
        return ""

def make_clickable(operator_id):
    return f'<a href="?operator_id={operator_id}" target="_self">{operator_id}</a>'

def download_csv(df):
    return df.to_csv(index=False).encode('utf-8')

def download_jsonl_button():
    # Generate JSONL data
    data = generate_jsonl_data()

    # Create a StringIO buffer
    buffer = io.StringIO()

    # Write each entry as a separate line in JSONL format
    for entry in data:
        buffer.write(json.dumps(entry) + "\n")

    # Seek back to the start of the buffer
    buffer.seek(0)

    # Download the JSONL file
    st.download_button(
        label="Download JSONL",
        data=buffer.getvalue(),
        file_name="operator_summaries.jsonl",
        mime="application/json"
    )

def main():
    st.title("Operator Master Summary")

    # Load data
    summary_data = load_summary_data()
    outputs_df = load_outputs_data()
    system_prompt = load_system_prompt("master/system_prompt.txt")

    # Editable System Prompt in an expander
    with st.expander("Edit System Prompt"):
        edited_system_prompt = st.text_area("System Prompt", system_prompt, height=200)
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Save System Prompt"):
                save_system_prompt("master/system_prompt.txt", edited_system_prompt)
                st.success("System prompt saved successfully!")
        with col2:
            if st.button("Reset System Prompt"):
                st.rerun()

    # Check if an operator_id is in the URL parameters
    params = st.query_params
    if 'operator_id' in params:
        operator_id = params['operator_id']
        show_detailed_view(operator_id, summary_data, outputs_df, edited_system_prompt)
    else:
        show_overview_table(summary_data, outputs_df)

def show_overview_table(summary_data, outputs_df):
    st.subheader("Operator Comments Overview")

    # Merge the summary_data and outputs_df for the overview table
    merged_df = pd.merge(summary_data[['operator_name', 'operator_id']], 
                         outputs_df[['operator_id', 'master_summary', 'is_approved']],
                         on='operator_id', how='left')

    # Fill NaN values for new columns
    merged_df['master_summary'].fillna('', inplace=True)
    merged_df['is_approved'].fillna(False, inplace=True)
    
    # Add clickable links to operator_id
    merged_df['operator_id'] = merged_df['operator_id'].apply(make_clickable)
    
    # Display the overview table
    st.write(merged_df[['operator_name', 'operator_id', 'master_summary', 'is_approved']].to_html(escape=False, index=False), unsafe_allow_html=True)

    # Add download button for the CSV
    csv = download_csv(merged_df)
    st.download_button(
        label="Download CSV",
        data=csv,
        file_name="operator_comments_summary.csv",
        mime="text/csv"
    )
    download_jsonl_button()

def show_detailed_view(operator_id, summary_data, outputs_df, system_prompt):
    st.subheader(f"Operator: {operator_id}")

    # Fetch operator data
    operator_row = summary_data[summary_data['operator_id'] == operator_id]
    if operator_row.empty:
        st.error("Operator data not found.")
        return

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("OPENAI INPUT")
        operator_name = operator_row['operator_name'].values[0]
        review_meta_summary = operator_row['review_meta_summary'].values[0]
        comment_meta_summary = operator_row['comment_meta_summary'].values[0]

        # Create a JSON structure for user prompt using review_meta_summary and comment_meta_summary
        user_prompt = {
            "operator_name": operator_name,
            "review_meta_summary": review_meta_summary,
            "comment_meta_summary": comment_meta_summary
        }

        # Display the prompt in a text area (prettified JSON format)
        st.text_area("User Prompt", json.dumps(user_prompt, indent=2), height=200)

    with col2:
        st.subheader("OPENAI OUTPUT")

        # Initialize session state for the generated summary
        if 'generated_summary' not in st.session_state:
            current_summary = outputs_df[outputs_df['operator_id'] == operator_id]['master_summary'].iloc[0] if not outputs_df[outputs_df['operator_id'] == operator_id].empty else ""
            st.session_state.generated_summary = current_summary

        # Display the current or generated summary
        summary = st.text_area("Generated Summary", st.session_state.generated_summary, height=200)

        is_approved = outputs_df[outputs_df['operator_id'] == operator_id]['is_approved'].iloc[0] if not outputs_df[outputs_df['operator_id'] == operator_id].empty else False
        is_approved = st.checkbox("Approve Summary", value=is_approved)

        if st.button("Generate Summary"):
            try:
                # Generate new summary using review_meta_summary and comment_meta_summary
                new_summary = generate_summary(operator_id, user_prompt, system_prompt)

                # Update the session state with the new summary and rerun the app to refresh UI
                st.session_state.generated_summary = new_summary
                st.success("Summary generated successfully!")
                st.rerun()  # Force rerun to immediately update the UI
            except json.JSONDecodeError:
                st.error("Invalid JSON in user prompt. Please check the format.")

        # Make sure the summary in the session state is passed for saving
        if st.button("Save Changes"):
            if st.session_state.generated_summary:
                save_output(operator_id, st.session_state.generated_summary, is_approved)
                st.success("Changes saved successfully!")
            else:
                st.error("Cannot save an empty summary.")
    
    if st.button("Back to Overview"):
        st.query_params.clear()  # Clears the query params, taking user back to the overview
        st.rerun()



if __name__ == "__main__":
    main()
