import streamlit as st
import pandas as pd
import json
import io
import hashlib
from openai_utils import chat_with_model

st.set_page_config(layout="wide")

def generate_jsonl_data():
    data = []
    summary_data = load_summary_data()
    outputs_df = load_outputs_data()
    system_prompt = load_system_prompt("comments_meta\\system_prompt.txt")

    for _, row in summary_data.iterrows():
        operator_id = row['operator_id']
        operator_name = row['operator_name']
        
        # Create the user prompt using available data
        user_prompt = {
            "operator_id": operator_id,
            "operator_name": operator_name,
            "total_comments": row['total_comments'],
            "average_sentiment_mean": row['average_sentiment_mean'],
            "average_sentiment_median": row['average_sentiment_median'],
            "top_positive_mentions": row['top_positive_mentions'],
            "top_negative_mentions": row['top_negative_mentions'],
            "top_positive_comments": row['top_positive_comments'],
            "top_negative_comments": row['top_negative_comments']
        }

        # Fetch the generated summary for this operator
        generated_summary = outputs_df[outputs_df['operator_id'] == operator_id]['comments_summary'].values[0] if not outputs_df[outputs_df['operator_id'] == operator_id].empty else ""

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
        df = pd.read_csv("comments_meta/outputs.csv")
        return df
    except FileNotFoundError:
        return pd.DataFrame(columns=['operator_name', 'operator_id', 'comments_summary', 'is_approved'])

def save_output(operator_id, comments_summary, is_approved):
    df = load_outputs_data()
    existing_row = df[df['operator_id'] == operator_id]
    
    if not existing_row.empty:
        # Update existing row
        df.loc[df['operator_id'] == operator_id, 'comments_summary'] = comments_summary
        df.loc[df['operator_id'] == operator_id, 'is_approved'] = is_approved
    else:
        # Add new row
        new_row = pd.DataFrame({
            'operator_name': [''],  # Leave blank, it will be filled later if needed
            'operator_id': [operator_id],
            'comments_summary': [comments_summary],
            'is_approved': [is_approved]
        })
        df = pd.concat([df, new_row], ignore_index=True)
    
    df.to_csv("comments_meta/outputs.csv", index=False)

def load_summary_data():
    try:
        with open("comments_meta/comments_data.json", 'r') as f:
            json_data = json.load(f)

        # Convert JSON into a flat DataFrame
        records = []
        for operator_id, operator_data in json_data.items():
            record = {
                'operator_id': operator_id,
                'operator_name': operator_data.get('casino_name', ''),
                'total_comments': operator_data.get('total_comments', 0),
                'average_sentiment_mean': operator_data.get('average_sentiment', {}).get('mean', None),
                'average_sentiment_median': operator_data.get('average_sentiment', {}).get('median', None),
                'top_positive_mentions': operator_data.get('contextual_analysis', {}).get('top_positive_mentions', {}),
                'top_negative_mentions': operator_data.get('contextual_analysis', {}).get('top_negative_mentions', {}),
                'top_positive_comments': operator_data.get('top_comments', {}).get('positive', []),
                'top_negative_comments': operator_data.get('top_comments', {}).get('negative', [])
            }
            records.append(record)

        # Convert the list of records into a DataFrame
        df = pd.DataFrame(records)
        return df

    except FileNotFoundError:
        st.error("Comments data file not found.")
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
        print(f"\n--- Generating summary for operator: {operator_id} ---")
        api_key = st.secrets["openai_api_key"]
        
        print("Sending request to OpenAI with:")
        print(f"User prompt: {json.dumps(user_prompt, indent=2)}")
        print(f"System prompt: {system_prompt}")
        
        summary, _, _ = chat_with_model(
            api_key,
            user_message_content=json.dumps(user_prompt),
            system_message_content=system_prompt,
            model="gpt-4o-mini"
        )
        
        print("\nResponse from OpenAI:")
        print(summary)
        print("--- End of OpenAI response ---\n")
        
        return summary
    except Exception as e:
        print(f"Error in generate_summary: {str(e)}")
        st.error(f"Error generating summary: {str(e)}")
        return ""

def make_clickable(operator_id):
    return f'<a href="?operator_id={operator_id}" target="_self">{operator_id}</a>'

def download_csv(df):
    return df.to_csv(index=False).encode('utf-8')

def download_jsonl_button():
    data = generate_jsonl_data()
    buffer = io.StringIO()
    for entry in data:
        buffer.write(json.dumps(entry) + "\n")
    buffer.seek(0)
    st.download_button(
        label="Download JSONL",
        data=buffer.getvalue(),
        file_name="operator_summaries.jsonl",
        mime="application/json"
    )

def main():
    st.title("Operator Comments Summary")

    summary_data = load_summary_data()
    outputs_df = load_outputs_data()
    system_prompt = load_system_prompt("comments_meta/system_prompt.txt")

    with st.expander("Edit System Prompt"):
        edited_system_prompt = st.text_area("System Prompt", system_prompt, height=200)
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Save System Prompt"):
                save_system_prompt("comments_meta/system_prompt.txt", edited_system_prompt)
                st.success("System prompt saved successfully!")
        with col2:
            if st.button("Reset System Prompt"):
                st.rerun()

    params = st.query_params
    if 'operator_id' in params:
        operator_id = params['operator_id']
        show_detailed_view(operator_id, summary_data, outputs_df, edited_system_prompt)
    else:
        show_overview_table(outputs_df)

def show_overview_table(outputs_df):
    st.subheader("Operator Comments Overview")

    if outputs_df.empty:
        st.error("No data available in the CSV.")
        return

    required_columns = ['operator_name', 'operator_id', 'comments_summary', 'is_approved']
    available_columns = [col for col in required_columns if col in outputs_df.columns]

    if len(available_columns) < len(required_columns):
        missing_columns = set(required_columns) - set(available_columns)
        st.warning(f"Some columns are missing in CSV data: {', '.join(missing_columns)}")

    if 'operator_id' in available_columns:
        outputs_df['operator_id'] = outputs_df['operator_id'].apply(make_clickable)

    # Display the table with available columns
    st.write(outputs_df[available_columns].to_html(escape=False, index=False), unsafe_allow_html=True)

    # Download CSV button
    csv = download_csv(outputs_df[available_columns])
    st.download_button(
        label="Download CSV",
        data=csv,
        file_name="operator_comments_summary.csv",
        mime="text/csv"
    )

    # Download JSONL button
    download_jsonl_button()
def show_detailed_view(operator_id, comments_data_df, outputs_df, system_prompt):
    st.subheader(f"Operator: {operator_id}")

    operator_id = str(operator_id)
    operator_data = comments_data_df[comments_data_df['operator_id'] == operator_id]

    if operator_data.empty:
        st.warning(f"No user data found for operator_id: {operator_id}")
        user_prompt = "{}"
    else:
        user_prompt_data = operator_data.to_dict(orient="records")[0]
        user_prompt = json.dumps(user_prompt_data, indent=2)

    operator_row = outputs_df[outputs_df['operator_id'] == operator_id]

    # Initialize session state if it doesn't exist
    if "generated_summary" not in st.session_state:
        if not operator_row.empty and 'comments_summary' in operator_row.columns:
            st.session_state["generated_summary"] = operator_row['comments_summary'].iloc[0]
        else:
            st.session_state["generated_summary"] = ""

    def generate_summary_callback():
        try:
            print("Generate summary button clicked")
            user_prompt_dict = json.loads(edited_user_prompt)
            new_summary = generate_summary(operator_id, user_prompt_dict, system_prompt)
            print(f"New summary generated: {new_summary}")
            st.session_state["generated_summary"] = new_summary
            save_output(operator_id, new_summary, st.session_state.get("is_approved", False))
            st.success("Summary generated and saved successfully!")
            print("About to rerun")
            st.experimental_rerun()
        except json.JSONDecodeError:
            st.error("Invalid JSON format in User Prompt.")
        except Exception as e:
            st.error(f"An error occurred while generating the summary: {str(e)}")

    def save_changes_callback():
        st.session_state["generated_summary"] = generated_summary
        save_output(operator_id, generated_summary, st.session_state.get("is_approved", False))
        st.success("Changes saved successfully!")
        
        # Force a reload of the outputs data
        st.session_state['outputs_df'] = load_outputs_data()
        
        # Clear the query parameters to go back to the overview
        st.query_params.clear()
        st.rerun()

    col1, col2 = st.columns(2)

    with col1:
        edited_user_prompt = st.text_area("User Prompt", value=user_prompt, height=400)

    with col2:
        try:
            # Ensure the session state value is a string and handle potential None values
            summary_value = st.session_state.get("generated_summary", "")
            if summary_value is None:
                summary_value = ""
            elif not isinstance(summary_value, str):
                summary_value = str(summary_value)
            
            generated_summary = st.text_area(
                "Generated Summary", 
                value=summary_value,
                height=400, 
                key="generated_summary_input"
            )
        except Exception as e:
            st.error(f"Error displaying text area: {str(e)}")
            generated_summary = ""

    is_approved = operator_row['is_approved'].iloc[0] if not operator_row.empty else False
    st.session_state["is_approved"] = st.checkbox("Approve Summary", value=is_approved)

    if st.button("Generate Summary", on_click=generate_summary_callback):
        pass

    if st.button("Save Changes", on_click=save_changes_callback):
        pass

    if st.button("Back to Overview"):
        st.query_params.clear()
        st.rerun()

if __name__ == "__main__":
    main()
