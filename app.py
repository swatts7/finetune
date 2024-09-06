import streamlit as st
import json
import os
import re
import io
import hashlib
import csv
import pandas as pd
from openai_utils import chat_with_model, print_token_usage_summary

st.set_page_config(layout="wide")

# Password hashing function
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

if 'password_correct' not in st.session_state:
    st.session_state['password_correct'] = False

def check_password():
    """Returns `True` if the user had the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if hash_password(st.session_state["password"]) == st.secrets["general"]["password"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # don't store password
        else:
            st.session_state["password_correct"] = False

    if st.session_state["password_correct"]:
        return True

    # Show input for password.
    st.text_input(
        "Password", type="password", on_change=password_entered, key="password"
    )
    if "password" in st.session_state:
        st.error("😕 Password incorrect")
    return False

def generate_jsonl_data(system_prompt):
    data = []
    for review in st.session_state['processed_reviews']:
        operator_id = review.get('operator_id', 'N/A')
        user_message = json.dumps(review.get('unique_reviews', []))
        output, _ = load_review(operator_id)

        data.append({
            "system_message": system_prompt,
            "user_message": user_message,
            "output": output
        })

    return data

def download_jsonl_button():
    with open("review_meta/system_prompt.txt", "r", encoding="utf-8") as f:
        system_prompt = f.read()

    data = generate_jsonl_data(system_prompt)
    buffer = io.StringIO()
    for entry in data:
        buffer.write(json.dumps(entry) + '\n')
    buffer.seek(0)

    st.download_button(
        label="Download JSONL",
        data=buffer.getvalue(),
        file_name="operator_reviews.jsonl",
        mime="application/json"
    )

def generate_csv_data(system_prompt):
    data = []
    for review in st.session_state['processed_reviews']:
        operator_id = review.get('operator_id', 'N/A')
        user_message = json.dumps(review.get('unique_reviews', []))
        output, _ = load_review(operator_id)

        data.append({
            "operator_id": operator_id,
            "system_message": system_prompt,
            "user_message": user_message,
            "output": output
        })

    df = pd.DataFrame(data, columns=['operator_id', 'system_message', 'user_message', 'output'])
    return df

def download_csv_button():
    with open("review_meta/system_prompt.txt", "r", encoding="utf-8") as f:
        system_prompt = f.read()

    df = generate_csv_data(system_prompt)
    buffer = io.StringIO()
    df.to_csv(buffer, index=False, encoding='utf-8', quoting=csv.QUOTE_NONNUMERIC)
    buffer.seek(0)

    st.download_button(
        label="Download CSV",
        data=buffer.getvalue(),
        file_name="operator_reviews.csv",
        mime="text/csv"
    )

def calculate_operator_statistics(reviews):
    valid_scores = [float(review.get('normalized_score', 0)) for review in reviews 
                    if review.get('normalized_score') not in [None, 'N/A'] and 0 <= float(review.get('normalized_score', 0)) <= 1]
    
    num_reviews = len(reviews)
    
    if valid_scores:
        total_score = sum(valid_scores)
        average_score = total_score / len(valid_scores) if len(valid_scores) > 0 else 0
        median_score = sorted(valid_scores)[len(valid_scores) // 2]
    else:
        average_score, median_score = 0, 0
    
    return num_reviews, average_score, median_score

if 'processed_reviews' not in st.session_state:
    with open("review_meta/processed_reviews.json", "r", encoding="utf-8") as f:
        st.session_state['processed_reviews'] = json.load(f)

def clean_text(text):
    if isinstance(text, str):
        return text
    return str(text)

def ensure_directory_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)

def save_review(operator_id, output, is_approved):
    review_data = {
        "output": output,
        "is_approved": is_approved
    }
    ensure_directory_exists("reviews")
    with open(f"reviews/{operator_id}.json", "w") as f:
        json.dump(review_data, f)

def load_review(operator_id):
    try:
        with open(f"reviews/{operator_id}.json", "r") as f:
            review_data = json.load(f)
        return review_data["output"], review_data["is_approved"]
    except FileNotFoundError:
        return "", False

def make_clickable(operator_id):
    return f'<a href="?operator_id={operator_id}" target="_self">{operator_id}</a>'

def load_and_update_dataframe():
    pd.set_option('display.max_colwidth', None)
    df = pd.read_csv("outputs.csv")
    
    for index, row in df.iterrows():
        output, is_approved = load_review(row['operator_id'])
        df.at[index, 'review_summary'] = clean_text(output) if output else ''
        df.at[index, 'is_approved'] = is_approved

        selected_operator = next((review for review in st.session_state['processed_reviews'] if review['operator_id'] == row['operator_id']), None)
        if selected_operator:
            num_reviews, average_score, median_score = calculate_operator_statistics(selected_operator['unique_reviews'])
            df.at[index, 'num_reviews'] = num_reviews
            df.at[index, 'average_score'] = round(average_score, 2)
            df.at[index, 'median_score'] = round(median_score, 2)
    
    df['operator_id'] = df['operator_id'].apply(make_clickable)
    return df

def save_dataframe(df):
    df['operator_id'] = df['operator_id'].apply(lambda x: x.split('>')[1].split('<')[0] if '>' in x else x)
    df.to_csv("outputs.csv", index=False)

def update_outputs_csv(operator_id, output, is_approved):
    df = pd.read_csv("outputs.csv")
    df.loc[df['operator_id'] == operator_id, 'review_summary'] = clean_text(output)[:100] + '...'
    df.loc[df['operator_id'] == operator_id, 'is_approved'] = is_approved
    save_dataframe(df)

def format_reviews(reviews):
    formatted_reviews = ""
    for i, review in enumerate(reviews, 1):
        formatted_reviews += f"Review {i}:\n"
        formatted_reviews += f"Summary: {clean_text(review.get('review_summary', 'No summary available'))}\n\n"
        
        normalized_score = review.get('normalized_score', 'N/A')
        if normalized_score == 'N/A' or normalized_score is None:
            formatted_score = 'null'
        else:
            try:
                score_float = float(normalized_score)
                if score_float == 0 or score_float > 1:
                    formatted_score = 'null'
                else:
                    formatted_score = f"{score_float:.2f}"
            except ValueError:
                formatted_score = 'null'
        
        formatted_reviews += f"Normalized Score: {formatted_score}\n\n"
        
        pros = review.get('pros', [])
        if pros:
            formatted_reviews += "Pros:\n"
            for pro in pros:
                formatted_reviews += f"- {clean_text(pro)}\n"
        else:
            formatted_reviews += "Pros: None listed\n"
        
        cons = review.get('cons', [])
        if cons:
            formatted_reviews += "\nCons:\n"
            for con in cons:
                formatted_reviews += f"- {clean_text(con)}\n"
        else:
            formatted_reviews += "\nCons: None listed\n"
        
        quotes = review.get('quotes', [])
        if quotes:
            formatted_reviews += "\nQuotes:\n"
            full_quote = " ".join([clean_text(q) for q in quotes])
            formatted_quotes = re.split(r'(?<=[.!?])\s+', full_quote)
            for quote in formatted_quotes:
                if quote:
                    formatted_reviews += f"- {quote}\n"
        else:
            formatted_reviews += "\nQuotes: None available\n"
        
        formatted_reviews += f"\nSource: {review.get('review_source_url', 'No source URL available')}\n\n"
        formatted_reviews += "-" * 50 + "\n\n"
    
    return formatted_reviews

def generate_summary(operator_id, system_prompt):
    selected_operator = next((review for review in st.session_state['processed_reviews'] if review['operator_id'] == operator_id), None)
    if selected_operator:
        try:
            api_key = st.secrets["openai_api_key"]
            regenerated_output, _, _ = chat_with_model(
                api_key,
                user_message_content=json.dumps(selected_operator['unique_reviews']),
                system_message_content=system_prompt,
                model="gpt-4o-mini"
            )
            save_review(operator_id, regenerated_output, False)
            update_outputs_csv(operator_id, regenerated_output, False)
            return True, regenerated_output
        except Exception as e:
            print(f"Error generating summary: {str(e)}")
            return False, ""
    return False, ""

def show_detailed_review(operator_id, system_prompt):
    st.header(f"{operator_id}")

    selected_operator = next((review for review in st.session_state['processed_reviews'] if review['operator_id'] == operator_id), None)

    if selected_operator:
        num_reviews, average_score, median_score = calculate_operator_statistics(selected_operator['unique_reviews'])

        st.write(f"**Total Number of Reviews:** {num_reviews}")
        st.write(f"**Average Score:** {average_score:.2f}")
        st.write(f"**Median Score:** {median_score:.2f}")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("OPENAI INPUT")
            formatted_reviews = format_reviews(selected_operator['unique_reviews'])
            st.text_area("Reviews", formatted_reviews, height=400)

        with col2:
            st.subheader("OPENAI Output")
            edited_output, is_approved = load_review(operator_id)
            edited_output = st.text_area("Edit Desired Output", edited_output, height=400, key="desired_output")
            
            is_approved = st.checkbox("Mark as Approved", value=is_approved)
            
            if st.button("Save Changes"):
                save_review(operator_id, edited_output, is_approved)
                update_outputs_csv(operator_id, edited_output, is_approved)
                st.success("Changes saved successfully!")
            
            if st.button("Regenerate"):
                success, regenerated_output = generate_summary(operator_id, system_prompt)
                if success:
                    edited_output = regenerated_output
                    st.success("Summary regenerated successfully!")
                    st.rerun()
                else:
                    st.error("Failed to regenerate summary.")

        if st.button("Back to Overview"):
            st._set_query_params()
            st.rerun()
    else:
        st.error("Selected operator not found.")

def main():
    
    if not st.session_state["password_correct"]:
    if not check_password():
            st.stop()
        else:
            st.success("Password correct. Access granted.")
    st.header("Review Meta Summary Editor")

    with open("review_meta/system_prompt.txt", "r", encoding="utf-8") as f:
        system_prompt = f.read()

    with st.expander("Edit System Prompt"):
        edited_system_prompt = st.text_area("Edit System Prompt", system_prompt, height=300)

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Save System Prompt"):
                try:
                    with open("review_meta/system_prompt.txt", "w", encoding="utf-8") as f:
                        f.write(edited_system_prompt)
                    st.success("System prompt saved successfully!")
                    system_prompt = edited_system_prompt
                except Exception as e:
                    st.error(f"An error occurred while saving the system prompt: {str(e)}")

        with col2:
            if st.button("Cancel"):
                st.rerun()

    params = st._get_query_params()
    if 'operator_id' in params:
        show_detailed_review(params['operator_id'][0], system_prompt)
    else:
        st.subheader("Operator Overview")
        df = load_and_update_dataframe()
        st.write(df.to_html(escape=False, index=False), unsafe_allow_html=True)
        download_csv_button()
        download_jsonl_button()

if __name__ == "__main__":
    main()
