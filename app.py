import streamlit as st
import numpy as np
import pandas as pd
from statsmodels.stats.inter_rater import fleiss_kappa
from itertools import combinations
import json

def compute_fleiss_kappa(matrix):
    kappa = fleiss_kappa(matrix)
    if np.isnan(kappa):
        return -1  # Treat NaN as unacceptable kappa score
    return kappa

def create_contingency_matrix(data, n_categories):
    n_reviewers = len(data)
    n_items = len(data[0])
    matrix = np.zeros((n_items, n_categories))
    for ratings in data:
        for i, rating in enumerate(ratings):
            matrix[i, rating] += 1
    return matrix

def acceptable_kappa(group, data, threshold):
    group_data = [data[i] for i in group]
    n_categories = max(max(r) for r in data) + 1
    matrix = create_contingency_matrix(group_data, n_categories)
    return compute_fleiss_kappa(matrix) >= threshold

def classify_reviewers(data, threshold):
    n = len(data)
    reviewers = list(range(n))
    best_groups = []
    used_reviewers = set()
    for r in range(n, 1, -1):
        for comb in combinations(reviewers, r):
            if all(reviewer not in used_reviewers for reviewer in comb):
                if acceptable_kappa(comb, data, threshold):
                    best_groups.append(list(comb))
                    used_reviewers.update(comb)
    for reviewer in reviewers:
        if reviewer not in used_reviewers:
            best_groups.append([reviewer])
    return best_groups

def get_group_kappas(groups, data):
    results = []
    for group in groups:
        group_data = [data[i] for i in group]
        n_categories = max(max(r) for r in data) + 1
        matrix = create_contingency_matrix(group_data, n_categories)
        kappa = compute_fleiss_kappa(matrix)
        results.append((group, kappa))
    return results

def parse_decisions_correctly(json_string):
    try:
        clean_string = json_string.replace('RAYYAN-INCLUSION: ', '').strip()
        clean_string = clean_string.replace('=>', ':').replace("'", '"')
        decisions = json.loads(clean_string)
        return decisions
    except json.JSONDecodeError as e:
        return None

def convert_to_reviewer_arrays(df):
    reviewer_names = df['parsed'].iloc[0].keys()
    reviewer_arrays = {reviewer: [] for reviewer in reviewer_names}
    
    for index, row in df.iterrows():
        for reviewer, decision in row['parsed'].items():
            reviewer_arrays[reviewer].append(1 if decision == 'Included' else 0)
    
    return reviewer_arrays

st.title("Fleiss' Kappa Calculator")

threshold = st.number_input("Enter the threshold:", min_value=0.0, max_value=1.0, value=0.6)

uploaded_file = st.file_uploader("Upload CSV file with ratings", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.write("Uploaded CSV File:")
        st.write(df)

        # Parse the decisions
        df['parsed'] = df.iloc[:, 0].apply(parse_decisions_correctly)
        
        # Convert to arrays by reviewer
        reviewer_arrays = convert_to_reviewer_arrays(df)
        
        # Transpose the data to match Fleiss' Kappa expected format
        ratings = list(reviewer_arrays.values())
        n_reviewers = len(ratings)
        n_items = len(ratings[0])
        
        st.write("Reviewer Arrays:")
        st.write(reviewer_arrays)
        
        if st.button("Calculate"):
            try:
                # Calculate the overall Fleiss' Kappa before any grouping
                matrix = create_contingency_matrix(ratings, 2)
                overall_kappa = compute_fleiss_kappa(matrix)
                
                st.subheader("Overall Fleiss' Kappa")
                st.write(f"Fleiss' kappa for all reviewers: {overall_kappa:.4f}")
                
                # Grouping and calculating Kappa for groups
                groups = classify_reviewers(ratings, threshold)
                kappa_results = get_group_kappas(groups, ratings)

                st.subheader("Classified Groups and Fleiss' Kappa")
                for group, kappa in kappa_results:
                    group_names = [list(reviewer_arrays.keys())[i] for i in group]
                    st.markdown(f"**Group {group_names}**")
                    st.write(f"Fleiss' kappa: {kappa:.4f}")

            except Exception as e:
                st.error(f"Error: {str(e)}")
    except Exception as e:
        st.error(f"Error loading the file: {str(e)}")
