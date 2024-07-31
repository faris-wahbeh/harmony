import streamlit as st
import numpy as np
from statsmodels.stats.inter_rater import fleiss_kappa
from itertools import combinations

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

st.title("Rayyan Team Harmony Test")

n_reviewers = st.number_input("Number of reviewers:", min_value=1, value=5, step=1)
n_items = st.number_input("Number of items:", min_value=1, value=5, step=1)
threshold = st.number_input("Enter the threshold:", min_value=0.0, max_value=1.0, value=0.6)

ratings = []
st.subheader("Enter Ratings for Each Reviewer (comma-separated ones and zeros)")

for i in range(n_reviewers):
    reviewer_input = st.text_area(f"Reviewer {i+1}", key=f"reviewer_{i}")
    if reviewer_input:
        reviewer_ratings = list(map(int, reviewer_input.split(',')))
        if len(reviewer_ratings) != n_items:
            st.warning(f"Reviewer {i+1}: Number of ratings ({len(reviewer_ratings)}) does not match number of items ({n_items})")
        else:
            ratings.append(reviewer_ratings)

if st.button("Calculate"):
    if len(ratings) == n_reviewers and all(len(r) == n_items for r in ratings):
        try:
            groups = classify_reviewers(ratings, threshold)
            kappa_results = get_group_kappas(groups, ratings)
            
            st.subheader("Classified Groups and Fleiss' Kappa")
            for group, kappa in kappa_results:
                st.markdown(f"**Group {group}**")
                st.write(f"Fleiss' kappa: {kappa:.4f}")

        except Exception as e:
            st.error(f"Error: {str(e)}")
    else:
        st.error("Please ensure all reviewers have provided the correct number of ratings.")
