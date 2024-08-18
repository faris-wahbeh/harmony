import streamlit as st
import numpy as np
import pandas as pd
from statsmodels.stats.inter_rater import fleiss_kappa
from itertools import combinations
import json
from fpdf import FPDF

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
    if n == 2:
        matrix = create_contingency_matrix(data, 2)
        kappa = compute_fleiss_kappa(matrix)
        if kappa < threshold:
            return []
        else:
            return [list(range(n))]

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

def get_group_kappas(groups, data, reviewer_arrays):
    results = []
    for group in groups:
        group_names = [list(reviewer_arrays.keys())[i] for i in group]
        if len(group) > 1:
            group_data = [data[i] for i in group]
            n_categories = max(max(r) for r in data) + 1
            matrix = create_contingency_matrix(group_data, n_categories)
            kappa = compute_fleiss_kappa(matrix)
            results.append((group_names, kappa))
        else:
            results.append((group_names, None))
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

def generate_report(kappa_results, pdf_filename="report.pdf"):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    pdf.cell(200, 10, txt="Fleiss' Kappa Report", ln=True, align='C')
    pdf.ln(10)
    
    for group_names, kappa in kappa_results:
        group_str = ', '.join(group_names)
        if kappa is not None:
            if kappa <= 0.3:
                pdf.cell(200, 10, txt=f"Group: {group_str}", ln=True)
                pdf.cell(200, 10, txt=f"Kappa Score: {kappa:.4f} - Very Low Agreement", ln=True)
                pdf.ln(5)
                pdf.multi_cell(0, 10, "Recommendations:", align='L')
                pdf.multi_cell(0, 10, "- Comprehensive Criteria Overhaul with Group Input:\n  Facilitate a comprehensive overhaul of the evaluation criteria with active input from classified groups.", align='L')
                pdf.multi_cell(0, 10, "- Intensive Group-Based Training and Mentorship:\n  Implement intensive training programs tailored to the specific needs of each classified group, focusing on the correct application of criteria and addressing common misconception.\n  Pair grouped with experienced mentors who can provide guidance and support.", align='L')
            elif 0.3 < kappa <= 0.5:
                pdf.cell(200, 10, txt=f"Group: {group_str}", ln=True)
                pdf.cell(200, 10, txt=f"Kappa Score: {kappa:.4f} - Low Agreement", ln=True)
                pdf.ln(5)
                pdf.multi_cell(0, 10, "Recommendations:", align='L')
                pdf.multi_cell(0, 10, "- Redesign Criteria with Group Collaboration:\n  Engage classified groups in a collaborative process to redesign and clarify evaluation criteria. Involve all group members in discussing potential ambiguities and revising criteria to ensure clarity and relevance.", align='L')
                pdf.multi_cell(0, 10, "- Group Calibration Workshops:\n  Conduct calibration workshops within each classified group, where members collaboratively evaluate sample items and discuss their decisions.", align='L')
            elif 0.5 < kappa <= 0.7:
                pdf.cell(200, 10, txt=f"Group: {group_str}", ln=True)
                pdf.cell(200, 10, txt=f"Kappa Score: {kappa:.4f} - Moderate Agreement", ln=True)
                pdf.ln(5)
                pdf.multi_cell(0, 10, "Recommendations:", align='L')
                pdf.multi_cell(0, 10, "- Facilitate In-Depth Group Discussions:\n  Organize in-depth discussions between classified group members to explore areas where disagreements occur. Encourage group members to share their interpretations and reasoning, aiming to develop a consensus on evaluation criteria.", align='L')
                pdf.multi_cell(0, 10, "- Peer Learning and Feedback:\n  Implement a peer learning system within each classified group where members review and provide feedback on each other's evaluations. This can help identify subtle misunderstandings and promote consistency.", align='L')
            else:
                pdf.cell(200, 10, txt=f"Group: {group_str}", ln=True)
                pdf.cell(200, 10, txt=f"Kappa Score: {kappa:.4f} - High Agreement", ln=True)
        else:
            pdf.cell(200, 10, txt=f"Group: {group_str}", ln=True)
        pdf.ln(10)
    
    pdf.output(pdf_filename)
    return pdf_filename

st.title("Fleiss' Kappa Calculator with PDF Report")

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
        
        if st.button("Calculate and Generate Report"):
            try:
                # Calculate the overall Fleiss' Kappa before any grouping
                matrix = create_contingency_matrix(ratings, 2)
                overall_kappa = compute_fleiss_kappa(matrix)
                
                st.subheader("Overall Fleiss' Kappa")
                st.write(f"Fleiss' kappa for all reviewers: {overall_kappa:.4f}")
                
                # Grouping and calculating Kappa for groups
                groups = classify_reviewers(ratings, threshold)
                
                if len(groups) == 0 and n_reviewers == 2:
                    st.subheader("Classification Result")
                    st.write("The team is not in harmony.")
                else:
                    kappa_results = get_group_kappas(groups, ratings, reviewer_arrays)

                    # Generate PDF report
                    pdf_filename = generate_report(kappa_results)
                    with open(pdf_filename, "rb") as file:
                        st.download_button(label="Download Report", data=file, file_name=pdf_filename, mime="application/pdf")

            except Exception as e
