from fpdf import FPDF
import os

def generate_report(data_summary, eda_plots, model_results, confusion_matrices, model_comparison_chart):
    report_path = 'static/report.pdf'
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font('Arial', size=12)

    # Add Title
    pdf.set_font('Arial', 'B', size=16)
    pdf.cell(200, 10, 'Data Analysis Report', ln=True, align='C')
    pdf.ln(10)  # Add a line break

    # Add Data Summary Section with Better Formatting
    pdf.set_font('Arial', 'B', size=14)
    pdf.cell(200, 10, 'Data Summary', ln=True, align='L')
    pdf.set_font('Arial', size=12)

    if data_summary:
        for key, value in data_summary.items():
            pdf.cell(200, 10, f"{key}:", ln=True)
            pdf.multi_cell(0, 10, f"  {value}", align='L')  # Multi-line cell for better readability
            pdf.ln(5)  # Line break between each summary item
    else:
        pdf.cell(200, 10, "No data summary available.", ln=True)

    # Add EDA plots
    if eda_plots:
        for plot in eda_plots:
            pdf.add_page()
            pdf.image(plot, x=10, y=20, w=190)
    else:
        pdf.add_page()
        pdf.cell(200, 10, "No EDA plots available.", ln=True)

    # Add Model Results
    pdf.add_page()
    pdf.set_font('Arial', 'B', size=14)
    pdf.cell(200, 10, 'Model Results', ln=True, align='L')
    pdf.set_font('Arial', size=12)
    
    if model_results:
        for model, report in model_results.items():
            pdf.cell(200, 10, f"Model: {model}", ln=True)
            for metric, score in report.items():
                pdf.cell(200, 10, f"  {metric}: {score}", ln=True)
            pdf.ln(5)  # Add space between models
    else:
        pdf.cell(200, 10, "No model results available.", ln=True)

    # Add Confusion Matrices
    if confusion_matrices:
        for model, cm_path in confusion_matrices.items():
            pdf.add_page()
            pdf.cell(200, 10, f"Confusion Matrix for {model}", ln=True, align='C')
            pdf.image(cm_path, x=10, y=20, w=190)
    else:
        pdf.add_page()
        pdf.cell(200, 10, "No confusion matrices available.", ln=True)

    # Add Model Comparison Chart
    if model_comparison_chart:
        pdf.add_page()
        pdf.cell(200, 10, 'Model Comparison', ln=True, align='C')
        pdf.image(model_comparison_chart, x=10, y=20, w=190)
    else:
        pdf.add_page()
        pdf.cell(200, 10, "No model comparison chart available.", ln=True)

    # Save the PDF
    pdf.output(report_path)

    return report_path
