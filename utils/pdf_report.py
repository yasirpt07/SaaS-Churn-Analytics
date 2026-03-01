from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.platypus import Table
import io

def generate_pdf(buffer, risk_percent, segment):

    doc = SimpleDocTemplate(buffer)
    elements = []
    styles = getSampleStyleSheet()

    elements.append(Paragraph("SaaS Customer Churn Report", styles['Title']))
    elements.append(Spacer(1, 20))

    elements.append(Paragraph(f"Churn Probability: {risk_percent}%", styles['Normal']))
    elements.append(Spacer(1, 10))

    elements.append(Paragraph(f"Customer Segment: {segment}", styles['Normal']))
    elements.append(Spacer(1, 20))

    table_data = [
        ["Metric", "Value"],
        ["Churn Probability", f"{risk_percent}%"],
        ["Segment", segment]
    ]

    table = Table(table_data)
    elements.append(table)

    doc.build(elements)