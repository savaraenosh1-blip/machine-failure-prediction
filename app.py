from flask import Flask, render_template, request, send_file
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
import datetime, uuid
import pickle, json, io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap
import numpy as np

app = Flask(__name__)

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("columns.json", "r") as f:
    columns = json.load(f)


@app.route('/')
def home():
    return render_template("index.html")


@app.route('/predict', methods=['POST'])
def predict():
    try:
        air_temp = float(request.form.get('air_temp', 0))
        process_temp = float(request.form.get('process_temp', 0))
        rpm = float(request.form.get('rpm', 0))
        torque = float(request.form.get('torque', 0))
        tool_wear = float(request.form.get('tool_wear', 0))
        machine_type = request.form.get('type')

        type_l = 1 if machine_type == "L" else 0
        type_m = 1 if machine_type == "M" else 0
        type_h = 1 if machine_type == "H" else 0

        data_dict = {
            "Air temperature [K]": air_temp,
            "Process temperature [K]": process_temp,
            "Rotational speed [rpm]": rpm,
            "Torque [Nm]": torque,
            "Tool wear [min]": tool_wear,
            "Type_L": type_l,
            "Type_M": type_m,
            "Type_H": type_h
        }

        input_data = [data_dict.get(col, 0) for col in columns]

        prediction = model.predict([input_data])[0]
        prob = model.predict_proba([input_data])[0][1]
        confidence = int(round(prob * 100))

        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(np.array([input_data]))
            if isinstance(shap_values, list):
                sv = shap_values[1][0].tolist()
            else:
                if len(shap_values.shape) == 3:
                    sv = shap_values[0, :, 1].tolist()
                else:
                    sv = shap_values[0].tolist()
        except Exception as e:
            print("SHAP Error:", e)
            sv = [0] * len(columns)

        risk = "HIGH 🔴" if confidence > 80 else "MEDIUM 🟡" if confidence > 50 else "LOW 🟢"
        status = "⚠️ Failure Detected" if prediction == 1 else "✅ Machine Healthy"

        root_causes = []
        recommendations = []
        
        if torque > 50:
            root_causes.append("High Torque Detected")
            recommendations.extend(["Reduce operational load immediately", "Check motor configuration for overload", "Inspect drive shaft for mechanical resistance"])
        
        if tool_wear > 200:
            root_causes.append("Critical Tool Wear")
            recommendations.extend(["Schedule immediate tool replacement", "Inspect machined parts for quality degradation", "Review cutting speeds to prevent premature wear"])
            
        if rpm > 2000:
            root_causes.append("Excessive Rotational Speed")
            recommendations.extend(["Reduce RPM to safe operating limits", "Check spindle bearings for overheating", "Verify control system speed settings"])
            
        if process_temp > air_temp + 20:
            root_causes.append("Abnormal Temperature Gradient (Overheating)")
            recommendations.extend(["Inspect cooling system for blockages", "Verify coolant fluid levels and flow rate", "Check heat dissipation mechanisms"])

        if prediction == 1 and not root_causes:
            root_causes.append("Complex Multivariable Anomaly")
            recommendations.extend(["Perform full system diagnostic", "Review recent maintenance logs", "Monitor machine closely for next 24 hours"])
        elif not root_causes:
            root_causes.append("None (System Healthy)")
            recommendations.extend(["Continue standard operating procedures", "Perform scheduled preventive maintenance", "Maintain routine monitoring"])

        unique_recs = []
        for r in recommendations:
            if r not in unique_recs:
                unique_recs.append(r)
        recommendations = unique_recs

        importance = model.feature_importances_.tolist() if hasattr(model,"feature_importances_") else [0]*len(columns)

        return render_template("index.html",
            status=status, risk=risk, confidence=confidence,
            root_causes=root_causes, recommendations=recommendations,
            air_temp=air_temp, process_temp=process_temp,
            rpm=rpm, torque=torque, tool_wear=tool_wear,
            importance=importance, labels=columns,
            machine_type=machine_type, shap_values=sv)

    except Exception as e:
        return render_template("index.html", error=str(e))


@app.route('/download')
def download():
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    content = []

    status = request.args.get('status', 'Unknown')
    risk = request.args.get('risk', 'Unknown')
    confidence = int(request.args.get('confidence', 0))

    air = float(request.args.get('air', 0))
    process = float(request.args.get('process', 0))
    rpm = float(request.args.get('rpm', 0))
    torque = float(request.args.get('torque', 0))
    wear = float(request.args.get('wear', 0))
    machine_type = request.args.get('type', 'M')

    recs = request.args.getlist('rec')
    rcs = request.args.getlist('rc')

    # Calculate SHAP values for the PDF
    type_l = 1 if machine_type == "L" else 0
    type_m = 1 if machine_type == "M" else 0
    type_h = 1 if machine_type == "H" else 0

    data_dict = {
        "Air temperature [K]": air,
        "Process temperature [K]": process,
        "Rotational speed [rpm]": rpm,
        "Torque [Nm]": torque,
        "Tool wear [min]": wear,
        "Type_L": type_l,
        "Type_M": type_m,
        "Type_H": type_h
    }
    input_data = [data_dict.get(col, 0) for col in columns]

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(np.array([input_data]))
        if isinstance(shap_values, list):
            sv = shap_values[1][0].tolist()
        else:
            if len(shap_values.shape) == 3:
                sv = shap_values[0, :, 1].tolist()
            else:
                sv = shap_values[0].tolist()
    except:
        sv = [0] * len(columns)

    # Document Header
    title_style = styles['Heading1']
    title_style.alignment = 1 # Center
    content.append(Paragraph("<b>Predictive Analytics Engine</b>", title_style))
    content.append(Paragraph("<b>Machine Diagnostics & Failure Report</b>", styles['Title']))
    content.append(Spacer(1, 15))

    report_id = str(uuid.uuid4())[:8].upper()
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    content.append(Paragraph(f"<b>Generated On:</b> {timestamp} &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <b>Report ID:</b> {report_id}", styles['Normal']))
    content.append(Spacer(1, 20))

    # Status
    content.append(Paragraph(f"<b>Prediction Status:</b> {status}", styles['Normal']))
    content.append(Paragraph(f"<b>Risk Level:</b> {risk}", styles['Normal']))
    content.append(Paragraph(f"<b>Confidence:</b> {confidence}%", styles['Normal']))
    content.append(Spacer(1, 20))

    # Sensor Inputs Table
    content.append(Paragraph("<b>Sensor Input Data:</b>", styles['Heading3']))
    content.append(Spacer(1, 5))
    table_data = [
        ["Parameter", "Value"],
        ["Air Temperature [K]", str(air)],
        ["Process Temperature [K]", str(process)],
        ["Rotational Speed [rpm]", str(rpm)],
        ["Torque [Nm]", str(torque)],
        ["Tool Wear [min]", str(wear)],
        ["Machine Type", machine_type]
    ]
    t = Table(table_data, colWidths=[200, 150])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#1e293b")),
        ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0,0), (-1,0), 12),
        ('BACKGROUND', (0,1), (-1,-1), colors.HexColor("#f8fafc")),
        ('GRID', (0,0), (-1,-1), 1, colors.HexColor("#cbd5e1")),
    ]))
    content.append(t)
    content.append(Spacer(1, 25))

    # SHAP Chart
    content.append(Paragraph("<b>Explainable AI (Local Feature Impact):</b>", styles['Heading3']))
    content.append(Spacer(1, 5))
    
    plt.figure(figsize=(7, 3.5))
    colors_bar = ['#ef4444' if v > 0 else '#22c55e' for v in sv]
    plt.barh(columns, sv, color=colors_bar)
    plt.xlabel('SHAP Value (Impact on Failure Risk)')
    plt.tight_layout()
    img_shap = io.BytesIO()
    plt.savefig(img_shap, format='png', dpi=150)
    plt.close()
    img_shap.seek(0)
    
    content.append(Image(img_shap, width=450, height=225))
    content.append(Spacer(1, 25))
    
    # Causes and Recs
    content.append(Paragraph("<b>Root Causes:</b>", styles['Heading3']))
    for rc in rcs:
        content.append(Paragraph(f"• {rc}", styles['Normal']))
        
    content.append(Spacer(1, 15))
    content.append(Paragraph("<b>Recommendations:</b>", styles['Heading3']))
    for r in recs:
        content.append(Paragraph(f"• {r}", styles['Normal']))

    doc.build(content)
    buffer.seek(0)

    return send_file(buffer, as_attachment=True, download_name="machine_diagnostics_report.pdf")


if __name__ == "__main__":
    app.run(debug=True)