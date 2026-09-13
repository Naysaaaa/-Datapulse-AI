import os
import re
from io import BytesIO
from datetime import datetime
from typing import Dict, Any, List

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

class PDFReportGenerator:
    """
    Generates a production-ready, highly polished PDF report containing
    the Ollama executive narrative, system alerts, and MLOps drift status.
    """
    
    @staticmethod
    def _clean_markdown(text: str) -> str:
        """
        Converts basic markdown syntax (like headers and bold text) 
        into ReportLab-supported XML tags (<b>, <i>, <font>).
        """
        # Convert header lines like '### 1. Title' to HTML bold formatting
        text = re.sub(r'###\s+(.*)', r'<font size="12" color="#0F172A"><b>\1</b></font>', text)
        text = re.sub(r'##\s+(.*)', r'<font size="14" color="#0F172A"><b>\1</b></font>', text)
        text = re.sub(r'#\s+(.*)', r'<font size="16" color="#0F172A"><b>\1</b></font>', text)
        
        # Convert markdown bold '**text**' to '<b>text</b>'
        text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)
        
        # Convert markdown italic '*text*' to '<i>text</i>'
        text = re.sub(r'\*(.*?)\*', r'<i>\1</i>', text)
        
        # Convert bullet lists '- item' to '• item'
        text = re.sub(r'^\s*-\s+(.*)', r'• \1', text, flags=re.MULTILINE)
        
        # Replace newlines with break tags, but handle double newlines as paragraph boundaries
        text = text.replace('\n\n', '<br/><br/>')
        text = text.replace('\n', '<br/>')
        
        return text

    def generate_report(self, data: Dict[str, Any]) -> BytesIO:
        """
        Builds the PDF document in memory.
        """
        buffer = BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=letter,
            rightMargin=54,
            leftMargin=54,
            topMargin=54,
            bottomMargin=54
        )
        
        # Custom color palette (Slate / Teal theme)
        primary_color = colors.HexColor("#0F172A")    # Dark Slate
        secondary_color = colors.HexColor("#0EA5E9")  # Ocean Blue
        accent_color = colors.HexColor("#64748B")     # Slate Gray
        bg_light = colors.HexColor("#F8FAFC")         # Off-white
        border_color = colors.HexColor("#E2E8F0")     # Light gray border
        
        # Setup styles
        styles = getSampleStyleSheet()
        
        # Modify existing styles to avoid conflicts
        styles['Normal'].textColor = primary_color
        styles['Normal'].fontSize = 10
        styles['Normal'].leading = 14
        
        title_style = ParagraphStyle(
            'ReportTitle',
            parent=styles['Normal'],
            fontName='Helvetica-Bold',
            fontSize=24,
            leading=28,
            textColor=primary_color,
            spaceAfter=6
        )
        
        subtitle_style = ParagraphStyle(
            'ReportSubtitle',
            parent=styles['Normal'],
            fontName='Helvetica',
            fontSize=10,
            leading=14,
            textColor=accent_color,
            spaceAfter=20
        )
        
        section_heading = ParagraphStyle(
            'SectionHeading',
            parent=styles['Normal'],
            fontName='Helvetica-Bold',
            fontSize=14,
            leading=18,
            textColor=primary_color,
            spaceBefore=15,
            spaceAfter=8,
            keepWithNext=True
        )
        
        body_style = ParagraphStyle(
            'BodyTextCustom',
            parent=styles['Normal'],
            fontName='Helvetica',
            fontSize=10,
            leading=15,
            spaceAfter=10
        )
        
        th_style = ParagraphStyle(
            'TableHeader',
            parent=styles['Normal'],
            fontName='Helvetica-Bold',
            fontSize=10,
            leading=12,
            textColor=colors.white
        )
        
        td_style = ParagraphStyle(
            'TableCell',
            parent=styles['Normal'],
            fontName='Helvetica',
            fontSize=9,
            leading=12
        )
        
        story = []
        
        # --- HEADER SECTION ---
        story.append(Paragraph("DATAPULSE AI", title_style))
        report_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        story.append(Paragraph(f"EXECUTIVE REPORT & MLOPS AUDIT  |  GENERATED: {report_time}", subtitle_style))
        
        # Draw a horizontal rule
        hr_table = Table([[""]], colWidths=[doc.width], rowHeights=[2])
        hr_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), secondary_color),
            ('BOTTOMPADDING', (0,0), (-1,-1), 0),
            ('TOPPADDING', (0,0), (-1,-1), 0),
        ]))
        story.append(hr_table)
        story.append(Spacer(1, 15))
        
        # --- OVERVIEW METRICS BLOCK ---
        telemetry = data.get("latest_telemetry", {})
        system_status = data.get("system_status", "UNKNOWN")
        
        status_color = "#22C55E" # Green
        if system_status == "DEGRADED":
            status_color = "#EAB308" # Yellow
        elif system_status == "DRIFTING":
            status_color = "#EF4444" # Red
            
        overview_data = [
            [
                Paragraph("<b>System Status</b>", td_style),
                Paragraph(f"<font color='{status_color}'><b>{system_status}</b></font>", td_style),
                Paragraph("<b>Ingested Events</b>", td_style),
                Paragraph(str(data.get("total_points", 0)), td_style),
            ],
            [
                Paragraph("<b>CPU Load</b>", td_style),
                Paragraph(f"{telemetry.get('cpu_utilization', 0.0):.1f}%", td_style),
                Paragraph("<b>Memory Load</b>", td_style),
                Paragraph(f"{telemetry.get('memory_utilization', 0.0):.1f}%", td_style),
            ],
            [
                Paragraph("<b>Network Latency</b>", td_style),
                Paragraph(f"{telemetry.get('network_latency', 0.0):.1f}ms", td_style),
                Paragraph("<b>Error Rate</b>", td_style),
                Paragraph(f"{telemetry.get('error_rate', 0.0):.2f}%", td_style),
            ]
        ]
        
        overview_table = Table(overview_data, colWidths=[1.5*inch, 1.7*inch, 1.5*inch, 1.7*inch])
        overview_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), bg_light),
            ('ALIGN', (0,0), (-1,-1), 'LEFT'),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('BOTTOMPADDING', (0,0), (-1,-1), 6),
            ('TOPPADDING', (0,0), (-1,-1), 6),
            ('LEFTPADDING', (0,0), (-1,-1), 10),
            ('RIGHTPADDING', (0,0), (-1,-1), 10),
            ('BOX', (0,0), (-1,-1), 1, border_color),
            ('INNERGRID', (0,0), (-1,-1), 0.5, border_color),
        ]))
        story.append(overview_table)
        story.append(Spacer(1, 20))
        
        # --- OLLAMA EXECUTIVE SUMMARY NARRATIVE ---
        story.append(Paragraph("Executive Narrative", section_heading))
        narrative_raw = data.get("narrative", "No narrative available.")
        narrative_html = self._clean_markdown(narrative_raw)
        story.append(Paragraph(narrative_html, body_style))
        story.append(Spacer(1, 15))
        
        # --- MLOPS DRIFT AUDIT TABLE ---
        story.append(Paragraph("MLOps Feature Drift Audit", section_heading))
        feature_psi = data.get("feature_psi", {})
        
        drift_rows = [[
            Paragraph("Feature Name", th_style),
            Paragraph("Population Stability Index (PSI)", th_style),
            Paragraph("Status Threshold Check", th_style)
        ]]
        
        for feature, psi in feature_psi.items():
            status_text = "Stable"
            text_color = "#22C55E"
            if psi >= 0.25:
                status_text = "Drifting (Action Required)"
                text_color = "#EF4444"
            elif psi >= 0.10:
                status_text = "Moderate Shift"
                text_color = "#EAB308"
                
            drift_rows.append([
                Paragraph(feature, td_style),
                Paragraph(f"{psi:.4f}", td_style),
                Paragraph(f"<font color='{text_color}'><b>{status_text}</b></font>", td_style)
            ])
            
        # Prediction drift
        pred_psi = data.get("prediction_drift", 0.0)
        pred_status = "Stable"
        pred_color = "#22C55E"
        if pred_psi >= 0.25:
            pred_status = "Drifting (Action Required)"
            pred_color = "#EF4444"
        elif pred_psi >= 0.10:
            pred_status = "Moderate Shift"
            pred_color = "#EAB308"
            
        drift_rows.append([
            Paragraph("<b>Model Health Prediction</b>", td_style),
            Paragraph(f"{pred_psi:.4f}", td_style),
            Paragraph(f"<font color='{pred_color}'><b>{pred_status}</b></font>", td_style)
        ])
        
        drift_table = Table(drift_rows, colWidths=[2.2*inch, 2.0*inch, 2.2*inch])
        drift_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), primary_color),
            ('ALIGN', (0,0), (-1,-1), 'LEFT'),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('BOTTOMPADDING', (0,0), (-1,-1), 6),
            ('TOPPADDING', (0,0), (-1,-1), 6),
            ('LEFTPADDING', (0,0), (-1,-1), 8),
            ('GRID', (0,0), (-1,-1), 0.5, border_color),
            ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, bg_light])
        ]))
        story.append(drift_table)
        story.append(Spacer(1, 15))
        
        # --- ACTIVE ALERTS ---
        alerts = data.get("active_alerts", [])
        if alerts:
            story.append(Paragraph("Triggered Alerts & Model Failures", section_heading))
            alert_rows = [[
                Paragraph("Severity", th_style),
                Paragraph("Alert Notification Details", th_style)
            ]]
            
            for alert in alerts:
                sev = "CRITICAL" if "drift" in alert.lower() or "degraded" in alert.lower() or "failure" in alert.lower() else "WARNING"
                sev_color = "#EF4444" if sev == "CRITICAL" else "#EAB308"
                
                alert_rows.append([
                    Paragraph(f"<font color='{sev_color}'><b>{sev}</b></font>", td_style),
                    Paragraph(alert, td_style)
                ])
                
            alert_table = Table(alert_rows, colWidths=[1.5*inch, 4.9*inch])
            alert_table.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), primary_color),
                ('ALIGN', (0,0), (-1,-1), 'LEFT'),
                ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
                ('BOTTOMPADDING', (0,0), (-1,-1), 6),
                ('TOPPADDING', (0,0), (-1,-1), 6),
                ('LEFTPADDING', (0,0), (-1,-1), 8),
                ('GRID', (0,0), (-1,-1), 0.5, border_color),
                ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, bg_light])
            ]))
            story.append(alert_table)
            
        doc.build(story)
        buffer.seek(0)
        return buffer

pdf_report_generator = PDFReportGenerator()
