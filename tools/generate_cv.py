"""Generate an ATS-friendly CV (.docx) for Daren Munene.

Design choices for ATS compatibility:
- Single-column layout, no text boxes, no tables for content, no images.
- Standard fonts (Calibri), standard bullet lists, real heading text.
- Contact details as plain selectable text.
- Section headers in a consistent, parseable style.
Content is condensed to fit within three pages.
"""
from docx import Document
from docx.shared import Pt, RGBColor, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

ACCENT = RGBColor(0x1F, 0x3A, 0x5F)   # navy
INK = RGBColor(0x20, 0x20, 0x20)

doc = Document()

# ---- Global defaults ----
style = doc.styles["Normal"]
style.font.name = "Calibri"
style.font.size = Pt(10)
style.font.color.rgb = INK
style.paragraph_format.space_after = Pt(2)
style.paragraph_format.line_spacing = 1.0

for section in doc.sections:
    section.top_margin = Inches(0.5)
    section.bottom_margin = Inches(0.5)
    section.left_margin = Inches(0.6)
    section.right_margin = Inches(0.6)


def _no_space(p):
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(0)


def add_name(text):
    p = doc.add_paragraph()
    _no_space(p)
    r = p.add_run(text)
    r.bold = True
    r.font.size = Pt(20)
    r.font.color.rgb = ACCENT


def add_contact(text):
    p = doc.add_paragraph()
    _no_space(p)
    p.paragraph_format.space_after = Pt(4)
    r = p.add_run(text)
    r.font.size = Pt(9.5)
    r.font.color.rgb = RGBColor(0x44, 0x44, 0x44)


def add_heading(text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(7)
    p.paragraph_format.space_after = Pt(2)
    r = p.add_run(text.upper())
    r.bold = True
    r.font.size = Pt(11)
    r.font.color.rgb = ACCENT
    # bottom border
    pPr = p._p.get_or_add_pPr()
    pbdr = OxmlElement("w:pBdr")
    bottom = OxmlElement("w:bottom")
    bottom.set(qn("w:val"), "single")
    bottom.set(qn("w:sz"), "6")
    bottom.set(qn("w:space"), "1")
    bottom.set(qn("w:color"), "1F3A5F")
    pbdr.append(bottom)
    pPr.append(pbdr)


def add_role(title, org, dates):
    p = doc.add_paragraph()
    _no_space(p)
    p.paragraph_format.space_before = Pt(4)
    r = p.add_run(title)
    r.bold = True
    r.font.size = Pt(10.5)
    sub = doc.add_paragraph()
    _no_space(sub)
    rr = sub.add_run(org)
    rr.italic = True
    rr.font.size = Pt(9.5)
    rr.font.color.rgb = RGBColor(0x44, 0x44, 0x44)
    rr2 = sub.add_run(f"   |   {dates}")
    rr2.font.size = Pt(9.5)
    rr2.font.color.rgb = RGBColor(0x66, 0x66, 0x66)


def add_bullet(text, bold_lead=None):
    p = doc.add_paragraph(style="List Bullet")
    p.paragraph_format.space_after = Pt(1)
    p.paragraph_format.left_indent = Inches(0.22)
    p.paragraph_format.line_spacing = 1.0
    if bold_lead:
        r = p.add_run(bold_lead)
        r.bold = True
        r.font.size = Pt(10)
    r2 = p.add_run(text)
    r2.font.size = Pt(10)


def add_kv(label, value):
    p = doc.add_paragraph()
    _no_space(p)
    p.paragraph_format.left_indent = Inches(0.0)
    r = p.add_run(f"{label}: ")
    r.bold = True
    r.font.size = Pt(9.5)
    r2 = p.add_run(value)
    r2.font.size = Pt(9.5)


# ===================== HEADER =====================
add_name("Daren Munene Mungania")
add_contact("Software Engineer  •  Nairobi, Kenya  •  +254 795 354 727  •  "
            "darennesh77@gmail.com  •  LinkedIn  •  github.com/Darennesh")

# ===================== SUMMARY =====================
add_heading("Professional Summary")
p = doc.add_paragraph()
p.paragraph_format.space_after = Pt(2)
p.add_run(
    "Software Engineer and Data Analyst with hands-on experience building and "
    "operating high-availability, enterprise-scale platforms and delivering "
    "data-driven, machine-learning solutions. Skilled in microservices and "
    "RESTful API design, cloud-native deployment (AWS, Azure, Kubernetes), "
    "CI/CD automation, observability, and applied machine learning. Adept at "
    "integrating AI into production workflows to improve efficiency, "
    "reliability, and compliance across diverse, distributed teams."
).font.size = Pt(10)

# ===================== EXPERIENCE =====================
add_heading("Work Experience")

add_role("Software Engineer Intern — Research, Design & Innovation",
         "Safaricom PLC, Nairobi, Kenya", "May 2025 – Feb 2026")
add_bullet("an enterprise AIOps Root Cause Analysis (RCA/CAPA) tool for "
           "10,000+ vendors and customers, automating agent workflows and "
           "cutting incident resolution time by 60%.", "Led development of ")
add_bullet("25+ RESTful APIs across a microservices architecture "
           "(Node.js/Express, PostgreSQL, Sequelize ORM) delivering sub-200ms "
           "latency at 1,000+ concurrent users.", "Designed and documented ")
add_bullet("responsive, accessible UIs in React, Next.js and TypeScript with "
           "reusable components, plus unit/integration tests (Jest) enforced "
           "in CI/CD quality gates.", "Built ")
add_bullet("end-to-end CI/CD pipelines (Jenkins, GitLab CI) with SAST "
           "(SonarQube), SCA, container scanning and Policy-as-Code, achieving "
           "30% faster deployments.", "Implemented ")
add_bullet("containerized microservices on Kubernetes (AWS EKS, Azure AKS) "
           "with zero-downtime deployments; defined SLIs/SLOs and error "
           "budgets to sustain 99.99% availability.", "Deployed and operated ")
add_bullet("a full observability/AIOps platform (ELK, Grafana, Dynatrace, "
           "Prometheus) with automated alerting, improving traceability by "
           "70%.", "Built ")
add_bullet("infrastructure provisioning and scaling with Bash and Python, "
           "reducing manual operations by ~40%; managed identity and secrets "
           "via RBAC and OAuth2 (Microsoft Entra ID).", "Automated ")
add_bullet("incident management and on-call rotation, leading major-incident "
           "recovery and post-incident reviews; delivered in Agile/Scrum "
           "(Jira, Confluence).", "Drove ")

add_role("ICT Officer — Enhanced Continuous Voter Registration (ECVR I)",
         "Independent Electoral & Boundaries Commission (IEBC), Igembe North",
         "Mar – Apr 2026")
add_bullet("configured and operated KIEMS biometric kits and ICT equipment "
           "across multiple registration centres with zero-downtime data "
           "capture and VRS updates.", "Set up, ")
add_bullet("quantitative trend analysis of daily voter-registration data in "
           "Microsoft Excel (pivot tables, charts) and the IEBC Mapping Tool, "
           "identifying hotspots, low-uptake areas and anomalies.",
           "Performed ")
add_bullet("M&E trackers and Excel dashboards measuring daily indicators "
           "against targets, delivering periodic progress reports to "
           "constituency and government stakeholders.", "Designed ")
add_bullet("secure synchronization, transmission and backup of registration "
           "data per IEBC data-integrity and confidentiality standards; "
           "trained officials on ICT operations and reporting.", "Supported ")

# ===================== PROJECTS =====================
add_heading("Projects")

add_role("FIFA World Cup 2026 Match Prediction System",
         "Personal Project — Python, LightGBM, scikit-learn, Monte Carlo, CI/CD",
         "github.com/Darennesh/World-Cup-2026-Predictor")
add_bullet("an end-to-end machine learning system predicting 2026 FIFA World "
           "Cup match outcomes and simulating the full 48-team tournament, "
           "deployed as a live, daily-updated public web dashboard.", "Built ")
add_bullet("a leakage-free feature pipeline over 49,000+ historical "
           "international matches (1872–present) with as-of Elo ratings, "
           "recency-weighted form and contextual features under strict "
           "temporal validation.", "Engineered ")
add_bullet("a LightGBM Poisson goal model achieving 0.21 Ranked Probability "
           "Score (RPS) — 23% better than base rate — beating a Dixon-Coles "
           "statistical baseline in temporal cross-validation; used SHAP for "
           "interpretability and temperature scaling for calibration.",
           "Developed ")
add_bullet("model accuracy by back-testing the 2018 and 2022 World Cups "
           "(12.5% RPS improvement over baseline); reached 67% live group-stage "
           "outcome accuracy tracked via an automated walk-forward record.",
           "Validated ")
add_bullet("a 50,000-iteration Monte Carlo simulation encoding the official "
           "FIFA knockout bracket and tiebreaker rules to produce calibrated "
           "advancement probabilities for all 48 teams.", "Implemented ")
add_bullet("a responsive dashboard (HTML/CSS/JavaScript) with live fixtures, "
           "dark mode and an interactive bracket; automated daily rebuilds and "
           "deployment via GitHub Actions and Vercel, with 74 pytest unit "
           "tests including data-leakage and tournament-rule guards.",
           "Shipped ")

add_role("CAPA Root Cause Analysis Tool", "AI-Powered Incident Management",
         "Jun 2025 – Sep 2025")
add_bullet("an enterprise AI-powered RCA platform for 10,000+ users; "
           "integrated LangChain with multiple LLMs (GPT-4o, Llama 3, Gemma 3, "
           "DeepSeek-R1), improving RCA accuracy by 45% and cutting resolution "
           "time by 60%.", "Engineered ")
add_bullet("a scalable microservices stack (Node.js/Express, React/Next.js, "
           "PostgreSQL/Sequelize) with secure RBAC (JWT, OAuth, Microsoft "
           "Entra ID), handling 1,000+ concurrent users at sub-200ms latency.",
           "Architected ")
add_bullet("containerized services (Podman, Kubernetes) at 99.8% uptime with "
           "a full observability stack (ELK, Grafana, Dynatrace) and CI/CD "
           "(Jenkins, GitLab CI) with SAST/SCA and Policy-as-Code gates.",
           "Deployed ")

add_role("Enterprise Data Migration & Automation", "Data Engineering",
         "May 2024 – Aug 2024")
add_bullet("500GB+ of enterprise data at 99.9% integrity using Docker-based "
           "pipelines on AWS, with backup validation and DR testing; optimized "
           "PostgreSQL to reduce query times by 40%.", "Migrated ")
add_bullet("Python and Bash automation to orchestrate migration tasks, "
           "reducing manual effort and ensuring cross-environment "
           "consistency.", "Wrote ")

# ===================== SKILLS =====================
add_heading("Skills & Technologies")
add_kv("Programming", "Python, Java, JavaScript, TypeScript, Node.js, C/C++, SQL, R, Bash")
add_kv("Machine Learning & Data Science", "LightGBM, scikit-learn, TensorFlow, "
       "Gradient Boosting, Poisson Regression, Feature Engineering, Model "
       "Calibration, SHAP, Time-Series Cross-Validation, Monte Carlo "
       "Simulation, Probabilistic Forecasting, Ranked Probability Score (RPS)")
add_kv("Data & Analytics", "pandas, NumPy, SciPy, statsmodels, matplotlib, "
       "seaborn, Microsoft Excel (pivot tables, trend analysis), Data Analytics")
add_kv("AI", "LangChain, OpenAI, Azure AI Services, LLMs, AI Agents, Prompt Engineering")
add_kv("Frameworks", "Spring Boot, Django, React, Next.js, Express.js")
add_kv("DevSecOps & CI/CD", "Jenkins, GitLab CI, AWS CodePipeline, Maven, "
       "SonarQube (SAST), SCA, Policy-as-Code")
add_kv("Cloud & Containers", "AWS, Azure, Docker, Podman, Kubernetes (EKS, AKS), Rancher, Helm")
add_kv("IaC & Automation", "Terraform, Ansible, Bash, Python Automation")
add_kv("Observability", "ELK Stack, Grafana, Prometheus, Dynatrace, Splunk")
add_kv("Databases", "PostgreSQL, MySQL, MongoDB, MSSQL, Oracle, Redis")
add_kv("Big Data", "Hadoop, Spark, Kusto, COSMOS")
add_kv("Testing & Tools", "pytest, Jest, Git, GitLab, Postman, VS Code, IntelliJ, Linux, Jira, Confluence")

# ===================== EDUCATION =====================
add_heading("Education")
add_role("B.Sc. Computer Science", "Presbyterian University of East Africa", "2025")
add_role("High School Diploma", "Kagumo High School", "2021")

# ===================== CERTIFICATIONS =====================
add_heading("Certifications")
add_bullet("NVIDIA — Building LLM Applications with Prompt Engineering")
add_bullet("Certified Kubernetes Administrator (CKA) — In progress")
add_bullet("Kubernetes & Cloud Native Associate (KCNA) — Andela & Linux Foundation (Ongoing)")

out = r"C:\Users\user\OneDrive\Documents\Daren_Munene_CV.docx"
doc.save(out)
print(f"Saved: {out}")
