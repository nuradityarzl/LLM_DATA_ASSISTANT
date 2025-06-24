
# streamlit_app_ai_insight.py
import streamlit as st
import pandas as pd
import openai
import time
import gspread
import json
import requests
from oauth2client.service_account import ServiceAccountCredentials
import altair as alt
from pandasql import sqldf

st.set_page_config(page_title="LLM Data Assistant", layout="wide")

# ---------------- Sidebar ----------------
st.sidebar.title("⚙️ Konfigurasi")

llm_provider = st.sidebar.selectbox("Pilih Model", ["OpenAI (ChatGPT)", "Claude", "Gemini", "DeepSeek"])
api_key = st.sidebar.text_input("API Key", type="password")

source_type = st.sidebar.radio("Sumber Data", ["Google Sheets", "Excel/CSV"])

sheet_data = None
schema_info = ""

if source_type == "Google Sheets":
    st.sidebar.subheader("Google Sheets")
    sheet_url = st.sidebar.text_input("URL Google Sheet")
    creds_json = st.sidebar.file_uploader("Upload Credential JSON", type="json")
    if creds_json and sheet_url:
        try:
            scope = ['https://spreadsheets.google.com/feeds','https://www.googleapis.com/auth/drive']
            creds_dict = json.load(creds_json)
            creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
            client = gspread.authorize(creds)
            sheet = client.open_by_url(sheet_url).sheet1
            sheet_data = pd.DataFrame(sheet.get_all_records())
            schema_info = "\n".join([f"{col} (string)" for col in sheet_data.columns])
            st.sidebar.success("🟢 Terkoneksi ke Google Sheets")
        except Exception as e:
            st.sidebar.error(f"❌ Gagal: {e}")

else:
    st.sidebar.subheader("Excel / CSV")
    uploaded_file = st.sidebar.file_uploader("Upload File", type=["csv", "xlsx"])
    if uploaded_file:
        try:
            if uploaded_file.name.endswith(".csv"):
                sheet_data = pd.read_csv(uploaded_file)
            else:
                sheet_data = pd.read_excel(uploaded_file)
            schema_info = "\n".join([f"{col} (string)" for col in sheet_data.columns])
            sheet_data.columns = [col.replace(" ", "_") for col in sheet_data.columns]

            st.dataframe(sheet_data, use_container_width=True)
            st.sidebar.success("🟢 File Dibaca")
        except Exception as e:
            st.sidebar.error(f"❌ Gagal membaca file: {e}")

def clean_sql_output(raw_sql: str) -> str:
    cleaned = raw_sql.strip()
    if cleaned.startswith("```sql"):
        cleaned = cleaned.replace("```sql", "").strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.replace("```", "").strip()
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3].strip()
    return cleaned

# ---------------- Prompt to SQL ----------------
def generate_sql(query, schema_info):
    if not api_key:
        st.error("❗ Masukkan API Key.")
        return ""

    prompt = f"""
You are a data analyst assistant using SQLite syntax (for use with pandasql in Python). 
Given the user's question and the simplified table schema below (with underscore column names), 
write a SELECT SQL query that answers the question using the 'sheet_data' table.

Schema:
{schema_info}

Note: Use 'sheet_data' as the table name when writing SQL.

Pertanyaan: {query}
SQL:
"""

    try:
        if llm_provider.startswith("OpenAI"):
            client = openai.OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "Kamu adalah pembuat SQL handal."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0
            )
            return response.choices[0].message.content.strip()
        elif llm_provider.startswith("Claude"):
            response = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "claude-3-opus-20240229",
                    "max_tokens": 1024,
                    "messages": [{"role": "user", "content": prompt}]
                }
            )
            return response.json()["content"][0]["text"].strip()
        elif llm_provider.startswith("Gemini"):
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={api_key}"
            body = {"contents": [{"parts": [{"text": prompt}]}]}
            response = requests.post(url, json=body)
            return response.json()['candidates'][0]['content']['parts'][0]['text']
        elif llm_provider.startswith("DeepSeek"):
            response = requests.post(
                "https://api.deepseek.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "deepseek-chat",
                    "messages": [
                        {"role": "system", "content": "Kamu adalah pembuat SQL handal."},
                        {"role": "user", "content": prompt}
                    ]
                }
            )
            return response.json()['choices'][0]['message']['content'].strip()
    except Exception as e:
        return f"-- Gagal membuat SQL: {e}"

# ---------------- Insight AI ----------------
def generate_insight(df, query_text):
    if df.empty or not api_key:
        return "Tidak ada data atau API Key tidak valid."

    sample_csv = df.head(20).to_csv(index=False)
    prompt = f"""
Berdasarkan data CSV berikut (maks 20 baris), dan pertanyaan: "{query_text}", berikan 2–3 insight bisnis yang relevan dan bernilai. Ringkas dan profesional.

Data:
{sample_csv}

Insight:
"""
    try:
        if llm_provider.startswith("OpenAI"):
            client = openai.OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "Kamu adalah analis bisnis profesional."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=300
            )
            return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Gagal menghasilkan insight: {e}"

# ---------------- UI ----------------
st.title("📊 LLM Data Assistant")
st.write("Ajukan pertanyaan dan dapatkan jawaban dari datamu.")

query = st.text_area("Pertanyaan tentang data", placeholder="Contoh: Tampilkan total penjualan per produk", height=100)

if st.button("🔍 Analisis"):
    if not query or sheet_data is None:
        st.warning("Harap lengkapi input dan koneksi data.")
    else:
        #sql = generate_sql(query, schema_info)
        sql_query_raw = generate_sql(query, schema_info)
        sql = clean_sql_output(sql_query_raw)
        if sql:
            st.markdown("#### 🔢 SQL yang dihasilkan")
            st.code(sql, language="sql")

            try:
                df = sqldf(sql, {"sheet_data": sheet_data})
                st.markdown("#### 🧾 Hasil Query")
                st.dataframe(df, use_container_width=True)

                # Visualisasi sederhana
                if df.shape[1] == 2:
                    col1, col2 = df.columns
                    if df[col2].dtype in ["int64", "float64"]:
                        chart = alt.Chart(df).mark_bar().encode(x=col1, y=col2, tooltip=[col1, col2])
                        st.altair_chart(chart.properties(width=700), use_container_width=True)

                st.markdown("#### 💡 Insight business")
                insight = generate_insight(df, query)
                st.info(insight)

                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button("⬇️ Download CSV", csv, "hasil.csv", "text/csv")
            except Exception as e:
                st.error(f"Query error: {e}")
