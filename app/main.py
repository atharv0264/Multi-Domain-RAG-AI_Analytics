import streamlit as st
import pandas as pd
import ollama
import os
from pypdf import PdfReader
import pdfplumber 


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Multi-Domain AI Analytics",
    page_icon="📊",
    layout="wide"
)


# ============================================================
# CUSTOM UI STYLE
# ============================================================

st.markdown("""
<style>

    /* Main title */
    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 18px;
        margin-bottom: 25px;
    }

    /* Dashboard cards */
    .metric-card {
        padding: 20px;
        border-radius: 12px;
        border: 1px solid rgba(128,128,128,0.25);
        text-align: center;
        margin-bottom: 10px;
    }

    .metric-value {
        font-size: 30px;
        font-weight: 700;
    }

    .metric-label {
        font-size: 15px;
        opacity: 0.75;
    }

    /* AI answer */
    .ai-answer {
        padding: 18px;
        border-radius: 10px;
        border: 1px solid rgba(128,128,128,0.25);
        font-size: 17px;
        margin-top: 10px;
        margin-bottom: 15px;
    }

</style>
""", unsafe_allow_html=True)


# ============================================================
# TITLE
# ============================================================

st.markdown(
    '<div class="main-title">📊 Multi-Domain AI Analytics</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Upload your data and interact with it using Local AI.'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# SETTINGS
# ============================================================

MODEL_NAME = "llama3.2:3b"


# ============================================================
# CHECK OLLAMA
# ============================================================

def check_ollama():
    try:
        ollama.list()
        return True
    except Exception:
        return False


# ============================================================
# LOAD CSV
# ============================================================

def load_csv(uploaded_file):
    try:
        df = pd.read_csv(uploaded_file)
        return df, None
    except Exception as e:
        return None, str(e)

# ============================================================
# EXTRACT TABLES FROM PDF
# ============================================================

def extract_pdf_tables(uploaded_file):
    tables = []

    try:
        with pdfplumber.open(uploaded_file) as pdf:

            for page_number, page in enumerate(pdf.pages, start=1):

                extracted_tables = page.extract_tables()

                for table_number, table in enumerate(
                    extracted_tables,
                    start=1
                ):

                    if not table:
                        continue

                    # Remove completely empty rows
                    cleaned_rows = []

                    for row in table:
                        if row and any(
                            cell is not None and str(cell).strip() != ""
                            for cell in row
                        ):
                            cleaned_rows.append(row)

                    if len(cleaned_rows) < 2:
                        continue

                    # First row becomes column headers
                    headers = cleaned_rows[0]

                    # Clean headers
                    headers = [
                        str(header).strip()
                        if header is not None
                        else f"Column_{i + 1}"
                        for i, header in enumerate(headers)
                    ]

                    # Make duplicate column names unique
                    seen = {}

                    final_headers = []

                    for header in headers:

                        if header not in seen:
                            seen[header] = 0
                            final_headers.append(header)

                        else:
                            seen[header] += 1
                            final_headers.append(
                                f"{header}_{seen[header]}"
                            )

                    data_rows = cleaned_rows[1:]

                    df = pd.DataFrame(
                        data_rows,
                        columns=final_headers
                    )

                    # Remove completely empty columns
                    df = df.dropna(
                        axis=1,
                        how="all"
                    )

                    if not df.empty:

                        tables.append(
                            {
                                "page": page_number,
                                "table": table_number,
                                "dataframe": df
                            }
                        )

        return tables, None

    except Exception as e:

        return [], str(e)
    
# ============================================================
# LOAD EXCEL
# ============================================================

def load_excel(uploaded_file):
    try:
        df = pd.read_excel(uploaded_file)
        return df, None
    except Exception as e:
        return None, str(e)


# ============================================================
# LOAD PDF
# ============================================================

def load_pdf(uploaded_file):
    try:
        reader = PdfReader(uploaded_file)

        pages = []

        for page in reader.pages:
            text = page.extract_text()

            if text:
                pages.append(text)

        full_text = "\n\n".join(pages)

        if not full_text.strip():
            return None, "No readable text was found in this PDF."

        return full_text, None

    except Exception as e:
        return None, str(e)


# ============================================================
# CREATE DATASET CONTEXT
# ============================================================

def create_dataset_context(df):
    context_parts = []

    # --------------------------------------------------------
    # Basic information
    # --------------------------------------------------------

    context_parts.append(
        f"Number of rows: {len(df)}"
    )

    context_parts.append(
        f"Number of columns: {len(df.columns)}"
    )

    # --------------------------------------------------------
    # Column information
    # --------------------------------------------------------

    column_info = []

    for column in df.columns:

        dtype = str(df[column].dtype)

        non_null = int(
            df[column].notna().sum()
        )

        unique = int(
            df[column].nunique(dropna=True)
        )

        column_info.append(
            f"- {column}: "
            f"type={dtype}, "
            f"non_null={non_null}, "
            f"unique_values={unique}"
        )

    context_parts.append(
        "Columns:\n" +
        "\n".join(column_info)
    )

    # --------------------------------------------------------
    # Numeric summary
    # --------------------------------------------------------

    numeric_columns = df.select_dtypes(
        include="number"
    ).columns.tolist()

    if numeric_columns:

        try:

            summary = (
                df[numeric_columns]
                .describe()
                .round(2)
                .to_string()
            )

            context_parts.append(
                "Numeric summary:\n" +
                summary
            )

        except Exception:
            pass

    # --------------------------------------------------------
    # Categorical information
    # --------------------------------------------------------

    categorical_columns = df.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    if categorical_columns:

        categorical_info = []

        for column in categorical_columns[:20]:

            try:

                values = (
                    df[column]
                    .dropna()
                    .astype(str)
                )

                counts = (
                    values
                    .value_counts()
                    .head(10)
                )

                categorical_info.append(
                    f"{column}:\n{counts.to_string()}"
                )

            except Exception:
                pass

        if categorical_info:

            context_parts.append(
                "Categorical information:\n" +
                "\n\n".join(categorical_info)
            )

    # --------------------------------------------------------
    # Missing values
    # --------------------------------------------------------

    missing_info = []

    for column in df.columns:

        missing = int(
            df[column].isna().sum()
        )

        if missing > 0:

            missing_info.append(
                f"{column}: {missing} missing values"
            )

    if missing_info:

        context_parts.append(
            "Missing values:\n" +
            "\n".join(missing_info)
        )

    # --------------------------------------------------------
    # First rows
    # --------------------------------------------------------

    try:

        sample = (
            df.head(30)
            .to_string(index=False)
        )

        context_parts.append(
            "First 30 rows:\n" +
            sample
        )

    except Exception:
        pass

    # --------------------------------------------------------
    # Last rows
    # --------------------------------------------------------

    try:

        sample = (
            df.tail(10)
            .to_string(index=False)
        )

        context_parts.append(
            "Last 10 rows:\n" +
            sample
        )

    except Exception:
        pass

    return "\n\n".join(context_parts)


# ============================================================
# CREATE PDF CONTEXT
# ============================================================

def chunk_pdf_text(pdf_text, chunk_size=1200, overlap=200):
    """
    Split PDF text into overlapping chunks for retrieval.
    """

    if not pdf_text:
        return []

    pdf_text = pdf_text.strip()

    chunks = []

    start = 0

    while start < len(pdf_text):

        end = start + chunk_size

        chunk = pdf_text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start = end - overlap

    return chunks

# ============================================================
# EXACT DATA CALCULATIONS
# ============================================================

def calculate_data_question(df, question):
    """
    Perform exact data retrieval and calculations using the complete DataFrame.
    Returns an answer string if the question can be handled,
    otherwise returns None.
    """

    import re

    if df is None or df.empty:
        return None

    q = question.lower().strip()

    # --------------------------------------------------------
    # Identify numeric columns
    # --------------------------------------------------------

    numeric_columns = df.select_dtypes(
        include="number"
    ).columns.tolist()

    if not numeric_columns:
        return None

   # --------------------------------------------------------
    # Identify likely student/name column
    # --------------------------------------------------------

    name_column = None

    name_keywords = [
        "student",
        "student name",
        "student_name",
        "name",
        "person",
        "employee",
        "employee name",
        "employee_name",
        "customer",
        "customer name",
        "customer_name",
        "player",
        "candidate"
    ]

    # Check every column, not only object columns
    for column in df.columns:

        column_text = str(column).strip().lower()

        if any(
            keyword in column_text
            for keyword in name_keywords
        ):
            name_column = column
            break

    # --------------------------------------------------------
    # Fallback: find a text-like column
    # --------------------------------------------------------

    if name_column is None:

        for column in df.columns:

            if (
                df[column].dtype == "object"
                or str(df[column].dtype) == "category"
            ):

                # Avoid obvious non-name columns
                column_text = str(column).lower()

                if not any(
                    word in column_text
                    for word in [
                        "date",
                        "address",
                        "email",
                        "phone",
                        "description"
                    ]
                ):
                    name_column = column
                    break

    # --------------------------------------------------------
    # Identify numeric column
    # --------------------------------------------------------

    target_column = None

    for column in numeric_columns:
        if column.lower() in q:
            target_column = column
            break

    if target_column is None:
        # For a simple dataset, use the first numeric column.
        target_column = numeric_columns[0]

    # ========================================================
    # WHO SCORED / EXACT VALUE
    # ========================================================

    # Examples:
    # "who scored 45"
    # "which student scored 45 marks"
    # "students with 45 marks"

    number_match = re.search(
        r"\b(\d+(?:\.\d+)?)\b",
        q
    )

    # ========================================================
    # WHO IS ABOVE / BELOW A VALUE
    # ========================================================

    if number_match and any(
        phrase in q
        for phrase in [
            "who is below",
            "who are below",
            "which student is below",
            "which students are below",
            "who is above",
            "who are above",
            "which student is above",
            "which students are above"
        ]
    ):

        threshold = float(
            number_match.group(1)
        )

        # Determine whether the question asks above or below
        if any(
            word in q
            for word in ["below"]
        ):
            matching_rows = df[
                df[target_column] < threshold
            ]

            condition_text = f"below {threshold:g}"

        else:
            matching_rows = df[
                df[target_column] > threshold
            ]

            condition_text = f"above {threshold:g}"

        if name_column is not None:

            names = (
                matching_rows[name_column]
                .dropna()
                .astype(str)
                .tolist()
            )

            if names:
                return (
                    f"The matching records with "
                    f"{target_column} {condition_text} are "
                    f"{', '.join(names)}."
                )

            return (
                f"No records have "
                f"{target_column} {condition_text}."
            )

        return (
            f"There are {len(matching_rows)} records "
            f"with {target_column} {condition_text}."
        )

    if number_match and any(
        phrase in q
        for phrase in [
            "who scored",
            "which student",
            "which students",
            "students who scored",
            "student who scored",
            "scored"
        ]
    ):

        requested_value = float(number_match.group(1))

        matching_rows = df[
            df[target_column] == requested_value
        ]

        if not matching_rows.empty:

            if name_column is not None:

                names = (
                    matching_rows[name_column]
                    .dropna()
                    .astype(str)
                    .tolist()
                )

                if len(names) == 1:
                    return (
                        f"{names[0]} scored "
                        f"{requested_value:g} {target_column}."
                    )

                return (
                    f"The records who scored "
                    f"{requested_value:g} {target_column} are "
                    f"{', '.join(names)}."
                )

            return (
                f"{len(matching_rows)} student(s) scored "
                f"{requested_value:g} {target_column}."
            )

        return (
            f"No student scored "
            f"{requested_value:g} {target_column}."
        )

    # ========================================================
    # HOW MANY ABOVE / BELOW A VALUE
    # ========================================================

    if number_match and any(
        phrase in q
        for phrase in [
            "how many",
            "number of students",
            "count of students"
        ]
    ):

        requested_value = float(number_match.group(1))

        if any(
            word in q
            for word in ["above", "greater than", "more than"]
        ):

            count = (
                df[target_column] > requested_value
            ).sum()

            return (
                f"There are {count} matching records with "
                f"{target_column} above "
                f"{requested_value:g}."
            )

        if any(
            word in q
            for word in ["below", "less than", "under"]
        ):

            count = (
                df[target_column] < requested_value
            ).sum()

            return (
                f"There are {count} records with "
                f"{target_column} below "
                f"{requested_value:g}."
            )

    # ========================================================
    # TOTAL / SUM
    # ========================================================

    if any(
        word in q
        for word in ["sum", "total"]
    ):

        total = df[target_column].sum()

        return (
            f"The total {target_column} is "
            f"{total:,.2f}."
        )

    # ========================================================
    # AVERAGE / MEAN
    # ========================================================

    if any(
        word in q
        for word in ["average", "mean"]
    ):

        average = df[target_column].mean()

        return (
            f"The average {target_column} is "
            f"{average:,.2f}."
        )

    # ========================================================
    # COUNT / NUMBER OF RECORDS
    # ========================================================

    if any(
        phrase in q
        for phrase in [
            "how many records",
            "how many rows",
            "number of records",
            "number of rows",
            "how many students"
        ]
    ):

        return (
            f"There are {len(df):,} records in the dataset."
        )

    # ========================================================
    # HIGHEST / MAXIMUM
    # ========================================================

    if any(
        phrase in q
        for phrase in [
            "highest",
            "maximum",
            "max",
            "top",
            "highest marks",
            "highest score",
            "who scored highest",
            "who has highest",
            "top student",
            "top scorer"
        ]
    ):

        # Prefer marks/score-like column
        target_column = None

        for numeric_column in numeric_columns:
            if any(
                word in numeric_column.lower()
                for word in [
                    "mark",
                    "score",
                    "point",
                    "grade"
                ]
            ):
                target_column = numeric_column
                break

        if target_column is None:
            target_column = numeric_columns[0]

        maximum = df[target_column].max()

        matching_rows = df[
            df[target_column] == maximum
        ]

        if name_column is not None:

            names = (
                matching_rows[name_column]
                .dropna()
                .astype(str)
                .tolist()
            )

            if names:
                return (
                    f"{', '.join(names)} scored the highest "
                    f"{target_column} with "
                    f"{maximum:,.2f}."
                )

        return (
            f"The highest {target_column} is "
            f"{maximum:,.2f}."
        )

    # ========================================================
    # LOWEST / MINIMUM
    # ========================================================

    if any(
        word in q
        for word in [
            "lowest",
            "minimum",
            "min",
            "bottom",
            "lowest marks",
            "lowest score"
        ]
    ):

        minimum = df[target_column].min()

        matching_rows = df[
            df[target_column] == minimum
        ]

        if name_column is not None:

            names = (
                matching_rows[name_column]
                .dropna()
                .astype(str)
                .tolist()
            )

            if len(names) == 1:

                return (
                    f"{names[0]} scored the lowest "
                    f"{target_column} with "
                    f"{minimum:,.2f}."
                )

            return (
                f"The lowest {target_column} is "
                f"{minimum:,.2f}, scored by "
                f"{', '.join(names)}."
            )

        return (
            f"The lowest {target_column} is "
            f"{minimum:,.2f}."
        )
# --------------------------------------------------------
    # WHO SCORED HIGHEST / TOP STUDENT
    # --------------------------------------------------------

    if any(
        phrase in q
        for phrase in [
            "who scored highest",
            "who got highest",
            "highest marks",
            "top student",
            "top scorer"
        ]
    ):

        # Prefer a marks/score-like numeric column
        target_column = None

        for numeric_column in numeric_columns:
            if any(
                word in numeric_column.lower()
                for word in ["mark", "score", "point", "grade"]
            ):
                target_column = numeric_column
                break

        if target_column is None:
            target_column = numeric_columns[0]

        max_value = df[target_column].max()

        top_rows = df[df[target_column] == max_value]

        # Find a likely student/name column
        name_column = None

        for column in df.columns:
            if column.lower() in [
                "student",
                "name",
                "student_name",
                "candidate"
            ]:
                name_column = column
                break

        if name_column is not None:
            names = top_rows[name_column].astype(str).tolist()

            return (
                f"{', '.join(names)} scored the highest marks "
                f"with {max_value:,.2f}."
            )

        return (
            f"The highest {target_column} is "
            f"{max_value:,.2f}."
        )


    # --------------------------------------------------------
    # COUNT ABOVE / BELOW A VALUE
    # --------------------------------------------------------

    import re

    above_match = re.search(
        r"(?:above|over|greater than|more than)\s+(\d+(?:\.\d+)?)",
        q
    )

    if above_match:

        threshold = float(above_match.group(1))

        target_column = None

        for numeric_column in numeric_columns:
            if any(
                word in numeric_column.lower()
                for word in ["mark", "score", "point", "grade"]
            ):
                target_column = numeric_column
                break

        if target_column is None:
            target_column = numeric_columns[0]

        count = (
            df[target_column] > threshold
        ).sum()

        return (
            f"There are {count} records with "
            f"{target_column} above {threshold:g}."
        )    

    return None

# ============================================================
# RAG — EMBEDDINGS
# ============================================================

def create_embeddings(texts):
    """
    Convert text chunks into vector embeddings
    using the local Ollama embedding model.
    """

    embeddings = []

    for text in texts:
        response = ollama.embeddings(
            model="nomic-embed-text",
            prompt=text
        )

        embeddings.append(response["embedding"])

    return embeddings

# ============================================================
# RAG — SIMILARITY SEARCH
# ============================================================

import numpy as np


def similarity_search(
    question,
    chunks,
    chunk_embeddings,
    top_k=2
):
    """
    Find the most relevant PDF chunks for a question
    using cosine similarity.
    """

    if not chunks or not chunk_embeddings:
        return []

    # Create embedding for the user's question
    response = ollama.embeddings(
        model="nomic-embed-text",
        prompt=question
    )

    question_embedding = np.array(
        response["embedding"]
    )

    scores = []

    for i, chunk_embedding in enumerate(
        chunk_embeddings
    ):

        chunk_vector = np.array(
            chunk_embedding
        )

        # Cosine similarity
        similarity = np.dot(
            question_embedding,
            chunk_vector
        ) / (
            np.linalg.norm(question_embedding)
            * np.linalg.norm(chunk_vector)
        )

        scores.append((similarity, i))

    # Highest similarity first
    scores.sort(
        key=lambda x: x[0],
        reverse=True
    )

    # Return the top matching chunks
    results = []

    for score, index in scores[:top_k]:

        results.append({
            "chunk": chunks[index],
            "score": float(score)
        })

    return results

# ============================================================
# ASK AI ABOUT DATA
# ============================================================

def ask_ai(df, pdf_text, question):
    # ========================================================
    # EXACT DATA ANSWER FOR CSV / EXCEL
    # ========================================================

    exact_answer = calculate_data_question(
        df,
        question
    )

    if exact_answer is not None:
        return exact_answer
    # --------------------------------------------------------
    # EXACT DATA RETRIEVAL
    # --------------------------------------------------------

    if df is not None:
        exact_answer = calculate_data_question(df, question)

        if exact_answer is not None:
            return exact_answer

    # --------------------------------------------------------
    # Build context
    # --------------------------------------------------------

    if df is not None:

        context = create_dataset_context(df)

        source_type = "CSV or Excel dataset"
    # ----------------------------------------------------
        # EXACT DATA CALCULATION
        # ----------------------------------------------------

        exact_answer = calculate_data_question(
            df,
            question
        )

        if exact_answer is not None:
            return exact_answer

    elif pdf_text is not None:

        chunks = chunk_pdf_text(pdf_text)
        embeddings = create_embeddings(chunks)
        results=similarity_search(
            question,
            chunks,
            embeddings,
            top_k=2
        )    

        #st.write("Number of chunks:", len(chunks))
        #st.write("Embedding vector size:", len(embeddings[0]) if embeddings else 0)

        

       

    
        context= "\n\n".join(
            result["chunk"] if isinstance(result,dict)
        else result
            for result in results


        )

        source_type = "PDF document"

    else:

        return "No file has been loaded."

    # --------------------------------------------------------
    # Prompt
    # --------------------------------------------------------

    prompt = f"""
You are an expert data analyst.

The user uploaded a {source_type}.

Your job is to answer the user's actual question using ONLY
the information available in the uploaded file.

IMPORTANT RULES:

1. Answer the user's actual question directly.

2. Do NOT restrict yourself to only:
   average, highest, lowest, median, row count, column count,
   or summary questions.

3. You can answer questions about:
   - what the dataset/document is about
   - subject or topic
   - trends
   - patterns
   - comparisons
   - categories
   - unusual values
   - numeric calculations
   - summaries
   - relationships between columns
   - specific rows
   - specific values
   - missing values
   - products
   - customers
   - sales
   - dates
   - rankings
   - distributions
   - any other information that can reasonably be determined
     from the uploaded file

4. If the answer cannot be determined from the uploaded file,
   clearly explain that the information is not available.

5. NEVER invent facts that are not contained in the file.

6. Use actual column names when useful.

7. If a calculation is possible, calculate it.

8. Give a clear, human-readable answer.

9. If the user asks "what is this dataset about?",
   infer the subject from the column names, values, and document
   content.

10. If the user asks a question about a PDF, answer from the
    extracted PDF content.

11. NEVER return SQL, Python code, formulas, or database queries as the final answer.

12. When the user asks for a calculation, return the calculated result, not instructions for how to calculate it.

13. Always answer in plain, human-readable language.

14. If a numerical result can be calculated from the provided data, provide the actual numerical result.

15. Do not describe a query that could answer the question. Actually answer the question using the uploaded data.
CRITICAL OUTPUT RULE:

Return ONLY the final answer to the user's question.

NEVER return SQL.
NEVER return Python code.
NEVER return a database query.
NEVER describe how to calculate the answer.

If the user asks for a numerical value, calculate it from the uploaded data and return the number.

For example:
User: "What is the total sum in Austria?"
Bad answer: SELECT SUM(...) FROM ...
Good answer: The total sales in Austria are 1,234,567.89.

Your response must be a human-readable final answer, not a query.
UPLOADED FILE INFORMATION:

{context}

USER QUESTION:

{question}

ANSWER:
"""

    # --------------------------------------------------------
    # Ask Ollama
    # --------------------------------------------------------

    try:

        response = ollama.chat(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        return response["message"]["content"]

    except Exception as e:

        return (
            "I couldn't connect to the local Ollama model.\n\n"
            f"Error: {str(e)}"
        )


# ============================================================
# SESSION STATE
# ============================================================

if "df" not in st.session_state:
    st.session_state.df = None

if "pdf_text" not in st.session_state:
    st.session_state.pdf_text = None

if "file_name" not in st.session_state:
    st.session_state.file_name = None

if "file_type" not in st.session_state:
    st.session_state.file_type = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ Settings")

    st.write(
        f"**AI Model:** `{MODEL_NAME}`"
    )

    if check_ollama():

        st.success("Ollama is running")

    else:

        st.error(
            "Ollama is not running.\n\n"
            "Start Ollama and try again."
        )

    st.divider()

    st.subheader("📁 Upload your file")

    uploaded_file = st.file_uploader(
        "CSV, Excel or PDF",
        type=[
            "csv",
            "xlsx",
            "xls",
            "pdf"
        ]
    )


# ============================================================
# PROCESS UPLOADED FILE
# ============================================================

if uploaded_file is not None:


    file_name = uploaded_file.name

    extension = (
        os.path.splitext(file_name)[1]
        .lower()
    )

    # --------------------------------------------------------
    # CSV
    # --------------------------------------------------------

    if extension == ".csv":

        df, error = load_csv(uploaded_file)

        if error:

            st.error(
                "Could not read the CSV file:\n\n"
                + error
            )

        else:

            st.session_state.df = df
            st.session_state.pdf_text = None
            st.session_state.file_name = file_name
            st.session_state.file_type = "CSV"

            st.success(
                f"CSV loaded successfully: "
                f"{len(df):,} rows × "
                f"{len(df.columns):,} columns"
            )

    # --------------------------------------------------------
    # Excel
    # --------------------------------------------------------

    elif extension in [".xlsx", ".xls"]:

        df, error = load_excel(uploaded_file)

        if error:

            st.error(
                "Could not read the Excel file:\n\n"
                + error
            )

        else:

            st.session_state.df = df
            st.session_state.pdf_text = None
            st.session_state.file_name = file_name
            st.session_state.file_type = "Excel"

            st.success(
                f"Excel file loaded successfully: "
                f"{len(df):,} rows × "
                f"{len(df.columns):,} columns"
            )

    # --------------------------------------------------------
    # PDF
    # --------------------------------------------------------

    elif extension == ".pdf":

        pdf_text, error = load_pdf(
            uploaded_file
        )

        if error:

            st.error(
                "Could not read the PDF file:\n\n"
                + error
            )

        else:

            st.session_state.df = None
            st.session_state.pdf_text = pdf_text
            st.session_state.file_name = file_name
            st.session_state.file_type = "PDF"

            st.success(
                "PDF loaded successfully."
            )


# ============================================================
# NO FILE LOADED
# ============================================================

if (
    st.session_state.df is None
    and st.session_state.pdf_text is None
):

    st.info(
        "👈 Upload a CSV, Excel, or PDF file "
        "from the sidebar to begin."
    )

    


# ============================================================
# DATASET APPLICATION
# ============================================================

elif st.session_state.df is not None:

    df = st.session_state.df

    # ========================================================
    # DATASET OVERVIEW
    # ========================================================

    st.header("📋 Dataset Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Rows",
            f"{len(df):,}"
        )

    with col2:

        st.metric(
            "Columns",
            f"{len(df.columns):,}"
        )

    with col3:

        st.metric(
            "Missing Values",
            f"{int(df.isna().sum().sum()):,}"
        )

    with col4:

        st.metric(
            "Duplicate Rows",
            f"{int(df.duplicated().sum()):,}"
        )

    # ========================================================
    # DATA PREVIEW
    # ========================================================

    with st.expander(
        "👀 View uploaded data",
        expanded=False
    ):

        st.dataframe(
            df,
            use_container_width=True,
            height=400
        )

    # ========================================================
    # COLUMN INFORMATION
    # ========================================================

    with st.expander(
        "🔎 Column information",
        expanded=False
    ):

        column_data = pd.DataFrame(
            {
                "Column": df.columns,

                "Data Type": [
                    str(df[column].dtype)
                    for column in df.columns
                ],

                "Non-Null": [
                    int(
                        df[column]
                        .notna()
                        .sum()
                    )
                    for column in df.columns
                ],

                "Missing": [
                    int(
                        df[column]
                        .isna()
                        .sum()
                    )
                    for column in df.columns
                ],

                "Unique Values": [
                    int(
                        df[column]
                        .nunique(
                            dropna=True
                        )
                    )
                    for column in df.columns
                ]
            }
        )

        st.dataframe(
            column_data,
            use_container_width=True
        )

    # ========================================================
    # AUTOMATIC INSIGHTS
    # ========================================================

    st.header("💡 Automatic Insights")

    numeric_columns = (
        df
        .select_dtypes(include="number")
        .columns
        .tolist()
    )

    if numeric_columns:

        first_numeric = numeric_columns[0]

        average_value = df[first_numeric].mean()
        highest_value = df[first_numeric].max()
        lowest_value = df[first_numeric].min()
        median_value = df[first_numeric].median()

        col1, col2, col3, col4 = st.columns(4)

        with col1:

            st.metric(
                f"Average ({first_numeric})",
                f"{average_value:.2f}"
            )

        with col2:

            st.metric(
                f"Highest ({first_numeric})",
                f"{highest_value:.2f}"
            )

        with col3:

            st.metric(
                f"Lowest ({first_numeric})",
                f"{lowest_value:.2f}"
            )

        with col4:

            st.metric(
                f"Median ({first_numeric})",
                f"{median_value:.2f}"
            )

    else:

        st.info(
            "No numeric columns were detected."
        )

    # ========================================================
    # DATA VISUALIZATION
    # ========================================================

    st.header("📊 Data Visualization")

    st.write(
        "Create charts from the columns in your uploaded dataset."
    )

    all_columns = df.columns.tolist()

    numeric_columns = (
        df
        .select_dtypes(include="number")
        .columns
        .tolist()
    )

    categorical_columns = (
        df
        .select_dtypes(
            include=["object", "category"]
        )
        .columns
        .tolist()
    )

    if len(df.columns) > 0:

        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:

            chart_type = st.selectbox(
                "📈 Chart type",
                [
                    "Bar Chart",
                    "Line Chart",
                    "Area Chart",
                    "Histogram"
                ],
                key="chart_type"
            )

        with chart_col2:

            if chart_type == "Histogram":

                if numeric_columns:

                    y_column = st.selectbox(
                        "Numeric column",
                        numeric_columns,
                        key="hist_column"
                    )

                else:

                    y_column = None

                    st.warning(
                        "A histogram requires a numeric column."
                    )

            else:

                x_column = st.selectbox(
                    "X-axis",
                    all_columns,
                    key="x_column"
                )

                if numeric_columns:

                    y_column = st.selectbox(
                        "Y-axis",
                        numeric_columns,
                        key="y_column"
                    )

                else:

                    y_column = None

                    st.warning(
                        "No numeric columns are available "
                        "for the Y-axis."
                    )

        # ----------------------------------------------------
        # Generate chart
        # ----------------------------------------------------

        if chart_type == "Histogram":

            if y_column:

                st.subheader(
                    f"Distribution of {y_column}"
                )

                histogram_data = (
                    df[[y_column]]
                    .dropna()
                )

                st.bar_chart(
                    histogram_data
                    .value_counts()
                    .sort_index()
                )

        elif y_column:

            # ------------------------------------------------
            # Prepare chart data
            # ------------------------------------------------

            chart_data = df[
                [x_column, y_column]
            ].copy()

            chart_data = chart_data.dropna()

            # ------------------------------------------------
            # If X is categorical, aggregate values
            # ------------------------------------------------

            if (
                chart_data[x_column].dtype == "object"
                or str(
                    chart_data[x_column].dtype
                ) == "category"
            ):

                chart_data = (
                    chart_data
                    .groupby(
                        x_column,
                        as_index=True
                    )[y_column]
                    .sum()
                    .sort_values(
                        ascending=False
                    )
                    .head(30)
                )

                chart_data = chart_data.to_frame()

            else:

                chart_data = (
                    chart_data
                    .set_index(x_column)
                )

                chart_data = chart_data.head(100)

            # ------------------------------------------------
            # Display chart
            # ------------------------------------------------

            if chart_type == "Bar Chart":

                st.subheader(
                    f"{y_column} by {x_column}"
                )

                st.bar_chart(
                    chart_data
                )

            elif chart_type == "Line Chart":

                st.subheader(
                    f"{y_column} over {x_column}"
                )

                st.line_chart(
                    chart_data
                )

            elif chart_type == "Area Chart":

                st.subheader(
                    f"{y_column} over {x_column}"
                )

                st.area_chart(
                    chart_data
                )

        else:

            st.info(
                "Select a numeric Y-axis column "
                "to create a chart."
            )

    # ========================================================
    # QUICK VISUALIZATIONS
    # ========================================================

    if numeric_columns:

        st.subheader("📊 Quick Visualizations")

        quick_column = st.selectbox(
            "Choose a numeric column",
            numeric_columns,
            key="quick_numeric_column"
        )

        st.write(
            f"Distribution of **{quick_column}**"
        )

        quick_data = (
            df[[quick_column]]
            .dropna()
            .reset_index(drop=True)
        )

        st.line_chart(
            quick_data
        )

    # ========================================================
    # ASK YOUR DATA
    # ========================================================

    st.header("💬 Ask Your Data")

    st.write(
        "Ask anything you want to know about "
        "the uploaded file."
    )

    question = st.text_input(
        "Your question",
        placeholder=(
            "Example: Which product has the highest sales?"
        ),
        key="data_question"
    )

    ask_button = st.button(
        "🤖 Ask AI",
        type="primary"
    )

    if ask_button:

        if not question.strip():

            st.warning(
                "Please enter a question."
            )

        elif not check_ollama():

            st.error(
                "Ollama is not running. "
                "Please start Ollama and try again."
            )

        else:

            with st.spinner(
                "🤖 Analyzing your data..."
            ):

                answer = calculate_data_question(df, question)
                if answer is None:
                    answer=ask_ai(
                        df,
                        None,
                        question
                    )
                
                
                    
                

            st.session_state.chat_history.append(
                {
                    "question": question,
                    "answer": answer
                }
            )

    # ========================================================
    # DISPLAY ANSWERS
    # ========================================================

    if st.session_state.chat_history:

        st.subheader("🧠 AI Answers")

        for item in reversed(
            st.session_state.chat_history
        ):

            st.markdown(
                f"**You:** {item['question']}"
            )

            st.info(
                item["answer"]
            )

            st.divider()


# ============================================================
# PDF APPLICATION
# ============================================================

elif st.session_state.pdf_text is not None:

    pdf_text = st.session_state.pdf_text

    # ========================================================
    # PDF OVERVIEW
    # ========================================================

    st.header("📄 PDF Overview")

    word_count = len(
        pdf_text.split()
    )

    character_count = len(
        pdf_text
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Document",
            st.session_state.file_name
        )

    with col2:

        st.metric(
            "Words",
            f"{word_count:,}"
        )

    with col3:

        st.metric(
            "Characters",
            f"{character_count:,}"
        )

    # ========================================================
    # PDF CONTENT
    # ========================================================

    with st.expander(
        "👀 View extracted PDF text",
        expanded=False
    ):

        st.text_area(
            "PDF content",
            pdf_text,
            height=500
        )

    # ========================================================
    # PDF VISUALIZATION NOTICE
    # ========================================================

    st.header("📊 Data Visualization")

    st.info(
        "This PDF is currently being processed as document text. "
        "Charts are available automatically for CSV and Excel "
        "datasets. If your PDF contains tables, we can add "
        "PDF-table extraction and visualization next."
    )

    # ========================================================
    # ASK PDF
    # ========================================================

    st.header("💬 Ask Your PDF")

    st.write(
        "Ask anything about the contents of this PDF."
    )

    question = st.text_input(
        "Your question",
        placeholder=(
            "Example: What is this document about?"
        ),
        key="pdf_question"
    )

    ask_button = st.button(
        "🤖 Ask AI",
        type="primary",
        key="pdf_ask_button"
    )

    if ask_button:

        if not question.strip():

            st.warning(
                "Please enter a question."
            )

        elif not check_ollama():

            st.error(
                "Ollama is not running. "
                "Please start Ollama and try again."
            )

        else:

            with st.spinner(
                "🤖 Analyzing your PDF..."
            ):

                answer = ask_ai(
                    None,
                    pdf_text,
                    question
                )

            st.session_state.chat_history.append(
                {
                    "question": question,
                    "answer": answer
                }
            )

    # ========================================================
    # DISPLAY PDF ANSWERS
    # ========================================================

    if st.session_state.chat_history:

        st.subheader("🧠 AI Answers")

        for item in reversed(
            st.session_state.chat_history
        ):

            st.markdown(
                f"**You:** {item['question']}"
            )

            st.info(
                item["answer"]
            )

            st.divider()


# ============================================================
# FOOTER
# ============================================================

st.caption(
    "Local AI powered by Ollama • "
    "CSV • Excel • PDF • Data Visualization • "
    "No OpenAI API credits required"
)
