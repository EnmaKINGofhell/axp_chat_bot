AXP Assistant
The AXP Assistant is a powerful, AI-powered conversational tool designed to streamline invoice management and financial analysis. It leverages the Groq API and LangChain to allow users to interact with a MySQL database using natural language, eliminating the need for complex SQL queries.

This application simplifies common tasks such as:

Invoice Data Extraction: Automatically processes and extracts key information from uploaded PDF invoices.

Conversational Data Query: Answer questions about invoices, suppliers, and financial data in plain English.

Database Interaction: Directly translates natural language queries into optimized SQL, executes them, and provides a clear, human-friendly summary of the results.

Features
Intelligent SQL Generation: The system uses an Advanced SQL Optimizer to analyze user queries and generate efficient, accurate SQL commands.

Dynamic Chat History: Maintains conversational context to answer follow-up questions without requiring users to repeat information.

PDF Invoice Processing: Extracts structured data (invoice numbers, dates, line items, etc.) from uploaded PDF documents using pdfplumber and a Groq-powered LLM.

Comprehensive Data Review: A user-friendly interface allows for reviewing and editing extracted invoice data before it is committed to the database.

Database Integration: Seamlessly connects with a MySQL database to store and retrieve invoice information.

Installation
To get the AXP Assistant up and running, follow these steps.

Prerequisites
Python 3.8+

MySQL Database (Ensure it's running and you have user credentials)

Groq API Key

Step-by-Step Guide
Clone the Repository

Bash

git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME
Install Python Dependencies
We recommend using a virtual environment.

Bash

pip install -r requirements.txt
Set Up Environment Variables
Create a .env file in the root directory of your project and add your Groq API key.

Code snippet

GROQ_API_KEY="YOUR_GROQ_API_KEY"
Configure MySQL Connection
Open app.py and ensure the database connection string is configured correctly for your MySQL instance.

Python

# In app.py
st.session_state.db = SQLDatabase.from_uri("mysql+pymysql://root:root@localhost/axp_demo_2")
(Modify root:root@localhost/axp_demo_2 with your own credentials and database name.)

Run the Application
Start the Streamlit application from your terminal.

Bash

streamlit run app.py
The application will open in your web browser.

Usage
1. Invoice Upload
Navigate to the Invoice Upload section in the sidebar.

Upload a PDF invoice. The application will automatically extract key details.

Review and edit the extracted information (invoice number, total amount, line items, etc.) in the form.

Click Confirm & Upload to Database to save the data.

2. Conversational Chat
Go to the Chat section in the sidebar.

Log in with the password admin123.

Ask the assistant questions about your financial data, such as:

"What is the total invoice amount for all invoices?"

"Show me the details for invoice number INV-96351659."

"List all suppliers and their total invoice amounts."

"Show me the SQL query for the last question."

Technologies Used
Python

Streamlit: For building the interactive web application.

Groq API: Provides the powerful language model for natural language processing and SQL generation.

LangChain: A framework for developing applications powered by language models.

SQLAlchemy: The Python SQL toolkit for database interaction.

pdfplumber: For reading and extracting data from PDFs.

Pandas: For data manipulation and display.

Contributing
We welcome contributions! Please feel free to fork the repository, create a new branch, and submit a pull request with your changes.

