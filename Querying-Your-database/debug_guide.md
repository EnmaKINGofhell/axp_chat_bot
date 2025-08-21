# Debugging Guide for SQL Query Issue

## Problem Summary
You're experiencing an issue where the SQL query `SELECT invoice_number FROM invoice_header;` works correctly in MySQL Workbench but returns "No data to display" when executed through the application.

## Diagnosis Steps

### 1. Add Debug Print Statements

Add the following debug print statements to your `main.py` file to understand what's happening with the SQL response:

```python
# Around line 349, after executing the SQL query
sql_response = st.session_state.db.run(sql_query)
print(f"DEBUG: SQL Query: {sql_query}")
print(f"DEBUG: Raw SQL Response: {sql_response}")
print(f"DEBUG: SQL Response Type: {type(sql_response)}")
if sql_response and isinstance(sql_response, list) and len(sql_response) > 0:
    print(f"DEBUG: First Item Type: {type(sql_response[0])}")
    print(f"DEBUG: First Item: {sql_response[0]}")
```

Also add debug statements in the `get_human_response_groq` function (around line 112):

```python
def get_human_response_groq(sql_query, schema, sql_response, user_query=None, invoice_id=None, is_invoice_specific=False):
    """
    Summarizes SQL results in plain English using the LLM.
    """
    print(f"DEBUG: get_human_response_groq - SQL Response: {sql_response}")
    print(f"DEBUG: get_human_response_groq - SQL Response Type: {type(sql_response)}")
    if sql_response and isinstance(sql_response, list) and len(sql_response) > 0:
        print(f"DEBUG: get_human_response_groq - First Item Type: {type(sql_response[0])}")
        print(f"DEBUG: get_human_response_groq - First Item: {sql_response[0]}")
    
    # Rest of the function remains the same...
```

### 2. Modify the DataFrame Conversion Logic

Based on the debug output, you may need to modify how the SQL response is converted to a DataFrame. Here's a more robust implementation to replace the existing code in the `get_human_response_groq` function:

```python
df = None
if sql_response:
    print(f"DEBUG: SQL response is not empty")
    if isinstance(sql_response, list):
        print(f"DEBUG: SQL response is a list with {len(sql_response)} items")
        if len(sql_response) > 0:
            if isinstance(sql_response[0], dict):
                print(f"DEBUG: First item is a dict, creating DataFrame from dicts")
                df = pd.DataFrame(sql_response)
            elif isinstance(sql_response[0], (list, tuple)):
                print(f"DEBUG: First item is a list/tuple, creating DataFrame with columns")
                # Try to get columns from schema if available
                columns = None
                if schema:
                    # Try to extract columns from schema string
                    lines = schema.splitlines()
                    for line in lines:
                        if line.strip().startswith('Table: invoice_header'):
                            current_table_cols = []
                            continue
                        if line.strip().startswith('  '):
                            col_name = line.strip().split()[0]
                            if current_table_cols is None:
                                current_table_cols = []
                            current_table_cols.append(col_name)
                    
                    if current_table_cols:
                        columns = current_table_cols[:len(sql_response[0])]
                
                if not columns:
                    print(f"DEBUG: No columns found, using generic column names")
                    columns = [f"col{i+1}" for i in range(len(sql_response[0]))]
                
                print(f"DEBUG: Creating DataFrame with columns: {columns}")
                df = pd.DataFrame(sql_response, columns=columns)
            else:
                print(f"DEBUG: First item is neither dict nor list/tuple: {type(sql_response[0])}")
                # Try to convert to DataFrame anyway
                try:
                    df = pd.DataFrame(sql_response)
                    print(f"DEBUG: Created DataFrame anyway: {df.shape}")
                except Exception as e:
                    print(f"DEBUG: Failed to create DataFrame: {e}")
        else:
            print(f"DEBUG: SQL response is an empty list")
    else:
        print(f"DEBUG: SQL response is not a list: {type(sql_response)}")
        # Try to convert to DataFrame anyway
        try:
            df = pd.DataFrame([sql_response])
            print(f"DEBUG: Created DataFrame from non-list: {df.shape}")
        except Exception as e:
            print(f"DEBUG: Failed to create DataFrame from non-list: {e}")

# If no data, return a friendly message
if df is None or df.empty:
    print(f"DEBUG: DataFrame is None or empty")
    if invoice_id and is_invoice_specific:
        return f"No data found for invoice number {invoice_id}."
    return "No data to display."
```

### 3. Alternative Direct SQL Execution

If the above changes don't resolve the issue, you can try implementing a direct SQL execution approach using SQLAlchemy:

```python
# Add this to your imports at the top of the file
from sqlalchemy import create_engine
import pandas as pd

# Add this function to your code
def execute_direct_sql(sql_query):
    """Execute SQL query directly using SQLAlchemy instead of LangChain's SQLDatabase"""
    try:
        # Get database credentials
        creds = get_default_db_creds()
        # Create SQLAlchemy engine
        engine = create_engine(f"mysql+pymysql://{creds['user']}:{creds['password']}@{creds['host']}/{creds['database']}")
        # Execute query and return results as DataFrame
        df = pd.read_sql(sql_query, engine)
        print(f"DEBUG: Direct SQL execution result: {df.shape}")
        return df
    except Exception as e:
        print(f"DEBUG: Direct SQL execution error: {e}")
        return None
```

Then modify the code around line 349 to use this function:

```python
try:
    st.session_state['last_sql_query'] = sql_query
    # Try direct SQL execution first
    print(f"DEBUG: Trying direct SQL execution")
    df_direct = execute_direct_sql(sql_query)
    if df_direct is not None and not df_direct.empty:
        print(f"DEBUG: Direct SQL execution successful")
        # Convert DataFrame to format expected by get_human_response_groq
        sql_response = df_direct.to_dict('records')
    else:
        print(f"DEBUG: Direct SQL execution failed or returned empty result, falling back to LangChain")
        # Fall back to LangChain's SQLDatabase
        sql_response = st.session_state.db.run(sql_query)
    
    # Use the refactored get_human_response_groq to summarize
    human_readable_response = get_human_response_groq(
        sql_query, 
        schema, 
        sql_response, 
        user_query=user_query, 
        invoice_id=st.session_state.invoice_id, 
        is_invoice_specific=is_invoice_specific
    )
    st.markdown(human_readable_response)
    
    st.session_state.chat_history.append(
        AIMessage(
            content=human_readable_response
        )
    )
except Exception as e:
    st.error(f"Error executing SQL query: {e}")
    st.code(sql_query, language="sql")
```

## Testing the Changes

After making these changes:

1. Run the application
2. Log in with the password "admin123"
3. Type "show me all the invoices" in the chat input
4. Check the console/terminal for the debug output
5. See if the application now displays the invoice data correctly

## Understanding the Issue

The most likely causes of the "No data to display" issue are:

1. The SQL response format from LangChain's SQLDatabase is not what the application expects
2. The DataFrame conversion logic is failing to properly process the SQL response
3. There might be an issue with how the application is connecting to the database

The debug print statements will help identify which of these is the root cause.