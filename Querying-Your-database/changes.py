import pandas as pd
from sqlalchemy import create_engine, text

# Database connection details
DB_USER = 'root'
DB_PASSWORD = 'root'
DB_HOST = 'localhost'
DB_NAME = 'axp_demo'

# --- 1. Read and Prepare CSV Data ---

# Supplier Master table
df_supplier_master = pd.read_csv('SuppMaster.csv')
supplier_master_cols = {
    'vd_domain': 'supplier_domain',
    'vd_addr': 'supplier_address_code',
    'vd_dataset': 'supplier_dataset',
    'vd_sort': 'supplier_sort_name',
    'ad_line1': 'address_line1',
    'ad_city': 'city',
    'ad_state': 'state',
    'ad_zip': 'zip_code',
    'ad_ctry': 'country',
    'vd_cr_terms': 'credit_terms',
    'ad_taxable': 'is_taxable',
    'ad_tax_usage': 'tax_usage_code',
    'ad_phone': 'phone_number'
}
df_supplier_master.rename(columns=supplier_master_cols, inplace=True)

# Add missing columns from the full schema and fill with None/default values
missing_cols = [
    'address_line2', 'address_line3', 'attention_to_2', 'attention_to',
    'currency', 'supplier_type', 'ap_account', 'ap_subaccount',
    'ap_cost_center', 'purchase_account', 'purchase_subaccount',
    'purchase_cost_center', 'custom_field_1', 'custom_field_2',
    'custom_field_3', 'custom_field_4', 'is_active'
]
for col in missing_cols:
    if col not in df_supplier_master.columns:
        df_supplier_master[col] = None
if 'is_active' in df_supplier_master.columns:
    df_supplier_master['is_active'] = df_supplier_master['is_active'].fillna(1)
    
# ADDED: Remove duplicate rows based on the original composite primary key
df_supplier_master.drop_duplicates(subset=['supplier_domain', 'supplier_address_code'], keep='first', inplace=True)


# Invoice Header table
df_invoice_header = pd.read_csv('InvHdr.csv')
invoice_header_cols = {
    'id': 'header_id', 'billtoid': 'bill_to_id', 'billtodesc': 'bill_to_description',
    'invno': 'invoice_number', 'invoicetypeid': 'invoice_type_id',
    'invdate': 'invoice_date', 'suppid': 'supplier_id',
    'suppdomain': 'supplier_domain', 'totinvamt': 'total_invoice_amount',
    'invstatus': 'invoice_status', 'datasetid': 'dataset_id',
    'datasetcode': 'dataset_code', 'erpbatchid': 'erp_batch_id',
    'voucherduedate': 'voucher_due_date', 'createdby': 'created_by',
    'createdon': 'created_on', 'modifiedby': 'modified_by',
    'modifiedon': 'modified_on', 'reviewactivationdate': 'review_activation_date',
    'currency': 'currency', 'version': 'version'
}
df_invoice_header.rename(columns=invoice_header_cols, inplace=True)
df_invoice_header['invoice_date'] = pd.to_datetime(df_invoice_header['invoice_date'], errors='coerce')
df_invoice_header['created_on'] = pd.to_datetime(df_invoice_header['created_on'], errors='coerce')
df_invoice_header['modified_on'] = pd.to_datetime(df_invoice_header['modified_on'], errors='coerce')
df_invoice_header['review_activation_date'] = pd.to_datetime(df_invoice_header['review_activation_date'], errors='coerce')


# Invoice Detail table
df_invoice_detail = pd.read_csv('InvDtl.csv')
invoice_detail_cols = {
    'Id': 'detail_id', 'Invoiceid': 'invoice_id', 'ItemSrl': 'product_id',
    'ItemLine': 'invoice_line_number', 'ItemDesc': 'product_description',
    'InvQty': 'invoice_quantity', 'ItemUnitPrice': 'item_unit_price',
    'ItemExtPrice': 'item_extended_price', 'Account': 'gl_account_number',
    'Project': 'project_code', 'Entity': 'entity_code'
}
df_invoice_detail.rename(columns=invoice_detail_cols, inplace=True)
df_invoice_detail.drop_duplicates(subset=['detail_id'], keep='first', inplace=True)


# --- 2. Create Database and Tables in MySQL ---
def create_and_populate_database():
    try:
        engine_url = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}"
        engine = create_engine(engine_url)

        with engine.connect() as connection:
            connection.execute(text(f"DROP DATABASE IF EXISTS {DB_NAME};"))
            connection.execute(text(f"CREATE DATABASE {DB_NAME};"))
            connection.execute(text(f"USE {DB_NAME};"))
            connection.commit()
            print(f"Database '{DB_NAME}' created and selected successfully.")

            create_table_sqls = [
                # FINAL FIX: Reverting to the composite primary key
                """
                CREATE TABLE `supplier_master` (
                  `supplier_domain` varchar(10) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL,
                  `supplier_address_code` varchar(10) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL,
                  `supplier_dataset` TEXT CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci DEFAULT NULL,
                  `supplier_sort_name` TEXT CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci DEFAULT NULL,
                  `address_line1` TEXT CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci DEFAULT NULL,
                  `address_line2` TEXT CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci DEFAULT NULL,
                  `address_line3` TEXT CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci DEFAULT NULL,
                  `city` TEXT CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci DEFAULT NULL,
                  `state` TEXT CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci DEFAULT NULL,
                  `zip_code` TEXT CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci DEFAULT NULL,
                  `country` varchar(10) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci DEFAULT NULL,
                  `credit_terms` TEXT CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci DEFAULT NULL,
                  `is_taxable` tinyint(1) DEFAULT NULL,
                  `tax_usage_code` TEXT CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci DEFAULT NULL,
                  `phone_number` TEXT CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci DEFAULT NULL,
                  `attention_to_2` TEXT CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci DEFAULT NULL,
                  `attention_to` TEXT CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci DEFAULT NULL,
                  `currency` varchar(5) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci DEFAULT NULL,
                  `supplier_type` TEXT CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci DEFAULT NULL,
                  `ap_account` TEXT CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci DEFAULT NULL,
                  `ap_subaccount` TEXT CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci DEFAULT NULL,
                  `ap_cost_center` TEXT CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci DEFAULT NULL,
                  `purchase_account` TEXT CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci DEFAULT NULL,
                  `purchase_subaccount` TEXT CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci DEFAULT NULL,
                  `purchase_cost_center` TEXT CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci DEFAULT NULL,
                  `custom_field_1` TEXT CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci DEFAULT NULL,
                  `custom_field_2` TEXT CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci DEFAULT NULL,
                  `custom_field_3` TEXT CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci DEFAULT NULL,
                  `custom_field_4` TEXT CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci DEFAULT NULL,
                  `is_active` tinyint(1) DEFAULT '1',
                  PRIMARY KEY (`supplier_domain`,`supplier_address_code`)
                );
                """,
                """
                CREATE TABLE invoice_header (
                    header_id VARCHAR(255) PRIMARY KEY,
                    bill_to_id VARCHAR(255),
                    bill_to_description VARCHAR(255),
                    invoice_number VARCHAR(255),
                    invoice_type_id VARCHAR(255),
                    invoice_date DATE,
                    supplier_id INT,
                    supplier_domain VARCHAR(255),
                    total_invoice_amount DECIMAL(10, 2),
                    invoice_status INT,
                    dataset_id VARCHAR(255),
                    dataset_code VARCHAR(255),
                    erp_batch_id VARCHAR(255),
                    voucher_due_date DATE,
                    created_by VARCHAR(255),
                    created_on DATETIME,
                    modified_by VARCHAR(255),
                    modified_on DATETIME,
                    review_activation_date DATETIME,
                    currency VARCHAR(255),
                    version INT,
                    FOREIGN KEY (supplier_domain) REFERENCES supplier_master(supplier_domain)
                );
                """,
                """
                CREATE TABLE invoice_detail (
                    detail_id VARCHAR(255) PRIMARY KEY,
                    invoice_id VARCHAR(255),
                    product_id VARCHAR(255),
                    invoice_line_number INT,
                    product_description TEXT,
                    invoice_quantity INT,
                    item_unit_price DECIMAL(10, 2),
                    item_extended_price DECIMAL(10, 2),
                    gl_account_number INT,
                    project_code VARCHAR(255),
                    entity_code VARCHAR(255),
                    FOREIGN KEY (invoice_id) REFERENCES invoice_header(header_id)
                );
                """,
                """
                CREATE TABLE invoices_extracted (
                    invoice_id VARCHAR(255) PRIMARY KEY,
                    invoice_number_extracted VARCHAR(255),
                    invoice_date_extracted DATE,
                    supplier_name_extracted VARCHAR(255),
                    customer_name_extracted VARCHAR(255),
                    total_amount_extracted DECIMAL(10, 2),
                    sales_tax_extracted DECIMAL(10, 2),
                    raw_text LONGTEXT,
                    customer_po_no_extracted VARCHAR(255),
                    sales_order_no_extracted VARCHAR(255),
                    delivery_no_extracted VARCHAR(255),
                    currency_extracted VARCHAR(255),
                    overall_description_extracted VARCHAR(255),
                    extracted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                """,
                """
                CREATE TABLE invoice_line_items_extracted (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    invoice_id VARCHAR(255),
                    product_id VARCHAR(255),
                    description TEXT,
                    quantity DECIMAL(10, 2),
                    unit_price DECIMAL(10, 2),
                    extended_price DECIMAL(10, 2),
                    customer_material_number VARCHAR(255),
                    customer_material_description VARCHAR(255),
                    line_sales_order_no VARCHAR(255),
                    line_delivery_no VARCHAR(255),
                    batch_no VARCHAR(255),
                    line_customer_po_no VARCHAR(255),
                    FOREIGN KEY (invoice_id) REFERENCES invoices_extracted(invoice_id)
                );
                """
            ]

            for sql in create_table_sqls:
                connection.execute(text(sql))
            connection.commit()
            print("Tables created successfully.")

        engine_with_db = create_engine(f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}")

        print("\nLoading data from DataFrames into tables...")
        df_supplier_master.to_sql(name='supplier_master', con=engine_with_db, if_exists='append', index=False)
        print("Data loaded into 'supplier_master'.")
        df_invoice_header.to_sql(name='invoice_header', con=engine_with_db, if_exists='append', index=False)
        print("Data loaded into 'invoice_header'.")
        df_invoice_detail.to_sql(name='invoice_detail', con=engine_with_db, if_exists='append', index=False)
        print("Data loaded into 'invoice_detail'.")
        
        print("\nAll tables have been populated successfully!")
        
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        print("Please ensure your MySQL server is running and the connection details are correct.")

if __name__ == '__main__':
    create_and_populate_database()