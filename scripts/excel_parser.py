import pandas as pd
import json
import re

def parse_excel_data(file_path, sheet_name="Combined"):
    """
    Parse Excel file and return structured data for Sankey chart generation
    """
    try:
        # Read Excel file
        df = pd.read_excel(file_path, sheet_name=sheet_name,
                          usecols=["ECO #", "Affected Item", "Customers", "PMs"])
        
        # Rename columns for consistency
        df = df.rename(columns={
            "ECO #": "Change_Order",
            "Affected Item": "Affected_PN",
            "Customers": "Customers",
            "PMs": "PMs"
        })
        
        # Clean and parse data
        df["Customers"] = df["Customers"].fillna("")
        df["PMs"] = df["PMs"].fillna("")
        
        # Parse comma-separated values
        df["CustList"] = df["Customers"].apply(
            lambda s: [c.strip() for c in re.split(r",\s*", s) if c.strip()]
        )
        df["PMList"] = df["PMs"].apply(
            lambda s: [p.strip() for p in re.split(r",\s*", s) 
                      if p.strip() and p.strip() != "#N/A"]
        )
        
        return df.to_dict('records')
        
    except Exception as e:
        print(f"Error parsing Excel file: {str(e)}")
        return []

def generate_sankey_data(df_records, eco_number):
    """
    Generate Sankey chart data structure from parsed Excel data
    """
    # Filter data for specific ECO
    filtered_data = [record for record in df_records 
                    if record.get('Change_Order') == eco_number]
    
    if not filtered_data:
        return None
    
    # Build node levels
    eco_nodes = [eco_number]
    part_nodes = list(set([record['Affected_PN'] for record in filtered_data 
                          if record.get('Affected_PN')]))
    customer_nodes = list(set([customer for record in filtered_data 
                              for customer in record.get('CustList', [])]))
    pm_nodes = list(set([pm for record in filtered_data 
                        for pm in record.get('PMList', [])]))
    
    # Create node list with colors
    nodes = []
    colors = {
        'ECO': '#1f77b4',
        'Part': '#ff7f0e', 
        'Customer': '#2ca02c',
        'PM': '#d62728'
    }
    
    # Add nodes with levels and colors
    for node in eco_nodes:
        nodes.append({'id': node, 'label': node, 'level': 0, 'color': colors['ECO']})
    for node in part_nodes:
        nodes.append({'id': node, 'label': node, 'level': 1, 'color': colors['Part']})
    for node in customer_nodes:
        nodes.append({'id': node, 'label': node, 'level': 2, 'color': colors['Customer']})
    for node in pm_nodes:
        nodes.append({'id': node, 'label': node, 'level': 3, 'color': colors['PM']})
    
    # Generate links
    links = []
    
    # ECO to Parts
    part_counts = {}
    for record in filtered_data:
        part = record.get('Affected_PN')
        if part:
            part_counts[part] = part_counts.get(part, 0) + 1
    
    for part, count in part_counts.items():
        links.append({'source': eco_number, 'target': part, 'value': count})
    
    # Parts to Customers
    for record in filtered_data:
        part = record.get('Affected_PN')
        customers = record.get('CustList', [])
        for customer in customers:
            links.append({'source': part, 'target': customer, 'value': 1})
    
    # Customers to PMs
    for record in filtered_data:
        customers = record.get('CustList', [])
        pms = record.get('PMList', [])
        for customer in customers:
            for pm in pms:
                links.append({'source': customer, 'target': pm, 'value': 1})
    
    return {
        'nodes': nodes,
        'links': links
    }

# Example usage
if __name__ == "__main__":
+    # Prompt user for file path and ECO number
+    file_path_input = input("Enter the path to your Excel file (e.g., my_data.xlsx): ")
+    eco_number_input = input("Enter the ECO number to analyze (e.g., C01798): ")
+
+    # Use user inputs
+    data = parse_excel_data(file_path_input)
+    sankey_data = generate_sankey_data(data, eco_number_input)
+
+    if sankey_data:
+        print("\nGenerated Sankey Data:")
+        print(json.dumps(sankey_data, indent=2))
+    else:
+        print(f"\nNo data found or error processing for ECO {eco_number_input}")
