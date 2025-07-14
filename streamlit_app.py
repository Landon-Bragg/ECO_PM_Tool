import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import re
from typing import Optional, Dict, List, Tuple, Set
from collections import defaultdict

# ───────────────── CONFIG ───────────────────────────────
REQUIRED_COLUMNS = ["ECO #", "Affected Item", "Customers", "PMs"]
OPTIONAL_COLUMNS = ["Days Open", "Ancestors"]
ALL_EXPECTED_COLUMNS = REQUIRED_COLUMNS + OPTIONAL_COLUMNS

# Strict hierarchical structure definition
HIERARCHY_LEVELS = {
    1: "ECO",           # Layer 1: ECO Number
    2: "Items",         # Layer 2: Affected Items + Ancestors
    3: "Customers",     # Layer 3: Customers (only if they exist)
    4: "PMs"           # Layer 4: Project Managers (only if they exist)
}

# Optional: ensure consistent Plotly theme
pio.templates.default = "plotly_white"

# ───────────────────────────────────────────────────────

def get_sheet_names(uploaded_file) -> List[str]:
    """Get all sheet names from the Excel file."""
    try:
        excel_file = pd.ExcelFile(uploaded_file)
        return excel_file.sheet_names
    except Exception as e:
        st.error(f"Error reading Excel file: {str(e)}")
        return []

@st.cache_data
def load_excel_data(uploaded_file, sheet_name: str) -> Optional[pd.DataFrame]:
    """
    Load and validate Excel data from uploaded file.
    
    Args:
        uploaded_file: Streamlit uploaded file object
        sheet_name: Name of the sheet to read
        
    Returns:
        DataFrame with all data or None if validation fails
    """
    try:
        # Read Excel file - read ALL data without any column restrictions
        df = pd.read_excel(uploaded_file, sheet_name=sheet_name)
        
        st.success(f"✅ Initial data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        
        # Display available columns
        available_columns = list(df.columns)
        
        # Check if required columns exist
        missing_required = [col for col in REQUIRED_COLUMNS if col not in df.columns]
        if missing_required:
            st.error(f"❌ Missing required columns: {', '.join(missing_required)}")
            st.error("Please ensure your Excel file contains the required columns:")
            for col in REQUIRED_COLUMNS:
                status = "✅" if col in df.columns else "❌"
                st.write(f"  {status} {col}")
            return None
        
        # Check for optional columns
        available_optional = [col for col in OPTIONAL_COLUMNS if col in df.columns]
        missing_optional = [col for col in OPTIONAL_COLUMNS if col not in df.columns]
        
        # Analyze ECO # column before any filtering
        original_count = len(df)
        eco_col = df["ECO #"]
        
        # Count different types of values in ECO # column
        null_count = eco_col.isnull().sum()
        empty_string_count = (eco_col.astype(str).str.strip() == '').sum()
        nan_string_count = (eco_col.astype(str).str.upper() == 'NAN').sum()
        valid_count = original_count - null_count - empty_string_count - nan_string_count
        

        df_clean = df.copy()
        
        # Remove null ECO #
        if null_count > 0:
            df_clean = df_clean.dropna(subset=['ECO #'])
        
        # Remove empty string ECO #
        mask_not_empty = df_clean['ECO #'].astype(str).str.strip() != ''
        empty_removed = len(df_clean) - mask_not_empty.sum()
        df_clean = df_clean[mask_not_empty]
        
        # Remove 'nan' string ECO #
        mask_not_nan = df_clean['ECO #'].astype(str).str.upper() != 'NAN'
        nan_removed = len(df_clean) - mask_not_nan.sum()
        df_clean = df_clean[mask_not_nan]

        
        # Clean and standardize ECO # column
        df_clean['ECO #'] = df_clean['ECO #'].astype(str).str.strip()
        
        # Add processed columns for internal use while keeping original columns
        df_clean['Change_Order'] = df_clean['ECO #']
        df_clean['Affected_PN'] = df_clean['Affected Item']
        
        final_count = len(df_clean)
        removed_count = original_count - final_count
    
        
        # Show sample of unique ECO numbers
        unique_ecos = sorted(df_clean['Change_Order'].unique())
        
        # Display first 10 ECO numbers as examples
        sample_ecos = unique_ecos[:10]

        
        return df_clean
        
    except Exception as e:
        st.error(f"❌ Error reading Excel file: {str(e)}")
        st.exception(e)
        return None

def parse_delimited_field(field_value: str) -> List[str]:
    """
    Parse comma-delimited field and return list of cleaned values.
    Handles missing values, empty strings, and #N/A entries.
    
    Args:
        field_value: String containing comma-separated values
        
    Returns:
        List of cleaned, non-empty strings
    """
    if pd.isna(field_value) or field_value == '' or str(field_value).upper() == 'NAN':
        return []
    
    # Split by comma and clean each item
    items = [item.strip() for item in re.split(r',\s*', str(field_value))]
    # Filter out empty strings, #N/A values, and NaN
    items = [item for item in items if item and item != '#N/A' and item.upper() != 'NAN' and item.strip() != '']
    
    return items

def get_available_ecos(df: pd.DataFrame) -> List[str]:
    """
    Get sorted list of unique ECO numbers from the dataset.
    
    Args:
        df: DataFrame containing ECO data
        
    Returns:
        Sorted list of unique ECO numbers
    """
    unique_ecos = df['Change_Order'].unique()
    # Remove any NaN values and sort
    unique_ecos = [eco for eco in unique_ecos if pd.notna(eco) and str(eco).upper() != 'NAN']
    return sorted(unique_ecos)

def extract_all_items_and_ancestors(filtered_df: pd.DataFrame) -> Tuple[Set[str], Dict[str, Dict]]:
    """
    Extract all affected items and ancestors, combining them into a single items set.
    Also track which items have customers and PMs for flow termination logic.
    
    Args:
        filtered_df: DataFrame filtered for specific ECO
        
    Returns:
        Tuple of (all_items_set, item_relationships_dict)
    """
    all_items = set()
    item_relationships = {} # Stores {item: {'customers': set(), 'pms': set()}}
    
    for _, row in filtered_df.iterrows():
        # Get affected item
        affected_item = str(row['Affected_PN']).strip() if pd.notna(row['Affected_PN']) else ''
        
        # Get ancestors (if column exists)
        ancestors = []
        if 'Ancestors' in row and pd.notna(row['Ancestors']):
            ancestors = parse_delimited_field(row['Ancestors'])
        
        # Get customers and PMs for this specific row
        row_customers = parse_delimited_field(row['Customers'])
        row_pms = parse_delimited_field(row['PMs'])
        
        # Process affected item
        if affected_item:
            all_items.add(affected_item)
            if affected_item not in item_relationships:
                item_relationships[affected_item] = {'customers': set(), 'pms': set()}
            item_relationships[affected_item]['customers'].update(row_customers)
            item_relationships[affected_item]['pms'].update(row_pms)
        
        # Process ancestors
        for ancestor in ancestors:
            if ancestor:
                all_items.add(ancestor)
                if ancestor not in item_relationships:
                    item_relationships[ancestor] = {'customers': set(), 'pms': set()}
                item_relationships[ancestor]['customers'].update(row_customers)
                item_relationships[ancestor]['pms'].update(row_pms)
    
    return all_items, item_relationships

def filter_data_by_eco(df: pd.DataFrame, eco_number: str) -> pd.DataFrame:
    """
    Filter DataFrame to only include rows for the specified ECO number.
    
    Args:
        df: Full DataFrame
        eco_number: ECO number to filter by
        
    Returns:
        Filtered DataFrame containing only rows for the specified ECO
    """
    st.info(f"🔍 Filtering data for ECO: '{eco_number}'")
    
    # Filter for exact match (case-sensitive)
    filtered_df = df[df['Change_Order'] == eco_number].copy()
    
    st.info(f"📊 Found {len(filtered_df)} rows for ECO '{eco_number}'")
    
    if len(filtered_df) > 0:
        # Parse delimited fields with strict validation
        filtered_df['CustList'] = filtered_df['Customers'].apply(parse_delimited_field)
        filtered_df['PMList'] = filtered_df['PMs'].apply(parse_delimited_field)
        
        # Clean and validate Affected Items
        filtered_df['Affected_PN'] = filtered_df['Affected_PN'].fillna('').astype(str).str.strip()
        
        # Extract all items (affected items + ancestors)
        all_items, item_relationships = extract_all_items_and_ancestors(filtered_df)
        
        # Show summary of parsed data
        total_customers = sum(len(cust_list) for cust_list in filtered_df['CustList'])
        total_pms = sum(len(pm_list) for pm_list in filtered_df['PMList'])
        
        # Count items with and without customers/PMs
        items_with_customers = sum(1 for item_data in item_relationships.values() if item_data['customers'])
        items_without_customers = len(all_items) - items_with_customers
        
        customers_with_pms = 0
        customers_without_pms = 0
        all_customers_in_relationships = set()
        for item_data in item_relationships.values():
            all_customers_in_relationships.update(item_data['customers'])
        
        for customer in all_customers_in_relationships:
            # Check if this customer is associated with any PM through any item
            has_pm = False
            for item_data in item_relationships.values():
                if customer in item_data['customers'] and item_data['pms']:
                    has_pm = True
                    break
            if has_pm:
                customers_with_pms += 1
            else:
                customers_without_pms += 1
        
        # Store the relationships for later use
        filtered_df.attrs['item_relationships'] = item_relationships
        filtered_df.attrs['all_items'] = all_items
        
        # Show sample of filtered data for verification

    else:
        # Show available ECOs for debugging
        all_ecos = sorted(df['Change_Order'].unique())
        st.warning(f"⚠️ No rows found for ECO '{eco_number}'")
        st.info(f"Available ECOs in dataset: {all_ecos[:20]}{'...' if len(all_ecos) > 20 else ''}")
    
    return filtered_df

def validate_hierarchical_data(filtered_df: pd.DataFrame, eco_number: str) -> Tuple[bool, str, Dict]:
    """
    Validate that the filtered ECO data has all required hierarchical levels.
    
    Args:
        filtered_df: DataFrame filtered for specific ECO
        eco_number: ECO number being validated
        
    Returns:
        Tuple of (is_valid, error_message, validation_summary)
    """
    validation_summary = {
        'eco_count': 1 if not filtered_df.empty else 0,
        'total_items': 0,
        'items_with_customers': 0,
        'items_without_customers': 0,
        'customers': 0,
        'customers_with_pms': 0,
        'customers_without_pms': 0,
        'pms': 0,
        'missing_levels': []
    }
    
    if filtered_df.empty:
        return False, f"No data found for ECO '{eco_number}'", validation_summary
    
    # Get the stored relationships
    item_relationships = filtered_df.attrs.get('item_relationships', {})
    all_items = filtered_df.attrs.get('all_items', set())
    
    validation_summary['total_items'] = len(all_items)
    
    if validation_summary['total_items'] == 0:
        validation_summary['missing_levels'].append('Items (Affected Items + Ancestors)')
        return False, f"No valid items (Affected Items or Ancestors) found for ECO '{eco_number}'", validation_summary
    
    # Count items with and without customers
    for item, item_data in item_relationships.items():
        if item_data['customers']:
            validation_summary['items_with_customers'] += 1
        else:
            validation_summary['items_without_customers'] += 1
    
    # Count unique customers and their PM relationships
    all_customers = set()
    for item_data in item_relationships.values():
        all_customers.update(item_data['customers'])
    
    validation_summary['customers'] = len(all_customers)
    
    # Count customers with and without PMs
    for customer in all_customers:
        has_pm = False
        for item_data in item_relationships.values():
            if customer in item_data['customers'] and item_data['pms']:
                has_pm = True
                break
        if has_pm:
            validation_summary['customers_with_pms'] += 1
        else:
            validation_summary['customers_without_pms'] += 1
    
    # Count unique PMs
    all_pms = set()
    for item_data in item_relationships.values():
        all_pms.update(item_data['pms'])
    validation_summary['pms'] = len(all_pms)
    
    # The structure is valid as long as we have items
    return True, "", validation_summary

def build_hierarchical_sankey_data(filtered_df: pd.DataFrame, eco_number: str) -> Dict:
    """
    Build Sankey diagram data with strict hierarchical structure and flow termination:
    Level 1: ECO Number
    Level 2: Items (Affected Items + Ancestors)
    Level 3: Customers (only if they exist for the item)
    Level 4: PMs (only if they exist for the customer)
    
    Args:
        filtered_df: DataFrame filtered for specific ECO
        eco_number: ECO number for the diagram
        
    Returns:
        Dictionary containing hierarchical Sankey data with flow termination
    """
    
    # Get the stored relationships
    item_relationships = filtered_df.attrs.get('item_relationships', {})
    all_items = filtered_df.attrs.get('all_items', set())
    
    # LEVEL 1: ECO (Root node)
    level_1_nodes = [eco_number]
    
    # LEVEL 2: All Items (Affected Items + Ancestors)
    level_2_nodes = sorted(list(all_items))
    
    # LEVEL 3: Customers (only those that exist)
    level_3_nodes = set()
    for item_data in item_relationships.values():
        level_3_nodes.update(item_data['customers'])
    level_3_nodes = sorted(list(level_3_nodes))
    
    # LEVEL 4: PMs (only those that exist)
    level_4_nodes = set()
    for item_data in item_relationships.values():
        level_4_nodes.update(item_data['pms'])
    level_4_nodes = sorted(list(level_4_nodes))
    

    
    # Build complete node list maintaining hierarchy
    all_nodes = level_1_nodes + level_2_nodes + level_3_nodes + level_4_nodes
    node_to_index = {node: i for i, node in enumerate(all_nodes)}
    
    # Build links with strict hierarchy enforcement and flow termination
    links = []
    
    # LINKS: Level 1 (ECO) → Level 2 (Items)
    # Count occurrences of each item (affected or ancestor) across all rows for the given ECO
    item_counts_from_eco = defaultdict(int)
    
    for _, row in filtered_df.iterrows():
        # Count affected item
        affected_item = str(row['Affected_PN']).strip() if pd.notna(row['Affected_PN']) else ''
        if affected_item:
            item_counts_from_eco[affected_item] += 1
        
        # Count ancestors
        if 'Ancestors' in row and pd.notna(row['Ancestors']):
            ancestors = parse_delimited_field(row['Ancestors'])
            for ancestor in ancestors:
                if ancestor:
                    item_counts_from_eco[ancestor] += 1
    
    for item, count in item_counts_from_eco.items():
        if item in node_to_index and eco_number in node_to_index:
            links.append({
                'source': node_to_index[eco_number],
                'target': node_to_index[item],
                'value': int(count),
                'level': '1→2',
                'flow_type': 'eco_to_item'
            })
    
    # LINKS: Level 2 (Items) → Level 3 (Customers)
    # Only create links for items that have customers
    item_customer_link_counts = defaultdict(lambda: defaultdict(int))
    
    for _, row in filtered_df.iterrows():
        # Get all items (affected + ancestors) from this row
        row_items = set()
        affected_item = str(row['Affected_PN']).strip() if pd.notna(row['Affected_PN']) else ''
        if affected_item:
            row_items.add(affected_item)
        if 'Ancestors' in row and pd.notna(row['Ancestors']):
            row_items.update(parse_delimited_field(row['Ancestors']))
        
        row_customers = parse_delimited_field(row['Customers'])
        
        for item in row_items:
            for customer in row_customers:
                item_customer_link_counts[item][customer] += 1
    
    for item, customer_counts in item_customer_link_counts.items():
        if item in node_to_index: # Ensure item is a valid node
            for customer, count in customer_counts.items():
                if customer in node_to_index: # Ensure customer is a valid node
                    links.append({
                        'source': node_to_index[item],
                        'target': node_to_index[customer],
                        'value': count,
                        'level': '2→3',
                        'flow_type': 'item_to_customer'
                    })
    
    # LINKS: Level 3 (Customers) → Level 4 (PMs)
    # Only create links for customers that have PMs
    customer_pm_link_counts = defaultdict(lambda: defaultdict(int))
    
    for _, row in filtered_df.iterrows():
        customers = parse_delimited_field(row['Customers'])
        pms = parse_delimited_field(row['PMs'])
        
        for customer in customers:
            for pm in pms:
                customer_pm_link_counts[customer][pm] += 1
    
    for customer, pm_counts in customer_pm_link_counts.items():
        if customer in node_to_index: # Ensure customer is a valid node
            for pm, count in pm_counts.items():
                if pm in node_to_index: # Ensure PM is a valid node
                    links.append({
                        'source': node_to_index[customer],
                        'target': node_to_index[pm],
                        'value': count,
                        'level': '3→4',
                        'flow_type': 'customer_to_pm'
                    })
    
    # Extract link data for Plotly
    source_indices = [link['source'] for link in links]
    target_indices = [link['target'] for link in links]
    values = [link['value'] for link in links]
    
    # Count flow terminations
    items_terminating_at_level_2 = len([item for item in level_2_nodes if not item_relationships.get(item, {}).get('customers')])
    customers_terminating_at_level_3 = len([customer for customer in level_3_nodes 
                                          if not any(customer_pm_link_counts.get(customer, {}).values())])
    
    st.success(f"✅ Generated {len(links)} hierarchical links with flow termination:")
    link_summary = defaultdict(int)
    for link in links:
        link_summary[link['level']] += 1
    

    return {
        'labels': all_nodes,
        'source': source_indices,
        'target': target_indices,
        'value': values,
        'levels': {
            1: level_1_nodes,
            2: level_2_nodes,
            3: level_3_nodes,
            4: level_4_nodes
        },
        'hierarchy': HIERARCHY_LEVELS,
        'item_relationships': item_relationships, # Keep for detailed breakdown
        'flow_termination': {
            'items_at_level_2': items_terminating_at_level_2,
            'customers_at_level_3': customers_terminating_at_level_3
        }
    }

def create_hierarchical_sankey_figure(sankey_data: Dict, eco_number: str) -> go.Figure:
    """
    Create Plotly Sankey figure with strict hierarchical coloring, positioning, and flow termination.
    
    Args:
        sankey_data: Dictionary containing hierarchical Sankey data
        eco_number: ECO number for the diagram title
        
    Returns:
        Plotly Figure object with hierarchical structure and flow termination
    """
    labels = sankey_data['labels']
    source = sankey_data['source']
    target = sankey_data['target']
    value = sankey_data['value']
    levels = sankey_data['levels']
    
    # Hierarchical color scheme
    LEVEL_COLORS = {
        1: '#1f77b4',  # ECO - Deep Blue
        2: '#ff7f0e',  # Items (Affected + Ancestors) - Orange
        3: '#2ca02c',  # Customers - Green
        4: '#d62728'   # PMs - Red
    }
    
    # Assign colors based on strict hierarchy
    node_colors = []
    for label in labels:
        color_assigned = False
        for level, nodes in levels.items():
            if label in nodes:
                node_colors.append(LEVEL_COLORS[level])
                color_assigned = True
                break
        
        if not color_assigned:
            node_colors.append('#808080')  # Gray for unassigned nodes (shouldn't happen with correct logic)
    
    # Create link colors with transparency based on value
    if value:
        max_value = max(value) if value else 1
        link_colors = []
        for v in value:
            # Use blue-based transparency for all links
            alpha = 0.3 + 0.5 * (v / max_value)  # Alpha between 0.3 and 0.8
            link_colors.append(f'rgba(31, 119, 180, {alpha})')
    else:
        link_colors = []
    
    # Calculate node positions for better hierarchy visualization
    x_positions = [0.05, 0.35, 0.65, 0.95]  # Fixed x positions for each level
    y_positions = {}
    
    for level, nodes in levels.items():
        if nodes:
            level_y_positions = []
            if len(nodes) == 1:
                level_y_positions = [0.5]
            else:
                # Distribute nodes evenly in y-axis for each level
                for i in range(len(nodes)):
                    y_pos = 0.1 + (0.8 * i / (len(nodes) - 1))
                    level_y_positions.append(y_pos)
            
            for i, node in enumerate(nodes):
                y_positions[node] = level_y_positions[i]
    
    # Create node position arrays
    node_x = []
    node_y = []
    
    for label in labels:
        level_found = False
        for level, nodes in levels.items():
            if label in nodes:
                node_x.append(x_positions[level - 1])  # level-1 because levels start at 1
                node_y.append(y_positions.get(label, 0.5)) # Use .get with default for safety
                level_found = True
                break
        
        if not level_found:
            node_x.append(0.5)  # Default position if not found in any level (shouldn't happen)
            node_y.append(0.5)
    
    # Create custom hover data for nodes
    node_customdata = []
    for label in labels:
        level_info = "Unknown"
        for level_num, nodes_at_level in levels.items():
            if label in nodes_at_level:
                level_info = f"Level {level_num} - {HIERARCHY_LEVELS[level_num]}"
                break
        node_customdata.append(level_info)
    
    # Create the Sankey diagram
    fig = go.Figure(go.Sankey(
        arrangement='fixed',  # Use fixed arrangement to maintain hierarchy
        node=dict(
            label=labels,
            color=node_colors,
            pad=20,
            thickness=30,
            line=dict(color='black', width=1),
            x=node_x,
            y=node_y,
            hovertemplate='<b>%{label}</b><br>%{customdata}<br>Connections: %{value}<extra></extra>',
            customdata=node_customdata
        ),
        link=dict(
            source=source,
            target=target,
            value=value,
            color=link_colors,
            hovertemplate='<b>%{source.label}</b> → <b>%{target.label}</b><br>Flow: %{value}<extra></extra>'
        )
    ))
    
    # Update layout with hierarchical title and annotations
    fig.update_layout(
        title=dict(
            text=f"ECO {eco_number} Hierarchical Flow Analysis",
            font=dict(size=18, family='Arial'),
            x=0.378
        ),
        font=dict(size=12, family='Arial'),
        margin=dict(l=50, r=50, t=120, b=50),
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=800,
        annotations=[
            # Level annotations
            dict(x=0.05, y=1.08, text="<b>ECO</b>", showarrow=False, font=dict(size=14, color=LEVEL_COLORS[1])),
            dict(x=0.35, y=1.08, text="<b>Items</b><br><sub>(Affected + Ancestors)</sub>", showarrow=False, font=dict(size=14, color=LEVEL_COLORS[2])),
            dict(x=0.65, y=1.08, text="<b>Customers</b>", showarrow=False, font=dict(size=14, color=LEVEL_COLORS[3])),
            dict(x=0.95, y=1.08, text="<b>Project Managers</b>", showarrow=False, font=dict(size=14, color=LEVEL_COLORS[4]))
        ]
    )
    
    return fig

def display_hierarchical_summary(df: pd.DataFrame, filtered_df: pd.DataFrame, eco_number: str, validation_summary: Dict):
    """
    Display summary statistics about the hierarchical data structure with flow termination info.
    
    Args:
        df: Full DataFrame
        filtered_df: Filtered DataFrame for specific ECO
        eco_number: ECO number being analyzed
        validation_summary: Summary of data validation
    """
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total ECOs", len(df['Change_Order'].unique()))
    
    with col2:
        st.metric("Total Items", validation_summary['total_items'])
        st.caption(f"Affected Items + Ancestors")
    
    with col3:
        st.metric("Customers", validation_summary['customers'])
        st.caption(f"{validation_summary['customers_without_pms']} terminate at Level 3")
    
    with col4:
        st.metric("Project Managers", validation_summary['pms'])

def main():
    """Main Streamlit application with hierarchical Sankey enforcement and flow termination."""
    st.set_page_config(
        page_title='ECO Hierarchical Sankey Analyzer',
        page_icon='📊',
        layout='wide',
        initial_sidebar_state='expanded'
    )
    
    st.title('📊 ECO Hierarchical Flow Analyzer')
    st.markdown('**Structure: ECO → Items (Affected + Ancestors) → Customers → Project Managers**')
    
    # Sidebar for file upload and controls
    with st.sidebar:
        st.header("📁 Data Upload")
        uploaded_file = st.file_uploader(
            "Upload Excel file",
            type=['xlsx', 'xls'],
            help="Upload an Excel file containing ECO data with the required columns"
        )
        
        if uploaded_file:
            st.success(f"✅ File uploaded: {uploaded_file.name}")
            
            # Get available sheets
            sheet_names = get_sheet_names(uploaded_file)
            if sheet_names:
                st.subheader("📋 Sheet Selection")
                # Try to default to 'Combined' sheet if it exists, otherwise use first sheet
                default_index = 0
                if 'Combined' in sheet_names:
                    default_index = sheet_names.index('Combined')
                
                selected_sheet = st.selectbox(
                    "Select sheet to analyze:",
                    sheet_names,
                    index=default_index
                )
            else:
                selected_sheet = None
        
        #hierarchy information
        st.header("🏗️ Hierarchy")
        st.info("**Level 1:** ECO Number")
        st.info("**Level 2:** Items (Affected Items + Ancestors)")
        st.info("**Level 3:** Customers (only if they exist)")
        st.info("**Level 4:** Project Managers (only if they exist)")
        
    
    # Main content area
    if not uploaded_file:
        st.info("👈 Please upload an Excel file to get started")
        st.markdown("""
        ### Required Columns for Hierarchical Analysis:
        - **ECO #**: Engineering Change Order number (Level 1)
        - **Affected Item**: Part numbers affected by the ECO (Level 2)
        - **Customers**: Customer names, comma-separated (Level 3)
        - **PMs**: Project Manager names, comma-separated (Level 4)
        """)
        return
    
    if not sheet_names:
        st.error("❌ Could not read sheet names from the Excel file")
        return
    
    # Load and validate data
    with st.spinner(f"Loading data from sheet '{selected_sheet}'..."):
        df = load_excel_data(uploaded_file, selected_sheet)
    
    if df is None:
        st.error("❌ Failed to load data. Please check your file format and try again.")
        return
    
    # ECO selection section
    st.header("PM Finder")
    
    available_ecos = get_available_ecos(df)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        eco_input = st.text_input(
            "Enter ECO Number",
            placeholder="e.g., C01798",
            help="Enter the exact ECO number you want to analyze (case-sensitive)"
        )
    
    with col2:
        st.markdown("**Available ECOs:**")
        with st.expander(f"View all {len(available_ecos)} ECOs"):
            # Show ECOs in a more organized way
            for i, eco in enumerate(available_ecos):
                if i < 4:  # Show first 4
                    st.text(eco)
                elif i == 4:
                    st.text(f"... and {len(available_ecos) - 100} more")
                    break
    
    # Generate Hierarchical Sankey diagram
    if st.button("🔍 Generate Hierarchical Sankey Diagram", type="primary"):
        if not eco_input.strip():
            st.error("❌ Please enter an ECO number")
            return
        
        eco_number = eco_input.strip()
        
        # Filter data for the specified ECO
        with st.spinner(f"Filtering data for ECO {eco_number}..."):
            filtered_df = filter_data_by_eco(df, eco_number)
        
        # Validate hierarchical data structure
        is_valid, error_message, validation_summary = validate_hierarchical_data(filtered_df, eco_number)
        
        if not is_valid:
            st.error(f"❌ {error_message}")
            
            # Suggest similar ECOs
            similar_ecos = [eco for eco in available_ecos if eco_number.upper() in eco.upper()]
            if similar_ecos:
                st.info(f"💡 Did you mean one of these? {', '.join(similar_ecos[:5])}")
            else:
                st.info("💡 Try checking the exact spelling and case of the ECO number")
            return
        
        # Display hierarchical summary
        st.header("📊 Hierarchical Analysis Results")
        display_hierarchical_summary(df, filtered_df, eco_number, validation_summary)
        
        # Generate and display Hierarchical Sankey diagram
        with st.spinner("Generating hierarchical Sankey diagram..."):
            try:
                sankey_data = build_hierarchical_sankey_data(filtered_df, eco_number)
                fig = create_hierarchical_sankey_figure(sankey_data, eco_number)
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Display detailed hierarchical breakdown
                with st.expander("📋 Hierarchical Data Breakdown"):
                    st.subheader("Raw Data for ECO")
                    display_columns = ['Change_Order', 'Affected_PN', 'Customers', 'PMs']
                    if 'Days Open' in filtered_df.columns:
                        display_columns.insert(1, 'Days Open')
                    if 'Ancestors' in filtered_df.columns:
                        display_columns.insert(-2, 'Ancestors')
                    
                    st.dataframe(filtered_df[display_columns], use_container_width=True)
                    
                    st.subheader("Hierarchical Structure Summary")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.write("**Level 1 - ECO:**")
                        for eco in sankey_data['levels'][1]:
                            st.text(f"• {eco}")
                    
                    with col2:
                        st.write("**Level 2 - Items:**")
                        st.caption("(Affected Items + Ancestors)")
                        for item in sankey_data['levels'][2]:
                            # Show if this item has customers or terminates
                            item_data = sankey_data['item_relationships'].get(item, {})
                            termination = " (→)" if item_data.get('customers') else " (END)"
                            st.text(f"• {item}{termination}")
                    
                    with col3:
                        st.write("**Level 3 - Customers:**")
                        for customer in sankey_data['levels'][3]:
                            # Show if this customer has PMs or terminates
                            # Check if this customer has any PMs associated through any item
                            has_pm = False
                            for item_data in sankey_data['item_relationships'].values():
                                if customer in item_data['customers'] and item_data['pms']:
                                    has_pm = True
                                    break
                            termination = " (→)" if has_pm else " (END)"
                            st.text(f"• {customer}{termination}")
                    
                    with col4:
                        st.write("**Level 4 - PMs:**")
                        for pm in sankey_data['levels'][4]:
                            st.text(f"• {pm} (END)")
                
            except Exception as e:
                st.error(f"❌ Error generating hierarchical Sankey diagram: {str(e)}")
                st.exception(e)

if __name__ == '__main__':
    main()
