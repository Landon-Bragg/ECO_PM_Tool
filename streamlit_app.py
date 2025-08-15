import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import re
from typing import Optional, Dict, List, Tuple, Set
from collections import defaultdict

# ───────────────── CONFIG ───────────────────────────────
REQUIRED_COLUMNS = ["ECO Number", "Affected Item", "ItemType"]
OPTIONAL_COLUMNS = ["Days Open", "Sold To Name", "Program Manager", "Where Used"]

# Hierarchical structure definition - Updated for correct 5 levels
HIERARCHY_LEVELS = {
    1: "ECO",
    2: "Parts", 
    3: "TLAs",
    4: "Customers",
    5: "PMs"
}

# Clean Plotly theme
pio.templates.default = "plotly_white"

# ───────────────── STYLING ───────────────────────────────
def load_custom_css():
    """Load custom CSS for compact, single-view design"""
    st.markdown("""
    <style>
    /* Main container styling - more compact */
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        max-width: 1400px;
    }
    
    /* Compact header styling */
    .compact-header {
        text-align: center;
        padding: 0.5rem 0;
        margin-bottom: 1rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        border-radius: 8px;
        color: white;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    /* Compact section headers */
    .section-header {
        font-size: 1.2rem;
        font-weight: 600;
        margin: 0.5rem 0;
        color: #333;
    }
    
    /* Enhanced PM Card styling - More Compact */
    .pm-card {
        background: linear-gradient(135deg, #f8f9ff 0%, #e8ecff 100%);
        border: 2px solid #9467bd;
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 4px 12px rgba(148, 103, 189, 0.15);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }

    .pm-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(148, 103, 189, 0.25);
    }

    .pm-card-header {
        font-size: 1.1rem;
        font-weight: 700;
        color: #6b46c1;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    .pm-card-content {
        font-size: 0.9rem;
        color: #4b5563;
        line-height: 1.4;
    }
    
    .pm-items-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin-top: 1rem;
    }
    
    .pm-items-section {
        background: rgba(255, 255, 255, 0.8);
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #9467bd;
    }
    
    .pm-items-title {
        font-weight: 600;
        color: #6b46c1;
        margin-bottom: 0.5rem;
        font-size: 0.9rem;
    }
    
    .pm-item {
        background: #f3f4f6;
        padding: 0.3rem 0.6rem;
        margin: 0.2rem 0;
        border-radius: 4px;
        font-size: 0.8rem;
        color: #374151;
        display: inline-block;
        margin-right: 0.3rem;
    }
    
    /* Compact metrics */
    [data-testid="metric-container"] {
        background: white;
        border: 1px solid #e9ecef;
        padding: 0.5rem;
        border-radius: 6px;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
    }
    
    /* Compact buttons */
    .stButton > button {
        padding: 0.5rem 1rem;
        font-size: 0.9rem;
        border-radius: 6px;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
    }
    
    /* Compact file uploader */
    .uploadedFile {
        background: #f8f9fa;
        border: 1px dashed #667eea;
        border-radius: 6px;
        padding: 0.5rem;
        text-align: center;
        font-size: 0.9rem;
    }
    
    /* Compact expanders */
    .streamlit-expanderHeader {
        font-size: 0.9rem;
        padding: 0.5rem;
    }
    
    /* Hide Streamlit branding */
    #MainMenu { display: none !important; }
    footer { display: none !important; }
    header { display: none !important; }
    
    /* Compact tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        padding: 0.5rem 1rem;
        font-size: 0.9rem;
    }
    
    /* Compact dataframes */
    .dataframe {
        font-size: 0.8rem;
    }
    
    /* Reduce spacing between elements */
    .element-container {
        margin-bottom: 0.5rem;
    }
    </style>
    """, unsafe_allow_html=True)

# ───────────────── UTILITY FUNCTIONS ───────────────────────────────

def is_valid_value(value) -> bool:
    """Check if a value is valid (not NaN, None, empty string, or 'nan')"""
    if pd.isna(value):
        return False
    if value is None:
        return False
    str_value = str(value).strip()
    if str_value.upper() in ('', 'NAN', 'N/A', 'NONE'):
        return False
    return True

def clean_value(value) -> Optional[str]:
    """Clean and return a value if valid, otherwise return None"""
    if not is_valid_value(value):
        return None
    return str(value).strip()

# ───────────────── CORE FUNCTIONS ───────────────────────────────

@st.cache_data
def load_excel_data(uploaded_file, sheet_name: str) -> Optional[pd.DataFrame]:
    """Load and validate Excel data with robust missing value handling"""
    try:
        df = pd.read_excel(uploaded_file, sheet_name=sheet_name)
        
        # Check required columns
        missing_required = [col for col in REQUIRED_COLUMNS if col not in df.columns]
        if missing_required:
            st.error(f"Missing required columns: {', '.join(missing_required)}")
            return None
        
        # Clean data - only filter out rows where required fields are missing
        df_clean = df.copy()
        
        # Remove rows where required fields are missing
        initial_count = len(df_clean)
        df_clean = df_clean[df_clean['ECO Number'].apply(is_valid_value)]
        df_clean = df_clean[df_clean['Affected Item'].apply(is_valid_value)]
        df_clean = df_clean[df_clean['ItemType'].apply(is_valid_value)]
        
        if len(df_clean) == 0:
            st.error("No valid data found. All rows are missing required fields.")
            return None
        
        # Report data cleaning results
        removed_count = initial_count - len(df_clean)
        if removed_count > 0:
            st.info(f"Removed {removed_count} rows with missing required fields")
        
        # Standardize columns for consistency
        df_clean['ECO Number'] = df_clean['ECO Number'].astype(str).str.strip()
        df_clean['Change_Order'] = df_clean['ECO Number']
        df_clean['Affected_PN'] = df_clean['Affected Item'].astype(str).str.strip()
        df_clean['Item_Type'] = df_clean['ItemType'].astype(str).str.strip()
        
        # Handle optional columns with missing values
        if 'Sold To Name' in df_clean.columns:
            df_clean['Customer'] = df_clean['Sold To Name'].apply(clean_value)
        else:
            df_clean['Customer'] = None
            
        if 'Program Manager' in df_clean.columns:
            df_clean['PM'] = df_clean['Program Manager'].apply(clean_value)
        else:
            df_clean['PM'] = None
            
        if 'Where Used' in df_clean.columns:
            df_clean['Where_Used'] = df_clean['Where Used'].apply(clean_value)
        else:
            df_clean['Where_Used'] = None
        
        return df_clean
        
    except Exception as e:
        st.error(f"Error reading Excel file: {str(e)}")
        return None

def get_available_ecos(df: pd.DataFrame) -> List[str]:
    """Get sorted list of unique ECO numbers"""
    unique_ecos = df['Change_Order'].unique()
    unique_ecos = [eco for eco in unique_ecos if is_valid_value(eco)]
    return sorted(unique_ecos)

def filter_data_by_eco(df: pd.DataFrame, eco_number: str) -> pd.DataFrame:
    """Filter DataFrame for specific ECO number"""
    filtered_df = df[df['Change_Order'] == eco_number].copy()
    return filtered_df

def analyze_data_completeness(filtered_df: pd.DataFrame) -> Dict:
    """Analyze data completeness for the filtered dataset"""
    total_rows = len(filtered_df)
    
    # Count valid values in each column
    valid_customers = filtered_df['Customer'].apply(lambda x: x is not None).sum()
    valid_pms = filtered_df['PM'].apply(lambda x: x is not None).sum()
    
    # Calculate percentages
    customer_completeness = (valid_customers / total_rows * 100) if total_rows > 0 else 0
    pm_completeness = (valid_pms / total_rows * 100) if total_rows > 0 else 0
    
    return {
        'total_rows': total_rows,
        'valid_customers': valid_customers,
        'valid_pms': valid_pms,
        'customer_completeness': customer_completeness,
        'pm_completeness': pm_completeness,
        'missing_customers': total_rows - valid_customers,
        'missing_pms': total_rows - valid_pms
    }

def build_hierarchical_sankey_data(filtered_df: pd.DataFrame, eco_number: str) -> Dict:
    """Build 5-level Sankey diagram data: ECO → Parts → TLAs → Customers → PMs"""
    
    # Analyze data completeness
    completeness = analyze_data_completeness(filtered_df)
    
    # Initialize sets for nodes at each level
    level_1_nodes_set = {eco_number}
    level_2_nodes_set = set() # Parts
    level_3_nodes_set = set() # TLAs
    level_4_nodes_set = set() # Customers
    level_5_nodes_set = set() # PMs
    
    # Populate node sets based on strict item types and columns
    for _, row in filtered_df.iterrows():
        # Level 2: Parts (only items with ItemType = "Part")
        if row['Item_Type'] == 'Part':
            level_2_nodes_set.add(row['Affected_PN'])
            # Add TLAs from "Where Used" to level 3 (but don't add the part itself to level 3)
            if row['Where_Used'] is not None:
                where_used = str(row['Where_Used']).strip()
                if where_used and where_used != 'nan':
                    tlas = [tla.strip() for tla in where_used.split(',') if tla.strip()]
                    level_3_nodes_set.update(tlas)
    
        # Level 3: TLAs (only items with ItemType = "TLA")
        elif row['Item_Type'] == 'TLA':
            level_3_nodes_set.add(row['Affected_PN'])
    
        # Level 4: Customers (from any record with valid customer data)
        if row['Customer'] is not None:
            level_4_nodes_set.add(row['Customer'])
    
        # Level 5: PMs (from any record with valid PM data)
        if row['PM'] is not None:
            level_5_nodes_set.add(row['PM'])
            
    # Create a definitive mapping of each node to its STRICT level based on data type
    node_to_definitive_level = {}

    # STRICT layer assignment - no conflicts allowed
    # Level 1: ECO (always)
    for node in sorted(list(level_1_nodes_set)):
        node_to_definitive_level[node] = 1

    # Level 2: Parts (ItemType = "Part" only)
    for node in sorted(list(level_2_nodes_set)):
        node_to_definitive_level[node] = 2

    # Level 3: TLAs (ItemType = "TLA" or referenced in "Where Used")
    for node in sorted(list(level_3_nodes_set)):
        node_to_definitive_level[node] = 3

    # Level 4: Customers (from "Sold To Name" only)
    for node in sorted(list(level_4_nodes_set)):
        node_to_definitive_level[node] = 4

    # Level 5: PMs (from "Program Manager" only)
    for node in sorted(list(level_5_nodes_set)):
        node_to_definitive_level[node] = 5
            
    # Build the final ordered list of all nodes for Sankey diagram
    # Sort by definitive level first, then alphabetically within each level
    all_nodes = sorted(node_to_definitive_level.keys(), key=lambda x: (node_to_definitive_level[x], x))
    node_to_index = {node: i for i, node in enumerate(all_nodes)}
    
    # Build links
    links = []

    # LINKS: Level 1 (ECO) → Level 2 (Parts) AND Level 1 (ECO) → Level 3 (TLAs)
    # ECO should connect directly to ALL affected items
    for _, row in filtered_df.iterrows():
        affected_item = row['Affected_PN']
        item_type = row['Item_Type']
        
        # Connect ECO to Parts
        if item_type == 'Part' and affected_item in node_to_index and eco_number in node_to_index:
            links.append({
                'source': node_to_index[eco_number],
                'target': node_to_index[affected_item],
                'value': 1,
                'level': '1→2',
                'flow_type': 'eco_to_part'
            })
        
        # Connect ECO to TLAs (direct TLA affected items)
        elif item_type == 'TLA' and affected_item in node_to_index and eco_number in node_to_index:
            links.append({
                'source': node_to_index[eco_number],
                'target': node_to_index[affected_item],
                'value': 1,
                'level': '1→3',
                'flow_type': 'eco_to_tla'
            })

    # LINKS: Level 2 (Parts) → Level 3 (TLAs via Where Used)
    for _, row in filtered_df[filtered_df['Item_Type'] == 'Part'].iterrows():
        part = row['Affected_PN']
        if row['Where_Used'] is not None:
            where_used = str(row['Where_Used']).strip()
            if where_used and where_used != 'nan':
                tlas = [tla.strip() for tla in where_used.split(',') if tla.strip()]
                for tla in tlas:
                    if part in node_to_index and tla in node_to_index:
                        links.append({
                            'source': node_to_index[part],
                            'target': node_to_index[tla],
                            'value': 1,
                            'level': '2→3',
                            'flow_type': 'part_to_tla'
                        })

    # LINKS: Level 3 (TLAs) → Level 4 (Customers)
    tla_customer_connections = defaultdict(lambda: defaultdict(int))

    for _, row in filtered_df.iterrows():
        customer = row['Customer']
        if customer is None:
            continue 
        
        # Connect TLAs to customers (both direct TLAs and TLAs from Where Used)
        if row['Item_Type'] == 'TLA':
            tla = row['Affected_PN']
            if tla in node_to_index and customer in node_to_index:
                tla_customer_connections[tla][customer] += 1
        
        # Also connect Where Used TLAs to customers through their parts
        elif row['Item_Type'] == 'Part' and row['Where_Used'] is not None:
            where_used = str(row['Where_Used']).strip()
            if where_used and where_used != 'nan':
                tlas = [tla.strip() for tla in where_used.split(',') if tla.strip()]
                for tla in tlas:
                    if tla in node_to_index and customer in node_to_index:
                        tla_customer_connections[tla][customer] += 1

    for tla, customer_counts in tla_customer_connections.items():
        for customer, count in customer_counts.items():
            links.append({
                'source': node_to_index[tla],
                'target': node_to_index[customer],
                'value': count,
                'level': '3→4',
                'flow_type': 'tla_to_customer'
            })

    # LINKS: Level 4 (Customers) → Level 5 (PMs)
    customer_pm_connections = defaultdict(lambda: defaultdict(int))

    for _, row in filtered_df.iterrows():
        customer = row['Customer']
        pm = row['PM']
        
        if customer is None or pm is None:
            continue
        
        if customer in node_to_index and pm in node_to_index:
            customer_pm_connections[customer][pm] += 1
            
    for customer, pm_counts in customer_pm_connections.items():
        for pm, count in pm_counts.items():
            links.append({
                'source': node_to_index[customer],
                'target': node_to_index[pm],
                'value': count,
                'level': '4→5',
                'flow_type': 'customer_to_pm'
            })
    
    # Aggregate link values (e.g., if multiple rows lead to the same A->B link)
    aggregated_links = defaultdict(int)
    for link in links:
        key = (link['source'], link['target'])
        aggregated_links[key] += link['value']
    
    final_source_indices = []
    final_target_indices = []
    final_values = []
    
    for (src, tgt), val in aggregated_links.items():
        final_source_indices.append(src)
        final_target_indices.append(tgt)
        final_values.append(val)
    
    # Build item relationships for detailed breakdown (for PM overview, etc.)
    item_relationships = {}
    for _, row in filtered_df.iterrows():
        item = row['Affected_PN']
        item_type = row['Item_Type']
        customer = row['Customer'] if row['Customer'] is not None else '[Missing Customer]'
        pm = row['PM'] if row['PM'] is not None else 'N/A'
        where_used = row['Where_Used'] if row['Where_Used'] is not None else 'N/A'
        
        if item not in item_relationships:
            item_relationships[item] = {
                'type': item_type,
                'customers': set(), 
                'pms': set(),
                'tlas': set()
            }
        
        item_relationships[item]['customers'].add(customer)
        item_relationships[item]['pms'].add(pm)
        
        if item_type == 'Part' and where_used != 'N/A':
            tlas = [tla.strip() for tla in str(where_used).split(',') if tla.strip()]
            item_relationships[item]['tlas'].update(tlas)
    
    return {
        'labels': all_nodes,
        'source': final_source_indices,
        'target': final_target_indices,
        'value': final_values,
        'levels': {
            1: [n for n, l in node_to_definitive_level.items() if l == 1],
            2: [n for n, l in node_to_definitive_level.items() if l == 2],
            3: [n for n, l in node_to_definitive_level.items() if l == 3],
            4: [n for n, l in node_to_definitive_level.items() if l == 4],
            5: [n for n, l in node_to_definitive_level.items() if l == 5]
        },
        'node_to_definitive_level': node_to_definitive_level,
        'hierarchy': HIERARCHY_LEVELS,
        'item_relationships': item_relationships,
        'raw_data': filtered_df,
        'completeness': completeness
    }

def create_hierarchical_sankey_figure(sankey_data: Dict, eco_number: str) -> go.Figure:
    """Create Plotly Sankey figure with 5-level hierarchy and strict visual positioning"""
    
    labels = sankey_data['labels']
    source = sankey_data['source']
    target = sankey_data['target']
    value = sankey_data['value']
    node_to_definitive_level = sankey_data['node_to_definitive_level']
    
    # Hierarchical color scheme for 5 levels
    LEVEL_COLORS = {
        1: '#1f77b4',  # ECO - Deep Blue
        2: '#ff7f0e',  # Parts - Orange
        3: '#2ca02c',  # TLAs - Green
        4: '#d62728',  # Customers - Red
        5: '#9467bd'   # PMs - Purple
    }
    
    # Assign colors based on definitive hierarchy
    node_colors = [LEVEL_COLORS[node_to_definitive_level[label]] for label in labels]
    
    # Create link colors with transparency based on value
    link_colors = []
    if value:
        max_value = max(value) if value else 1
        for v in value:
            alpha = 0.3 + 0.5 * (v / max_value)
            link_colors.append(f'rgba(31, 119, 180, {alpha})')
    
    # Calculate node positions for better hierarchy visualization (5 levels)
    x_positions = {
        1: 0.02, # ECO
        2: 0.25, # Parts
        3: 0.48, # TLAs
        4: 0.71, # Customers
        5: 0.94  # PMs
    }
    
    # Calculate y-positions within each column to spread nodes vertically
    nodes_by_level = defaultdict(list)
    for label in labels:
        nodes_by_level[node_to_definitive_level[label]].append(label)
    
    y_positions_map = {}
    for level_num, nodes_in_level in nodes_by_level.items():
        nodes_in_level.sort()
        if len(nodes_in_level) == 1:
            y_positions_map[nodes_in_level[0]] = 0.5
        else:
            for i in range(len(nodes_in_level)):
                y_pos = 0.1 + (0.8 * i / (len(nodes_in_level) - 1))
                y_positions_map[nodes_in_level[i]] = y_pos
    
    # Create node position arrays and custom hover data
    node_x = [x_positions[node_to_definitive_level[label]] for label in labels]
    node_y = [y_positions_map.get(label, 0.5) for label in labels]
    
    node_customdata = [f"Level {node_to_definitive_level[label]} - {HIERARCHY_LEVELS[node_to_definitive_level[label]]}" for label in labels]
    
    # Create the Sankey diagram
    fig = go.Figure(go.Sankey(
        arrangement='fixed',
        node=dict(
            label=labels,
            color=node_colors,
            pad=15,
            thickness=25,
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
    
    # Update layout with 5-level hierarchical title and annotations
    fig.update_layout(
        font=dict(size=11, family='Arial', color='white'),
        margin=dict(l=40, r=40, t=80, b=40),
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=600,  # Reduced height for compact view
        annotations=[
            dict(x=0.02, y=1.08, text="<b>ECO</b>", showarrow=False, font=dict(size=12, color=LEVEL_COLORS[1])),
            dict(x=0.25, y=1.08, text="<b>Parts</b>", showarrow=False, font=dict(size=12, color=LEVEL_COLORS[2])),
            dict(x=0.48, y=1.08, text="<b>TLAs</b>", showarrow=False, font=dict(size=12, color=LEVEL_COLORS[3])),
            dict(x=0.71, y=1.08, text="<b>Customers</b>", showarrow=False, font=dict(size=12, color=LEVEL_COLORS[4])),
            dict(x=0.94, y=1.08, text="<b>PMs</b>", showarrow=False, font=dict(size=12, color=LEVEL_COLORS[5]))
        ]
    )
    
    return fig

def create_pm_overview(filtered_df: pd.DataFrame, eco_number: str) -> Dict:
    """Create a comprehensive overview of PM relationships with customers and parts"""
    
    pm_overview = {}
    
    # Group data by PM
    for _, row in filtered_df.iterrows():
        pm = row['PM'] if row['PM'] is not None else '[Missing PM]'
        customer = row['Customer'] if row['Customer'] is not None else '[Missing Customer]'
        part = row['Affected_PN']
        
        # Initialize PM entry if not exists
        if pm not in pm_overview:
            pm_overview[pm] = {
                'customers': {},
                'total_parts': set(),
                'total_customers': set(),
                'record_count': 0
            }
        
        # Initialize customer entry for this PM if not exists
        if customer not in pm_overview[pm]['customers']:
            pm_overview[pm]['customers'][customer] = {
                'parts': set(),
                'record_count': 0
            }
        
        # Add part to customer's parts list
        pm_overview[pm]['customers'][customer]['parts'].add(part)
        pm_overview[pm]['customers'][customer]['record_count'] += 1
        
        # Add to PM's totals
        pm_overview[pm]['total_parts'].add(part)
        pm_overview[pm]['total_customers'].add(customer)
        pm_overview[pm]['record_count'] += 1
    
    # Convert sets to sorted lists for better display
    for pm in pm_overview:
        pm_overview[pm]['total_parts'] = sorted(list(pm_overview[pm]['total_parts']))
        pm_overview[pm]['total_customers'] = sorted(list(pm_overview[pm]['total_customers']))
        
        for customer in pm_overview[pm]['customers']:
            pm_overview[pm]['customers'][customer]['parts'] = sorted(list(pm_overview[pm]['customers'][customer]['parts']))
    
    return pm_overview

def display_compact_pm_overview(pm_overview: Dict, eco_number: str):
    """Display compact PM overview"""
    
    st.markdown('<div class="section-header">👥 Program Manager Overview</div>', unsafe_allow_html=True)
    
    # Summary statistics in compact format
    total_pms = len(pm_overview)
    missing_pm_count = 1 if '[Missing PM]' in pm_overview else 0
    valid_pms = total_pms - missing_pm_count
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total PMs", total_pms, label_visibility="visible")
    with col2:
        st.metric("Valid PMs", valid_pms, label_visibility="visible")
    with col3:
        st.metric("Missing PM", missing_pm_count, label_visibility="visible")
    with col4:
        avg_records = sum(pm_data['record_count'] for pm_data in pm_overview.values()) / len(pm_overview) if pm_overview else 0
        st.metric("Avg Records/PM", f"{avg_records:.1f}", label_visibility="visible")
    
    # Compact PM list
    sorted_pms = sorted([pm for pm in pm_overview.keys() if not pm.startswith('[')])
    if '[Missing PM]' in pm_overview:
        sorted_pms.append('[Missing PM]')
    
    # Display PMs in a compact table format
    pm_data_list = []
    for pm in sorted_pms:
        pm_data = pm_overview[pm]
        customers_str = ', '.join(pm_data['total_customers'][:2])
        if len(pm_data['total_customers']) > 2:
            customers_str += f" (+{len(pm_data['total_customers'])-2} more)"
        
        parts_str = ', '.join(pm_data['total_parts'][:2])
        if len(pm_data['total_parts']) > 2:
            parts_str += f" (+{len(pm_data['total_parts'])-2} more)"
            
        pm_data_list.append({
            'PM': pm,
            'Records': pm_data['record_count'],
            'Customers': customers_str,
            'Parts': parts_str
        })
    
    if pm_data_list:
        pm_df = pd.DataFrame(pm_data_list)
        st.dataframe(pm_df, use_container_width=True, height=200)

def create_pm_cards_for_eco(df: pd.DataFrame, filtered_df: pd.DataFrame, eco_number: str) -> Dict:
    """Create PM cards showing their information for the current ECO only"""
    
    # Get all PMs that appear in this ECO's data
    eco_pms = filtered_df['PM'].dropna().unique()
    
    pm_cards_data = {}
    
    for pm in eco_pms:
        # Get data for this PM in the current ECO only
        pm_eco_data = filtered_df[filtered_df['PM'] == pm]
        
        # Get customers and parts for this PM in the context of the current ECO
        eco_customers = sorted(pm_eco_data['Customer'].dropna().unique())
        eco_parts = sorted(pm_eco_data['Affected_PN'].unique())
        
        # Separate parts by type
        parts_data = pm_eco_data[['Affected_PN', 'Item_Type']].drop_duplicates()
        actual_parts = sorted(parts_data[parts_data['Item_Type'] == 'Part']['Affected_PN'].tolist())
        tlas = sorted(parts_data[parts_data['Item_Type'] == 'TLA']['Affected_PN'].tolist())
        
        pm_cards_data[pm] = {
            'eco_customers': eco_customers,
            'eco_parts': actual_parts,
            'eco_tlas': tlas,
            'eco_record_count': len(pm_eco_data)
        }
    
    return pm_cards_data

def display_pm_cards(pm_cards_data: Dict, eco_number: str):
    """Display compact PM cards underneath the Sankey diagram"""
    
    if not pm_cards_data:
        st.info("No Program Managers found for this ECO.")
        return
    
    st.markdown('<div class="section-header">👤 Program Managers in this ECO</div>', unsafe_allow_html=True)
    
    # Display PM cards in a grid
    num_pms = len(pm_cards_data)
    cols_per_row = 2 if num_pms > 1 else 1
    
    pm_names = sorted(pm_cards_data.keys())
    
    for i in range(0, len(pm_names), cols_per_row):
        cols = st.columns(cols_per_row)
        
        for j, pm_name in enumerate(pm_names[i:i+cols_per_row]):
            with cols[j]:
                pm_data = pm_cards_data[pm_name]
                
                # Create compact PM card with enhanced styling
                st.markdown(f"""
                <div class="pm-card">
                    <div class="pm-card-header">👤 {pm_name}</div>
                    <div class="pm-card-content">
                        <strong>In ECO {eco_number}:</strong><br>
                        • {pm_data['eco_record_count']} records<br>
                        • {len(pm_data['eco_customers'])} customers<br>
                        • {len(pm_data['eco_parts'])} parts<br>
                        • {len(pm_data['eco_tlas'])} TLAs
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Compact expandable details
                with st.expander(f"Details for {pm_name}", expanded=False):
                    
                    # Customers in current ECO
                    st.markdown(f"**Customers in ECO {eco_number}:**")
                    if pm_data['eco_customers']:
                        customers_text = ', '.join(pm_data['eco_customers'])
                        st.text(customers_text)
                    else:
                        st.text("None")
                    
                    # Parts in current ECO
                    st.markdown(f"**Parts in ECO {eco_number}:**")
                    if pm_data['eco_parts']:
                        parts_text = ', '.join(pm_data['eco_parts'])
                        st.text(parts_text)
                    else:
                        st.text("None")
                    
                    # TLAs in current ECO
                    if pm_data['eco_tlas']:
                        st.markdown(f"**TLAs in ECO {eco_number}:**")
                        tlas_text = ', '.join(pm_data['eco_tlas'])
                        st.text(tlas_text)
                    else:
                        st.text("None")

import numpy as np
import pandas as pd

def ecos_missing_pm(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return ECOs that have zero PMs anywhere in the ECO (global).
    Treats NaN and empty/whitespace strings as missing.
    """
    if 'Change_Order' not in df.columns or 'PM' not in df.columns:
        raise KeyError("Expected columns 'Change_Order' and 'PM' to exist.")

    # Normalize PM field: trim whitespace; convert empty/'nan' strings to NaN
    pm_clean = (
        df['PM']
        .astype(str)
        .str.strip()
        .replace({'': np.nan, 'nan': np.nan, 'None': np.nan, 'NaN': np.nan})
    )
    df = df.copy()
    df['PM_clean'] = pm_clean

    # For each ECO, did ANY row have a non-missing PM?
    eco_has_pm = (
        df.groupby('Change_Order')['PM_clean']
          .apply(lambda s: s.notna().any())
          .reset_index(name='has_pm')
    )

    # ECOs with no PMs globally
    missing_ecos = eco_has_pm.loc[~eco_has_pm['has_pm'], 'Change_Order']

    # (Optional context) Customers present on those ECOs
    cust_col = 'Customer' if 'Customer' in df.columns else None
    if cust_col:
        customers = (
            df[df['Change_Order'].isin(missing_ecos)]
            .groupby('Change_Order')[cust_col]
            .apply(lambda s: sorted({x for x in s.dropna().astype(str).str.strip() if x} or ['[No Customers Listed]']))
            .reset_index(name='Customers')
        )
    else:
        customers = pd.DataFrame({'Change_Order': missing_ecos, 'Customers': ['[Column missing]'] * len(missing_ecos)})

    # Basic stats per ECO (records / parts / TLAs)
    if 'Item_Type' in df.columns:
        stats = (
            df[df['Change_Order'].isin(missing_ecos)]
            .groupby('Change_Order')
            .agg(
                records=('Change_Order', 'size'),
                parts=('Item_Type', lambda x: (x == 'Part').sum()),
                tlas=('Item_Type', lambda x: (x == 'TLA').sum()),
            )
            .reset_index()
        )
    else:
        stats = (
            df[df['Change_Order'].isin(missing_ecos)]
            .groupby('Change_Order')
            .agg(records=('Change_Order', 'size'))
            .reset_index()
        )
        stats['parts'] = np.nan
        stats['tlas'] = np.nan

    out = customers.merge(stats, on='Change_Order', how='left')

    # Neat ordering
    return out.sort_values(['records', 'Change_Order'], ascending=[False, True]).reset_index(drop=True)

import numpy as np
import pandas as pd
import streamlit as st

def normalize_pm_col(df: pd.DataFrame) -> pd.Series:
    """Trim whitespace and convert empty/placeholder strings to NaN."""
    return (
        df['PM']
        .astype(str)
        .str.strip()
        .replace({'': np.nan, 'nan': np.nan, 'NaN': np.nan, 'None': np.nan})
    )

def ecos_for_pm(df: pd.DataFrame, pm: str) -> pd.DataFrame:
    """
    List distinct ECOs where this PM appears anywhere.
    Returns quick stats + optional customer summary.
    """
    df = df.copy()
    df['PM_clean'] = normalize_pm_col(df)

    mask = df['PM_clean'] == pm
    ecos = df.loc[mask, 'Change_Order'].dropna().unique().tolist()
    if not ecos:
        return pd.DataFrame(columns=['Change_Order', 'records', 'parts', 'tlas', 'Customers'])

    sub = df[df['Change_Order'].isin(ecos)]

    # Stats
    stats = (
        sub.groupby('Change_Order')
           .agg(
               records=('Change_Order', 'size'),
               parts=('Item_Type', lambda x: (x == 'Part').sum() if 'Item_Type' in sub.columns else np.nan),
               tlas=('Item_Type', lambda x: (x == 'TLA').sum() if 'Item_Type' in sub.columns else np.nan),
           )
           .reset_index()
    )

    # Customers summary (if present)
    if 'Customer' in sub.columns:
        customers = (
            sub.groupby('Change_Order')['Customer']
               .apply(lambda s: sorted({c for c in s.dropna().astype(str).str.strip() if c}) or ['[None]'])
               .reset_index(name='Customers')
        )
        out = stats.merge(customers, on='Change_Order', how='left')
    else:
        out = stats.assign(Customers='[Column missing]')

    return out.sort_values(['records', 'Change_Order'], ascending=[False, True]).reset_index(drop=True)

def eco_detail(df: pd.DataFrame, eco: str) -> pd.DataFrame:
    """Return full-row details for a single ECO, sorted for readability."""
    sub = df[df['Change_Order'] == eco].copy()
    if 'PM' in sub.columns:
        sub['PM'] = sub['PM'].astype(str).str.strip()
    if 'Customer' in sub.columns:
        sub['Customer'] = sub['Customer'].astype(str).str.strip()
    # Nice sort: Item_Type -> Item -> whatever else exists
    sort_cols = [c for c in ['Item_Type', 'Item', 'Customer', 'PM'] if c in sub.columns]
    return sub.sort_values(sort_cols) if sort_cols else sub


# ───────────────── MAIN APPLICATION ───────────────────────────────

def main():
    """Main Streamlit application with compact, single-view design"""
    st.set_page_config(
        page_title='ECO Flow Analyzer',
        page_icon='📊',
        layout='wide',
        initial_sidebar_state='collapsed'
    )
    
    # Load custom CSS
    load_custom_css()
    
    # Compact header
    st.markdown("""
    <div class="compact-header">
        <h1 style="margin: 0; font-size: 1.8rem;">📊 ECO Flow Analyzer</h1>
        <p style="margin: 0.2rem 0 0 0; opacity: 0.9; font-size: 0.9rem;">
            Visualize Engineering Change Order flows through hierarchical layers
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Compact top section with file upload and ECO selection
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<div class="section-header">📁 Upload Data</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Choose Excel file",
            type=['xlsx', 'xls', 'xlsm'],
            help="Upload ECO data file",
            label_visibility="collapsed"
        )
        
        if uploaded_file:
            # Load data
            try:
                excel_file = pd.ExcelFile(uploaded_file)
                sheet_names = excel_file.sheet_names
                
                if len(sheet_names) > 1:
                    selected_sheet = st.selectbox("Sheet:", sheet_names, label_visibility="collapsed")
                else:
                    selected_sheet = sheet_names[0]
                    st.info(f"Using sheet: {selected_sheet}")
                
                df = load_excel_data(uploaded_file, selected_sheet)
                
                if df is not None:
                    with st.expander("🧭 ECOs with no PM (global)", expanded=False):
                        missing_pm_df = ecos_missing_pm(df)
                        st.write(f"**{len(missing_pm_df)}** ECO(s) with no PM found.")
                        st.dataframe(missing_pm_df, use_container_width=True, height=260)

                        csv = missing_pm_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            "⬇️ Download CSV",
                            csv,
                            file_name="ecos_without_pm.csv",
                            mime="text/csv",
                            use_container_width=True
                        )

                if df is not None:
                    with st.expander("👤 PM Workspace — My ECOs", expanded=False):
                        # --- Query params for deep linking ---
                        q = st.query_params
                        qp_pm  = q.get("pm", [None])
                        qp_eco = q.get("eco", [None])
                        qp_pm  = qp_pm[0] if isinstance(qp_pm, list) else qp_pm
                        qp_eco = qp_eco[0] if isinstance(qp_eco, list) else qp_eco

                        # --- PM selection ---
                        pm_options = sorted(
                            normalize_pm_col(df).dropna().unique().tolist()
                        )
                        default_index = 0
                        if qp_pm and qp_pm in pm_options:
                            default_index = ["Select a PM"] + pm_options.index(qp_pm) * [None]  # dummy line to compute
                            default_index = (pm_options.index(qp_pm) + 1)  # because we add "Select a PM" below

                        pm_choice = st.selectbox("Choose your name", ["Select a PM"] + pm_options, index=(default_index if isinstance(default_index, int) else 0))
                        selected_pm = None if pm_choice == "Select a PM" else pm_choice

                        if selected_pm:
                            # --- ECO list for this PM ---
                            pm_ecos_df = ecos_for_pm(df, selected_pm)
                            st.caption(f"Found **{len(pm_ecos_df)}** ECO(s) for **{selected_pm}**.")
                            st.dataframe(pm_ecos_df, use_container_width=True, height=260)

                            # Optional CSV export
                            st.download_button(
                                "⬇️ Download my ECOs (CSV)",
                                pm_ecos_df.to_csv(index=False).encode('utf-8'),
                                file_name=f"{selected_pm}_ecos.csv",
                                mime="text/csv",
                                use_container_width=True
                            )

                            # --- Clickable ECO list & details ---
                            st.markdown("#### Open an ECO")
                            left, right = st.columns([1, 2], gap="large")

                            with left:
                                # Let PM click an ECO from a selectbox (scales better than many buttons)
                                eco_list = pm_ecos_df['Change_Order'].tolist()
                                # Respect deep link if provided
                                default_eco_idx = 0
                                if qp_eco and qp_eco in eco_list:
                                    default_eco_idx = eco_list.index(qp_eco)

                                eco_choice = st.selectbox("Select an ECO", eco_list, index=default_eco_idx if eco_list else 0, key="pm_workspace_eco")

                                # Update query params so the page URL can be shared/bookmarked
                                if eco_choice:
                                    st.query_params.update({"pm": selected_pm, "eco": eco_choice})

                            with right:
                                if eco_choice:
                                    details = eco_detail(df, eco_choice)
                                    st.write(f"**Details for** `{eco_choice}`")
                                    st.dataframe(details, use_container_width=True, height=360)

                                    # Quick exports for the selected ECO
                                    col_a, col_b = st.columns(2)
                                    with col_a:
                                        st.download_button(
                                            "⬇️ Download ECO rows (CSV)",
                                            details.to_csv(index=False).encode('utf-8'),
                                            file_name=f"{eco_choice}_rows.csv",
                                            mime="text/csv",
                                            use_container_width=True
                                        )
                                    with col_b:
                                        # A compact summary of the selected ECO
                                        summary_cols = [c for c in ['Change_Order','Item','Item_Type','Customer','PM'] if c in details.columns]
                                        summary = details[summary_cols] if summary_cols else details
                                        st.download_button(
                                            "⬇️ Download summary (CSV)",
                                            summary.to_csv(index=False).encode('utf-8'),
                                            file_name=f"{eco_choice}_summary.csv",
                                            mime="text/csv",
                                            use_container_width=True
                                        )
                        else:
                            st.info("Select your name to see your ECOs.")




                    
                                        
            except Exception as e:
                st.error(f"Error: {str(e)}")
                df = None
        else:
            df = None
    
    with col2:
        if df is not None:
            st.markdown('<div class="section-header">🔍 Analyze ECO</div>', unsafe_allow_html=True)

            available_ecos = get_available_ecos(df)

            # Unified ECO search: Combines dropdown and text input, default blank
            eco_options = [""] + available_ecos  # Add blank option at the top
            eco_to_analyze = st.selectbox(
                "ECO Number (type or select)",
                options=eco_options,
                index=0,
                placeholder="Type or select ECO number...",
                label_visibility="collapsed"
            )

            if st.button("🚀 Generate Analysis", use_container_width=True):
                if not eco_to_analyze or not eco_to_analyze.strip():
                    st.error("Please enter or select an ECO number")
                else:
                    eco_number = eco_to_analyze.strip()
                    # Filter data
                    filtered_df = filter_data_by_eco(df, eco_number)
                    if filtered_df.empty:
                        st.error(f"No data found for ECO '{eco_number}'")
                    else:
                        # Store in session state for display
                        st.session_state['eco_data'] = {
                            'eco_number': eco_number,
                            'filtered_df': filtered_df,
                            'full_df': df
                        }
                        st.rerun()
        else:
            pass
    
    # Main dashboard view
    if 'eco_data' in st.session_state:
        eco_number = st.session_state['eco_data']['eco_number']
        filtered_df = st.session_state['eco_data']['filtered_df']
        full_df = st.session_state['eco_data']['full_df']
        
        st.markdown(f'<div class="section-header">📈 Flow Analysis for ECO {eco_number}</div>', unsafe_allow_html=True)
        
        # Compact metrics row
        col1, col2, col3, col4, col5 = st.columns(5)
        
        try:
            sankey_data = build_hierarchical_sankey_data(filtered_df, eco_number)
            
            with col1:
                st.metric("Records", len(filtered_df))
            with col2:
                parts = [n for n, l in sankey_data['node_to_definitive_level'].items() if l == 2]
                st.metric("Parts", len(parts))
            with col3:
                tlas = [n for n, l in sankey_data['node_to_definitive_level'].items() if l == 3]
                st.metric("TLAs", len(tlas))
            with col4:
                customers = [n for n, l in sankey_data['node_to_definitive_level'].items() if l == 4]
                st.metric("Customers", len(customers))
            with col5:
                pms = [n for n, l in sankey_data['node_to_definitive_level'].items() if l == 5]
                st.metric("PMs", len(pms))
            
            # Sankey diagram (unchanged design, just more compact)
            fig = create_hierarchical_sankey_figure(sankey_data, eco_number)
            st.plotly_chart(fig, use_container_width=True)
            
            # Enhanced PM Cards underneath the Sankey diagram
            pm_cards_data = create_pm_cards_for_eco(full_df, filtered_df, eco_number)
            display_pm_cards(pm_cards_data, eco_number)
            
            # Details section underneath
            st.markdown("---")
            st.markdown('<div class="section-header">📋 Detailed Analysis</div>', unsafe_allow_html=True)
            
            # Compact tabs for details
            tab1, tab2, tab3 = st.tabs(["PM Overview", "Data Summary", "Raw Data"])
            
            with tab1:
                pm_overview = create_pm_overview(filtered_df, eco_number)
                display_compact_pm_overview(pm_overview, eco_number)
            
            with tab2:
                completeness = sankey_data['completeness']
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Records", completeness['total_rows'])
                with col2:
                    st.metric("Customer Data", f"{completeness['customer_completeness']:.1f}%")
                with col3:
                    st.metric("PM Data", f"{completeness['pm_completeness']:.1f}%")
                with col4:
                    st.metric("Missing Data", completeness['missing_customers'] + completeness['missing_pms'])
                
                # Compact item relationships
                st.markdown("**Item Relationships:**")
                for _, row in filtered_df.head(10).iterrows():  # Show only first 10 for compactness
                    item = row['Affected_PN']
                    item_type = row['Item_Type']
                    customer = row['Customer'] if row['Customer'] is not None else '[Missing]'
                    pm = row['PM'] if row['PM'] is not None else '[Missing]'
                    st.text(f"{item} ({item_type}) → {customer} → {pm}")
                
                if len(filtered_df) > 10:
                    st.info(f"Showing first 10 of {len(filtered_df)} items")
            
            with tab3:
                display_columns = ['Change_Order', 'Affected_PN', 'Item_Type', 'Customer', 'PM', 'Where_Used']
                display_df = filtered_df[display_columns].copy()
                display_df['Customer'] = display_df['Customer'].fillna('[Missing]')
                display_df['PM'] = display_df['PM'].fillna('[Missing]')
                display_df['Where_Used'] = display_df['Where_Used'].fillna('N/A')
                
                st.dataframe(display_df, use_container_width=True, height=400)
                
        except Exception as e:
            st.error(f"Error creating visualization: {str(e)}")

if __name__ == '__main__':
    main()
