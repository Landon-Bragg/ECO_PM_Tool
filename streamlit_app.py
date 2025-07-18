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
    """Load custom CSS for clean, minimalist design"""
    st.markdown("""
    <style>
    /* Main container styling */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    
    /* Header styling */
    .main-header {
        text-align: center;
        padding: 1.5rem 0;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        border-radius: 12px;
        color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Card styling */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        border-left: 4px solid #667eea;
        margin-bottom: 1rem;
    }
    
    /* Clean button styling */
    .stButton > button {
        width: 100%;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
    
    /* File uploader styling */
    .uploadedFile {
        background: #f8f9fa;
        border: 2px dashed #667eea;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }
    
    /* Success/Error message styling */
    .stSuccess, .stError, .stInfo, .stWarning {
        border-radius: 8px;
        border: none;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: #f8f9fa;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Clean metrics */
    [data-testid="metric-container"] {
        background: white;
        border: 1px solid #e9ecef;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
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
        customer = row['Customer'] if row['Customer'] is not None else '[Missing Customer]' # Keep placeholders for textual summary
        pm = row['PM'] if row['PM'] is not None else '[Missing PM]' # Keep placeholders for textual summary
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
        'node_to_definitive_level': node_to_definitive_level, # Pass this for positioning
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
            link_colors.append(f'rgba(31, 119, 180, {alpha})') # Default blue for links
    
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
        nodes_in_level.sort() # Sort alphabetically for consistent y-positioning
        if len(nodes_in_level) == 1:
            y_positions_map[nodes_in_level[0]] = 0.5
        else:
            for i in range(len(nodes_in_level)):
                y_pos = 0.1 + (0.8 * i / (len(nodes_in_level) - 1))
                y_positions_map[nodes_in_level[i]] = y_pos # Corrected line
    
    # Create node position arrays and custom hover data
    node_x = [x_positions[node_to_definitive_level[label]] for label in labels]
    node_y = [y_positions_map.get(label, 0.5) for label in labels] # Use calculated y-positions
    
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
        margin=dict(l=40, r=40, t=120, b=40),
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=900,
        annotations=[
            dict(x=0.02, y=1.1, text="<b>ECO</b>", showarrow=False, font=dict(size=14, color=LEVEL_COLORS[1])),
            dict(x=0.25, y=1.1, text="<b>Parts</b>", showarrow=False, font=dict(size=14, color=LEVEL_COLORS[2])),
            dict(x=0.48, y=1.1, text="<b>TLAs</b>", showarrow=False, font=dict(size=14, color=LEVEL_COLORS[3])),
            dict(x=0.71, y=1.1, text="<b>Customers</b>", showarrow=False, font=dict(size=14, color=LEVEL_COLORS[4])),
            dict(x=0.94, y=1.1, text="<b>PMs</b>", showarrow=False, font=dict(size=14, color=LEVEL_COLORS[5]))
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

def display_pm_overview(pm_overview: Dict, eco_number: str):
    """Display the PM overview in a structured, readable format"""
    
    st.subheader(f"👥 Program Manager Overview for ECO {eco_number}")
    
    # Summary statistics
    total_pms = len(pm_overview)
    missing_pm_count = 1 if '[Missing PM]' in pm_overview else 0
    valid_pms = total_pms - missing_pm_count
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total PMs", total_pms)
    with col2:
        st.metric("Valid PMs", valid_pms)
    with col3:
        st.metric("Missing PM Data", missing_pm_count)
    
    # Sort PMs - put missing PM at the end
    sorted_pms = sorted([pm for pm in pm_overview.keys() if not pm.startswith('[')])
    if '[Missing PM]' in pm_overview:
        sorted_pms.append('[Missing PM]')
    
    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["📋 Detailed View", "📊 Summary Table", "🔍 Search & Filter"])
    
    with tab1:
        # Detailed view for each PM
        for pm in sorted_pms:
            pm_data = pm_overview[pm]
            
            # PM header with styling
            if pm.startswith('['):
                st.markdown(f"""
                <div style="background-color: #f8f9fa; padding: 1rem; border-radius: 8px; border-left: 4px solid #6c757d; margin: 1rem 0;">
                    <h4 style="margin: 0; color: #6c757d;">⚠️ {pm}</h4>
                    <p style="margin: 0.5rem 0 0 0; color: #6c757d;">
                        {pm_data['record_count']} records • {len(pm_data['total_customers'])} customers • {len(pm_data['total_parts'])} parts
                    </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="background-color: #e3f2fd; padding: 1rem; border-radius: 8px; border-left: 4px solid #2196f3; margin: 1rem 0;">
                    <h4 style="margin: 0; color: #1976d2;">👤 {pm}</h4>
                    <p style="margin: 0.5rem 0 0 0; color: #1976d2;">
                        {pm_data['record_count']} records • {len(pm_data['total_customers'])} customers • {len(pm_data['total_parts'])} parts
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
            # Customer and parts breakdown
            with st.expander(f"View details for {pm}", expanded=False):
                
                # Quick summary
                st.write("**📈 Quick Summary:**")
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Total Customers:** {len(pm_data['total_customers'])}")
                    for customer in pm_data['total_customers']:
                        customer_display = customer if not customer.startswith('[') else f"⚠️ {customer}"
                        st.write(f"  • {customer_display}")
                
                with col2:
                    st.write(f"**Total Parts:** {len(pm_data['total_parts'])}")
                    for part in pm_data['total_parts']:
                        st.write(f"  • {part}")
                
                st.divider()
                
                # Detailed customer-part relationships
                st.write("**🔗 Customer-Part Relationships:**")
                
                for customer in sorted(pm_data['customers'].keys()):
                    customer_data = pm_data['customers'][customer]
                    
                    if customer.startswith('['):
                        st.markdown(f"**⚠️ {customer}** ({customer_data['record_count']} records)")
                    else:
                        st.markdown(f"**🏢 {customer}** ({customer_data['record_count']} records)")
                    
                    # Display parts for this customer
                    parts_per_row = 3
                    parts_list = customer_data['parts']
                    
                    for i in range(0, len(parts_list), parts_per_row):
                        cols = st.columns(parts_per_row)
                        for j, part in enumerate(parts_list[i:i+parts_per_row]):
                            with cols[j]:
                                st.write(f"  📦 {part}")
                    
                    st.write("")  # Add spacing
    
    with tab2:
        # Summary table view
        st.write("**📊 PM Summary Table**")
        
        # Create summary data for table
        summary_data = []
        for pm in sorted_pms:
            pm_data = pm_overview[pm]
            summary_data.append({
                'Program Manager': pm,
                'Total Records': pm_data['record_count'],
                'Customers': len(pm_data['total_customers']),
                'Parts': len(pm_data['total_parts']),
                'Customer List': ', '.join(pm_data['total_customers'][:3]) + ('...' if len(pm_data['total_customers']) > 3 else ''),
                'Part List': ', '.join(pm_data['total_parts'][:3]) + ('...' if len(pm_data['total_parts']) > 3 else '')
            })
        
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True)
        
        # Additional statistics
        st.write("**📈 Statistics:**")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            avg_customers = sum(len(pm_data['total_customers']) for pm_data in pm_overview.values()) / len(pm_overview)
            st.metric("Avg Customers per PM", f"{avg_customers:.1f}")
        
        with col2:
            avg_parts = sum(len(pm_data['total_parts']) for pm_data in pm_overview.values()) / len(pm_overview)
            st.metric("Avg Parts per PM", f"{avg_parts:.1f}")
        
        with col3:
            avg_records = sum(pm_data['record_count'] for pm_data in pm_overview.values()) / len(pm_overview)
            st.metric("Avg Records per PM", f"{avg_records:.1f}")
    
    with tab3:
        # Search and filter functionality
        st.write("**🔍 Search & Filter**")
        
        # Search by PM name
        pm_search = st.text_input("Search Program Manager:", placeholder="Enter PM name...")
        
        # Filter by customer
        all_customers = set()
        for pm_data in pm_overview.values():
            all_customers.update(pm_data['total_customers'])
        customer_filter = st.selectbox("Filter by Customer:", ['All'] + sorted(list(all_customers)))
        
        # Filter by part
        all_parts = set()
        for pm_data in pm_overview.values():
            all_parts.update(pm_data['total_parts'])
        part_filter = st.selectbox("Filter by Part:", ['All'] + sorted(list(all_parts)))
        
        # Apply filters
        filtered_pms = []
        for pm in sorted_pms:
            pm_data = pm_overview[pm]
            
            # PM name filter
            if pm_search and pm_search.lower() not in pm.lower():
                continue
            
            # Customer filter
            if customer_filter != 'All' and customer_filter not in pm_data['total_customers']:
                continue
            
            # Part filter
            if part_filter != 'All' and part_filter not in pm_data['total_parts']:
                continue
            
            filtered_pms.append(pm)
        
        # Display filtered results
        if filtered_pms:
            st.write(f"**Found {len(filtered_pms)} matching PM(s):**")
            
            for pm in filtered_pms:
                pm_data = pm_overview[pm]
                
                with st.expander(f"{pm} ({pm_data['record_count']} records)"):
                    
                    # Show relevant customers and parts based on filters
                    relevant_customers = pm_data['total_customers']
                    relevant_parts = pm_data['total_parts']
                    
                    if customer_filter != 'All':
                        relevant_customers = [customer_filter]
                    if part_filter != 'All':
                        relevant_parts = [part_filter]
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**Customers:**")
                        for customer in relevant_customers:
                            st.write(f"  • {customer}")
                    
                    with col2:
                        st.write("**Parts:**")
                        for part in relevant_parts:
                            st.write(f"  • {part}")
                    
                    # Show specific relationships if both filters are applied
                    if customer_filter != 'All' and part_filter != 'All':
                        if customer_filter in pm_data['customers'] and part_filter in pm_data['customers'][customer_filter]['parts']:
                            st.success(f"✅ {pm} manages {part_filter} for {customer_filter}")
                        else:
                            st.warning(f"❌ No direct relationship found between {customer_filter} and {part_filter} for {pm}")
        else:
            st.info("No PMs match the current filters.")

# ───────────────── MAIN APPLICATION ───────────────────────────────

def main():
    """Main Streamlit application with clean, minimalist design"""
    st.set_page_config(
        page_title='ECO Flow Analyzer',
        page_icon='📊',
        layout='wide',
        initial_sidebar_state='collapsed'
    )
    
    # Load custom CSS
    load_custom_css()
    
    # Main header
    st.markdown("""
    <div class="main-header">
        <h1 style="margin: 0; font-size: 2.5rem;">📊 ECO Flow Analyzer</h1>
        <p style="margin: 0.5rem 0 0 0; opacity: 0.9; font-size: 1.1rem;">
            Visualize Engineering Change Order flows through hierarchical layers
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # File upload section
    st.subheader("📁 Upload Your Data")
    uploaded_file = st.file_uploader(
        "Choose an Excel file",
        type=['xlsx', 'xls'],
        help="Upload an Excel file containing ECO data"
    )
    
    if not uploaded_file:
        st.info("👆 Upload an Excel file to get started")
        
        # Show requirements in a clean format
        with st.expander("📋 Required Data Format"):
            st.markdown("""
            **Required Columns:**
            - `ECO Number` - Engineering Change Order number
            - `Affected Item` - Part numbers or TLA names
            - `ItemType` - Either "Part" or "TLA"
            
            **Optional Columns:**
            - `Sold To Name` - Customer names (can have missing values)
            - `Program Manager` - Project Manager names (can have missing values)
            - `Days Open` - Duration information
            - `Where Used` - TLA references for Parts (comma-separated if multiple)
            
            **Flow Structure:**
            - ECO → Parts → TLAs → Customers → PMs
            - Parts (ItemType = "Part") connect to TLAs via "Where Used" field
            - TLAs (ItemType = "TLA") can also be direct affected items
            - "Where Used" field supports comma-separated multiple TLAs
            - Missing customer or PM data will be shown as termination points
            """)
        return
    
    # Load data
    with st.spinner("Loading data..."):
        # Get sheet names
        try:
            excel_file = pd.ExcelFile(uploaded_file)
            sheet_names = excel_file.sheet_names
        except Exception as e:
            st.error(f"Error reading Excel file: {str(e)}")
            return
        
        # Sheet selection
        if len(sheet_names) > 1:
            selected_sheet = st.selectbox("Select sheet:", sheet_names)
        else:
            selected_sheet = sheet_names[0]
        
        # Load data
        df = load_excel_data(uploaded_file, selected_sheet)
    
    if df is None:
        return
    
    # Success message and data summary
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📊 Total Records", len(df))
    with col2:
        st.metric("🔢 Unique ECOs", len(df['Change_Order'].unique()))
    with col3:
        st.metric("📋 Sheet", selected_sheet)
    
    st.success(f"✅ Data loaded successfully!")
    
    # ECO selection
    st.subheader("🔍 Analyze ECO")

    available_ecos = get_available_ecos(df)

    col1, col2 = st.columns([2, 1])

    with col1:
        eco_input = st.text_input(
            "Enter ECO Number",
            placeholder="e.g., C05706",
            help="Enter the exact ECO number to analyze",
            key="eco_text_input"
        )

    with col2:
        # Simple selectbox dropdown that's scrollable by default
        selected_eco = st.selectbox(
            "Or select from list:",
            options=[''] + available_ecos,
            format_func=lambda x: "Choose an ECO..." if x == '' else x,
            help=f"Select from {len(available_ecos)} available ECOs",
            key="eco_selectbox"
        )

    # Use the selected ECO if available, otherwise use manual input
    if selected_eco and selected_eco != '':
        eco_to_analyze = selected_eco
        # Show which ECO was selected
        st.info(f"Selected ECO: {selected_eco}")
    else:
        eco_to_analyze = eco_input

    # Generate analysis
    if st.button("🚀 Generate Flow Analysis"):
        if not eco_to_analyze.strip():
            st.error("Please enter or select an ECO number")
            return
        
        eco_number = eco_to_analyze.strip()
        
        # Filter data
        with st.spinner("Analyzing data..."):
            filtered_df = filter_data_by_eco(df, eco_number)
        
        if filtered_df.empty:
            st.error(f"No data found for ECO '{eco_number}'")
            
            # Suggest similar ECOs
            similar_ecos = [eco for eco in available_ecos if eco_number.upper() in eco.upper()]
            if similar_ecos:
                st.info(f"💡 Similar ECOs: {', '.join(similar_ecos[:5])}")
            return
        
        # Display results
        st.success(f"Found {len(filtered_df)} records for ECO {eco_number}")
        
        # Generate Sankey diagram
        with st.spinner("Creating visualization..."):
            try:
                sankey_data = build_hierarchical_sankey_data(filtered_df, eco_number)
                fig = create_hierarchical_sankey_figure(sankey_data, eco_number)
                
                # Display chart
                st.plotly_chart(fig, use_container_width=True)
                
                # PM Overview Section
                pm_overview = create_pm_overview(filtered_df, eco_number)
                display_pm_overview(pm_overview, eco_number)
                
                # Data completeness summary
                completeness = sankey_data['completeness']
                
                st.subheader("📊 Data Completeness Summary")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Records", completeness['total_rows'])
                with col2:
                    st.metric("Customer Data", f"{completeness['customer_completeness']:.1f}%", 
                             delta=f"{completeness['valid_customers']}/{completeness['total_rows']}")
                with col3:
                    st.metric("PM Data", f"{completeness['pm_completeness']:.1f}%",
                             delta=f"{completeness['valid_pms']}/{completeness['total_rows']}")
                with col4:
                    st.metric("Missing Data Points", completeness['missing_customers'] + completeness['missing_pms'])
                
                # Summary metrics
                st.subheader("📈 Flow Summary")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    # Count only actual parts, not including ECO or TLAs that are not parts
                    actual_parts = [n for n, l in sankey_data['node_to_definitive_level'].items() if l == 2]
                    st.metric("Parts", len(actual_parts))
                with col2:
                    actual_tlas = [n for n, l in sankey_data['node_to_definitive_level'].items() if l == 3]
                    st.metric("TLAs", len(actual_tlas))
                with col3:
                    actual_customers = [n for n, l in sankey_data['node_to_definitive_level'].items() if l == 4]
                    st.metric("Customers", len(actual_customers))
                with col4:
                    actual_pms = [n for n, l in sankey_data['node_to_definitive_level'].items() if l == 5]
                    st.metric("Project Managers", len(actual_pms))
                
                # Data details (collapsible)
                with st.expander("📋 Detailed Data Breakdown"):
                    st.subheader("Raw Data for ECO")
                    display_columns = ['Change_Order', 'Affected_PN', 'Item_Type']
                    if 'Days Open' in filtered_df.columns:
                        display_columns.append('Days Open')
                    display_columns.extend(['Customer', 'PM', 'Where_Used'])
                    
                    # Create display dataframe with missing value indicators
                    display_df = filtered_df[display_columns].copy()
                    display_df['Customer'] = display_df['Customer'].fillna('[Missing Customer]')
                    display_df['PM'] = display_df['PM'].fillna('[Missing PM]')
                    display_df['Where_Used'] = display_df['Where_Used'].fillna('N/A')
                    
                    st.dataframe(display_df, use_container_width=True)
                    
                    st.subheader("Item-Customer-PM Relationships")
                    
                    # Show detailed relationships with missing data indicators
                    # Iterate through the original filtered_df to get all relationships
                    for _, row in filtered_df.iterrows():
                        item = row['Affected_PN']
                        item_type = row['Item_Type']
                        customer = row['Customer'] if row['Customer'] is not None else '[Missing Customer]'
                        pm = row['PM'] if row['PM'] is not None else '[Missing PM]'
                        where_used = row['Where_Used'] if row['Where_Used'] is not None else 'N/A'
                        
                        st.write(f"**{item} (Type: {item_type}):**")
                        st.text(f"  → Customer: {customer} (PM: {pm})")
                        if item_type == 'Part' and where_used != 'N/A':
                            st.text(f"  → Where Used (TLAs): {where_used}")
                        st.write("")
                    
                    # Missing data summary
                    if completeness['missing_customers'] > 0 or completeness['missing_pms'] > 0:
                        st.subheader("⚠️ Missing Data Summary")
                        if completeness['missing_customers'] > 0:
                            st.warning(f"**{completeness['missing_customers']} records** are missing customer information. These flows terminate at the TLA level in the diagram.")
                        if completeness['missing_pms'] > 0:
                            st.warning(f"**{completeness['missing_pms']} records** are missing PM information. These flows terminate at the Customer level in the diagram.")
                        st.info("Flows with incomplete data terminate at the last available node in the diagram for clarity.")
              
            except Exception as e:
                st.error(f"Error creating visualization: {str(e)}")
                st.write("Debug info:")
                st.write(f"Filtered data shape: {filtered_df.shape}")
                st.write(f"Columns: {filtered_df.columns.tolist()}")
                if 'sankey_data' in locals():
                    st.write(f"Sankey Data (labels, source, target, value):")
                    st.write(f"Labels: {sankey_data.get('labels', 'N/A')}")
                    st.write(f"Source: {sankey_data.get('source', 'N/A')}")
                    st.write(f"Target: {sankey_data.get('target', 'N/A')}")
                    st.write(f"Value: {sankey_data.get('value', 'N/A')}")
                    st.write(f"Node to Definitive Level: {sankey_data.get('node_to_definitive_level', 'N/A')}")


if __name__ == '__main__':
    main()
