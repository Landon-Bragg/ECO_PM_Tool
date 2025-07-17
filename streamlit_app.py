import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import re
from typing import Optional, Dict, List, Tuple, Set
from collections import defaultdict

# ───────────────── CONFIG ───────────────────────────────
REQUIRED_COLUMNS = ["ECO Number", "Affected Item"]
OPTIONAL_COLUMNS = ["Days Open", "Sold To Name", "Program Manager"]

# Hierarchical structure definition
HIERARCHY_LEVELS = {
    1: "ECO",
    2: "Items", 
    3: "Customers",
    4: "PMs"
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
    if str_value == '' or str_value.upper() == 'NAN':
        return False
    return True

def clean_value(value) -> Optional[str]:
    """Clean and return a value if valid, otherwise return None"""
    if not is_valid_value(value):
        return None
    return str(value).strip()

def get_termination_placeholder(level: str) -> str:
    """Get placeholder name for terminated flows"""
    placeholders = {
        'customer': '[No Customer Data]',
        'pm': '[No PM Data]'
    }
    return placeholders.get(level, '[Missing Data]')

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
        
        # Clean data - only filter out rows where ECO Number or Affected Item are missing
        df_clean = df.copy()
        
        # Remove rows where required fields are missing
        initial_count = len(df_clean)
        df_clean = df_clean[df_clean['ECO Number'].apply(is_valid_value)]
        df_clean = df_clean[df_clean['Affected Item'].apply(is_valid_value)]
        
        if len(df_clean) == 0:
            st.error("No valid data found. All rows are missing ECO Number or Affected Item.")
            return None
        
        # Report data cleaning results
        removed_count = initial_count - len(df_clean)
        if removed_count > 0:
            st.info(f"Removed {removed_count} rows with missing ECO Number or Affected Item")
        
        # Standardize columns for consistency
        df_clean['ECO Number'] = df_clean['ECO Number'].astype(str).str.strip()
        df_clean['Change_Order'] = df_clean['ECO Number']
        df_clean['Affected_PN'] = df_clean['Affected Item'].astype(str).str.strip()
        
        # Handle optional columns with missing values
        if 'Sold To Name' in df_clean.columns:
            df_clean['Customer'] = df_clean['Sold To Name'].apply(clean_value)
        else:
            df_clean['Customer'] = None
            
        if 'Program Manager' in df_clean.columns:
            df_clean['PM'] = df_clean['Program Manager'].apply(clean_value)
        else:
            df_clean['PM'] = None
        
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
    """Build Sankey diagram data with graceful handling of missing values"""
    
    # Analyze data completeness
    completeness = analyze_data_completeness(filtered_df)
    
    # LEVEL 1: ECO (Root node)
    level_1_nodes = [eco_number]
    
    # LEVEL 2: All unique affected items (always present due to required field validation)
    level_2_nodes = sorted(filtered_df['Affected_PN'].unique().tolist())
    
    # LEVEL 3: Customers (including termination placeholder if needed)
    level_3_nodes = []
    valid_customers = filtered_df[filtered_df['Customer'].notna()]['Customer'].unique()
    if len(valid_customers) > 0:
        level_3_nodes.extend(sorted(valid_customers.tolist()))
    
    # Add termination placeholder if there are missing customers
    if completeness['missing_customers'] > 0:
        level_3_nodes.append(get_termination_placeholder('customer'))
    
    # LEVEL 4: PMs (including termination placeholder if needed)
    level_4_nodes = []
    valid_pms = filtered_df[filtered_df['PM'].notna()]['PM'].unique()
    if len(valid_pms) > 0:
        level_4_nodes.extend(sorted(valid_pms.tolist()))
    
    # Add termination placeholder if there are missing PMs
    if completeness['missing_pms'] > 0:
        level_4_nodes.append(get_termination_placeholder('pm'))
    
    # Build complete node list maintaining hierarchy
    all_nodes = level_1_nodes + level_2_nodes + level_3_nodes + level_4_nodes
    node_to_index = {node: i for i, node in enumerate(all_nodes)}
    
    # Build links with graceful termination handling
    links = []
    
    # LINKS: Level 1 (ECO) → Level 2 (Items)
    # Count how many times each item appears for this ECO
    item_counts = filtered_df['Affected_PN'].value_counts().to_dict()
    
    for item, count in item_counts.items():
        if item in node_to_index and eco_number in node_to_index:
            links.append({
                'source': node_to_index[eco_number],
                'target': node_to_index[item],
                'value': int(count),
                'level': '1→2',
                'flow_type': 'eco_to_item'
            })
    
    # LINKS: Level 2 (Items) → Level 3 (Customers or Termination)
    item_customer_counts = defaultdict(lambda: defaultdict(int))
    
    for _, row in filtered_df.iterrows():
        item = row['Affected_PN']
        customer = row['Customer']
        
        # If customer is missing, route to termination placeholder
        if customer is None:
            customer = get_termination_placeholder('customer')
        
        item_customer_counts[item][customer] += 1
    
    for item, customer_counts in item_customer_counts.items():
        if item in node_to_index:
            for customer, count in customer_counts.items():
                if customer in node_to_index:
                    links.append({
                        'source': node_to_index[item],
                        'target': node_to_index[customer],
                        'value': count,
                        'level': '2→3',
                        'flow_type': 'item_to_customer'
                    })
    
    # LINKS: Level 3 (Customers) → Level 4 (PMs or Termination)
    # Only create links from actual customers (not termination placeholders)
    customer_pm_counts = defaultdict(lambda: defaultdict(int))
    
    for _, row in filtered_df.iterrows():
        customer = row['Customer']
        pm = row['PM']
        
        # Only process if customer is valid (not None)
        if customer is not None:
            # If PM is missing, route to termination placeholder
            if pm is None:
                pm = get_termination_placeholder('pm')
            
            customer_pm_counts[customer][pm] += 1
    
    for customer, pm_counts in customer_pm_counts.items():
        if customer in node_to_index:
            for pm, count in pm_counts.items():
                if pm in node_to_index:
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
    
    # Build item relationships for detailed breakdown
    item_relationships = {}
    for _, row in filtered_df.iterrows():
        item = row['Affected_PN']
        customer = row['Customer'] if row['Customer'] is not None else get_termination_placeholder('customer')
        pm = row['PM'] if row['PM'] is not None else get_termination_placeholder('pm')
        
        if item not in item_relationships:
            item_relationships[item] = {'customers': set(), 'pms': set()}
        
        item_relationships[item]['customers'].add(customer)
        item_relationships[item]['pms'].add(pm)
    
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
        'item_relationships': item_relationships,
        'raw_data': filtered_df,
        'completeness': completeness
    }

def create_hierarchical_sankey_figure(sankey_data: Dict, eco_number: str) -> go.Figure:
    """Create Plotly Sankey figure with special styling for termination nodes"""
    
    labels = sankey_data['labels']
    source = sankey_data['source']
    target = sankey_data['target']
    value = sankey_data['value']
    levels = sankey_data['levels']
    
    # Hierarchical color scheme
    LEVEL_COLORS = {
        1: '#1f77b4',  # ECO - Deep Blue
        2: '#ff7f0e',  # Items - Orange
        3: '#2ca02c',  # Customers - Green
        4: '#d62728'   # PMs - Red
    }
    
    # Special color for termination placeholders
    TERMINATION_COLOR = '#808080'  # Gray
    
    # Assign colors based on hierarchy and termination status
    node_colors = []
    for label in labels:
        # Check if this is a termination placeholder
        if label.startswith('[') and label.endswith(']'):
            node_colors.append(TERMINATION_COLOR)
        else:
            color_assigned = False
            for level, nodes in levels.items():
                if label in nodes:
                    node_colors.append(LEVEL_COLORS[level])
                    color_assigned = True
                    break
            
            if not color_assigned:
                node_colors.append('#808080')
    
    # Create link colors with transparency based on value
    if value:
        max_value = max(value) if value else 1
        link_colors = []
        for i, v in enumerate(value):
            # Use different color for links going to termination nodes
            target_label = labels[target[i]]
            if target_label.startswith('[') and target_label.endswith(']'):
                # Gray color for termination links
                alpha = 0.2 + 0.3 * (v / max_value)
                link_colors.append(f'rgba(128, 128, 128, {alpha})')
            else:
                # Blue color for normal links
                alpha = 0.3 + 0.5 * (v / max_value)
                link_colors.append(f'rgba(31, 119, 180, {alpha})')
    else:
        link_colors = []
    
    # Calculate node positions for better hierarchy visualization
    x_positions = [0.05, 0.35, 0.65, 0.95]
    y_positions = {}
    
    for level, nodes in levels.items():
        if nodes:
            level_y_positions = []
            if len(nodes) == 1:
                level_y_positions = [0.5]
            else:
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
                node_x.append(x_positions[level - 1])
                node_y.append(y_positions.get(label, 0.5))
                level_found = True
                break
        
        if not level_found:
            node_x.append(0.5)
            node_y.append(0.5)
    
    # Create custom hover data for nodes
    node_customdata = []
    for label in labels:
        if label.startswith('[') and label.endswith(']'):
            level_info = "Termination Point - Missing Data"
        else:
            level_info = "Unknown"
            for level_num, nodes_at_level in levels.items():
                if label in nodes_at_level:
                    level_info = f"Level {level_num} - {HIERARCHY_LEVELS[level_num]}"
                    break
        node_customdata.append(level_info)
    
    # Create the Sankey diagram
    fig = go.Figure(go.Sankey(
        arrangement='fixed',
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
            font=dict(size=18, family='Arial', color='white'),
            x=0.378
        ),
        font=dict(size=12, family='Arial', color='white'),
        margin=dict(l=50, r=50, t=120, b=50),
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=800,
        annotations=[
            dict(x=0.05, y=1.08, text="<b>ECO</b>", showarrow=False, font=dict(size=14, color=LEVEL_COLORS[1])),
            dict(x=0.35, y=1.08, text="<b>Items</b>", showarrow=False, font=dict(size=14, color=LEVEL_COLORS[2])),
            dict(x=0.65, y=1.08, text="<b>Customers</b>", showarrow=False, font=dict(size=14, color=LEVEL_COLORS[3])),
            dict(x=0.95, y=1.08, text="<b>Project Managers</b>", showarrow=False, font=dict(size=14, color=LEVEL_COLORS[4]))
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
            - `Affected Item` - Part numbers affected by the ECO
            
            **Optional Columns:**
            - `Sold To Name` - Customer names (can have missing values)
            - `Program Manager` - Project Manager names (can have missing values)
            - `Days Open` - Duration information
            
            **Missing Data Handling:**
            - Rows with missing ECO Number or Affected Item will be excluded
            - Missing customer or PM data will be shown as termination points
            - The flow will gracefully terminate where data is incomplete
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
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        eco_input = st.text_input(
            "Enter ECO Number",
            placeholder="e.g., C05706",
            help="Enter the exact ECO number to analyze"
        )
    
    with col2:
        with st.expander(f"📝 Available ECOs ({len(available_ecos)})"):
            # Show first 4 ECOs
            for eco in available_ecos[:4]:
                st.text(eco)
            if len(available_ecos) > 4:
                st.text(f"... and {len(available_ecos) - 4} more")
    
    # Generate analysis
    if st.button("🚀 Generate Flow Analysis"):
        if not eco_input.strip():
            st.error("Please enter an ECO number")
            return
        
        eco_number = eco_input.strip()
        
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
                    st.metric("Items", len(sankey_data['levels'][2]))
                with col2:
                    st.metric("Customers", len([c for c in sankey_data['levels'][3] if not c.startswith('[')]))
                with col3:
                    st.metric("Project Managers", len([p for p in sankey_data['levels'][4] if not p.startswith('[')]))
                with col4:
                    st.metric("Total Connections", len(sankey_data['source']))
                
                # Data details (collapsible)
                with st.expander("📋 Detailed Data Breakdown"):
                    st.subheader("Raw Data for ECO")
                    display_columns = ['Change_Order', 'Affected_PN']
                    if 'Days Open' in filtered_df.columns:
                        display_columns.append('Days Open')
                    display_columns.extend(['Customer', 'PM'])
                    
                    # Create display dataframe with missing value indicators
                    display_df = filtered_df[display_columns].copy()
                    display_df['Customer'] = display_df['Customer'].fillna('[Missing Customer]')
                    display_df['PM'] = display_df['PM'].fillna('[Missing PM]')
                    
                    st.dataframe(display_df, use_container_width=True)
                    
                    st.subheader("Item-Customer-PM Relationships")
                    
                    # Show detailed relationships with missing data indicators
                    for item in sankey_data['levels'][2]:
                        item_data = filtered_df[filtered_df['Affected_PN'] == item]
                        if not item_data.empty:
                            st.write(f"**{item}:**")
                            for _, row in item_data.iterrows():
                                customer = row['Customer'] if row['Customer'] is not None else '[Missing Customer]'
                                pm = row['PM'] if row['PM'] is not None else '[Missing PM]'
                                st.text(f"  → {customer} (PM: {pm})")
                            st.write("")
                    
                
            except Exception as e:
                st.error(f"Error creating visualization: {str(e)}")
                st.write("Debug info:")
                st.write(f"Filtered data shape: {filtered_df.shape}")
                st.write(f"Columns: {filtered_df.columns.tolist()}")

if __name__ == '__main__':
    main()
