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

# ───────────────── CORE FUNCTIONS ───────────────────────────────

@st.cache_data
def load_excel_data(uploaded_file, sheet_name: str) -> Optional[pd.DataFrame]:
    """Load and validate Excel data with minimal logging"""
    try:
        df = pd.read_excel(uploaded_file, sheet_name=sheet_name)
        
        # Check required columns
        missing_required = [col for col in REQUIRED_COLUMNS if col not in df.columns]
        if missing_required:
            st.error(f"Missing required columns: {', '.join(missing_required)}")
            return None
        
        # Clean data
        df_clean = df.copy()
        df_clean = df_clean.dropna(subset=['ECO #'])
        df_clean = df_clean[df_clean['ECO #'].astype(str).str.strip() != '']
        df_clean = df_clean[df_clean['ECO #'].astype(str).str.upper() != 'NAN']
        
        # Standardize columns
        df_clean['ECO #'] = df_clean['ECO #'].astype(str).str.strip()
        df_clean['Change_Order'] = df_clean['ECO #']
        df_clean['Affected_PN'] = df_clean['Affected Item']
        
        return df_clean
        
    except Exception as e:
        st.error(f"Error reading Excel file: {str(e)}")
        return None

def parse_delimited_field(field_value: str) -> List[str]:
    """Parse comma-delimited field and return cleaned values"""
    if pd.isna(field_value) or field_value == '' or str(field_value).upper() == 'NAN':
        return []
    
    items = [item.strip() for item in re.split(r',\s*', str(field_value))]
    items = [item for item in items if item and item != '#N/A' and item.upper() != 'NAN' and item.strip() != '']
    
    return items

def get_available_ecos(df: pd.DataFrame) -> List[str]:
    """Get sorted list of unique ECO numbers"""
    unique_ecos = df['Change_Order'].unique()
    unique_ecos = [eco for eco in unique_ecos if pd.notna(eco) and str(eco).upper() != 'NAN']
    return sorted(unique_ecos)

def extract_all_items_and_ancestors(filtered_df: pd.DataFrame) -> Tuple[Set[str], Dict[str, Dict]]:
    """Extract all affected items and ancestors with relationships"""
    all_items = set()
    item_relationships = {}
    
    for _, row in filtered_df.iterrows():
        # Get affected item
        affected_item = str(row['Affected_PN']).strip() if pd.notna(row['Affected_PN']) else ''
        
        # Get ancestors if available
        ancestors = []
        if 'Ancestors' in row and pd.notna(row['Ancestors']):
            ancestors = parse_delimited_field(row['Ancestors'])
        
        # Get customers and PMs
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
    """Filter DataFrame for specific ECO number"""
    filtered_df = df[df['Change_Order'] == eco_number].copy()
    
    if len(filtered_df) > 0:
        # Parse delimited fields
        filtered_df['CustList'] = filtered_df['Customers'].apply(parse_delimited_field)
        filtered_df['PMList'] = filtered_df['PMs'].apply(parse_delimited_field)
        filtered_df['Affected_PN'] = filtered_df['Affected_PN'].fillna('').astype(str).str.strip()
        
        # Extract items and relationships
        all_items, item_relationships = extract_all_items_and_ancestors(filtered_df)
        
        # Store relationships
        filtered_df.attrs['item_relationships'] = item_relationships
        filtered_df.attrs['all_items'] = all_items
    
    return filtered_df

def build_hierarchical_sankey_data(filtered_df: pd.DataFrame, eco_number: str) -> Dict:
    """Build Sankey diagram data with hierarchical structure"""
    
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
    
    # LINKS: Level 3 (Customers) → Level 4 (PMs)
    customer_pm_link_counts = defaultdict(lambda: defaultdict(int))
    
    for _, row in filtered_df.iterrows():
        customers = parse_delimited_field(row['Customers'])
        pms = parse_delimited_field(row['PMs'])
        
        for customer in customers:
            for pm in pms:
                customer_pm_link_counts[customer][pm] += 1
    
    for customer, pm_counts in customer_pm_link_counts.items():
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
    
    # Count flow terminations
    items_terminating_at_level_2 = len([item for item in level_2_nodes if not item_relationships.get(item, {}).get('customers')])
    customers_terminating_at_level_3 = len([customer for customer in level_3_nodes 
                                          if not any(customer_pm_link_counts.get(customer, {}).values())])
    
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
        'flow_termination': {
            'items_at_level_2': items_terminating_at_level_2,
            'customers_at_level_3': customers_terminating_at_level_3
        }
    }

def create_hierarchical_sankey_figure(sankey_data: Dict, eco_number: str) -> go.Figure:
    """Create Plotly Sankey figure with strict hierarchical coloring, positioning, and flow termination"""
    
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
            node_colors.append('#808080')
    
    # Create link colors with transparency based on value
    if value:
        max_value = max(value) if value else 1
        link_colors = []
        for v in value:
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
            dict(x=0.35, y=1.08, text="<b>Items</b><br><sub>(Affected + Ancestors)</sub>", showarrow=False, font=dict(size=14, color=LEVEL_COLORS[2])),
            dict(x=0.65, y=1.08, text="<b>Customers</b>", showarrow=False, font=dict(size=14, color=LEVEL_COLORS[3])),
            dict(x=0.95, y=1.08, text="<b>Project Managers</b>", showarrow=False, font=dict(size=14, color=LEVEL_COLORS[4]))
        ]
    )
    
    return fig

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
            Simple • Clean • Intuitive
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
            - `ECO #` - Engineering Change Order number
            - `Affected Item` - Part numbers affected by the ECO
            - `Customers` - Customer names (comma-separated)
            - `PMs` - Project Manager names (comma-separated)
            
            **Optional Columns:**
            - `Ancestors` - Related ancestor items
            - `Days Open` - Duration information
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
            placeholder="e.g., C01798",
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
                
                # Summary metrics
                st.subheader("📈 Flow Summary")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Items", len(sankey_data['levels'][2]))
                with col2:
                    st.metric("Customers", len(sankey_data['levels'][3]))
                with col3:
                    st.metric("Project Managers", len(sankey_data['levels'][4]))
                with col4:
                    st.metric("Total Connections", len(sankey_data['source']))
                
                # Data details (collapsible)
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
                            item_data = sankey_data['item_relationships'].get(item, {})
                            termination = " (→)" if item_data.get('customers') else " (END)"
                            st.text(f"• {item}{termination}")
                    
                    with col3:
                        st.write("**Level 3 - Customers:**")
                        for customer in sankey_data['levels'][3]:
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
                st.error(f"Error creating visualization: {str(e)}")

if __name__ == '__main__':
    main()
