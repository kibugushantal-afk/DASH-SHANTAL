
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Student Performance Analytics",
    page_icon="Statistics",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS 
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #blue;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #black;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin-bottom: 1rem;
    }
    .section-header {
        color: #red;
        border-bottom: 2px solid #1f77b4;
        padding-bottom: 0.5rem;
        margin-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">Student Performance Analytics Dashboard</h1>', unsafe_allow_html=True)

# Generate sample student data
st.cache_data
def generate_student_data():
    np.random.seed(42)
    n_students = 250
    
    majors = ['Science', 'Engineering', 'Business', 'Arts', 'Education']
    
    data = {
        'StudentID': [f'STU{1000 + i}' for i in range(n_students)],
        'Gender': np.random.choice(['Male', 'Female', 'Other'], n_students, p=[0.45, 0.5, 0.05]),
        'Age': np.random.randint(18, 25, n_students),
        'StudyHoursPerWeek': np.random.normal(20, 8, n_students).clip(5, 40),
        'AttendanceRate': np.random.normal(85, 12, n_students).clip(50, 100),
        'GPA': np.random.normal(3.2, 0.6, n_students).clip(2.0, 4.0),
        'Major': np.random.choice(majors, n_students),
        'PartTimeJob': np.random.choice(['Yes', 'No'], n_students, p=[0.4, 0.6]),
        'ExtracurricularActivities': np.random.choice(['None', '1 activity', '2+ activities'], n_students, p=[0.3, 0.5, 0.2])
    }
    
    df = pd.DataFrame(data)
    
    # Create performance categories based on GPA
    df['Performance'] = pd.cut(df['GPA'], 
                              bins=[2.0, 2.5, 3.0, 3.5, 4.0], 
                              labels=['Needs Improvement', 'Average', 'Good', 'Excellent'])
    
    return df

# Load data
df = generate_student_data()

# Sidebar filters
st.sidebar.header("Search Filter Students")

# Major filter
selected_majors = st.sidebar.multiselect(
    "Select Major:",
    options=sorted(df['Major'].unique()),
    default=sorted(df['Major'].unique())
)

# Gender filtering
selected_genders = st.sidebar.multiselect(
    "Select Gender:",
    options=df['Gender'].unique(),
    default=df['Gender'].unique()
)

# Part-time job filter
selected_job_status = st.sidebar.multiselect(
    "Part-time Job:",
    options=df['PartTimeJob'].unique(),
    default=df['PartTimeJob'].unique()
)

# Extracurricular activities filteration
selected_activities = st.sidebar.multiselect(
    "Extracurricular Activities:",
    options=df['ExtracurricularActivities'].unique(),
    default=df['ExtracurricularActivities'].unique()
)

# GPA range filtering
gpa_range = st.sidebar.slider(
    "GPA Range:",
    min_value=2.0,
    max_value=4.0,
    value=(2.0, 4.0),
    step=0.1
)

# Study hours filtering
study_hours_range = st.sidebar.slider(
    "Study Hours Per Week:",
    min_value=5,
    max_value=40,
    value=(5, 40)
)

# Attendance filteration
attendance_range = st.sidebar.slider(
    "Attendance Rate (%):",
    min_value=50,
    max_value=100,
    value=(50, 100)
)

# Apply filters
filtered_df = df[
    (df['Major'].isin(selected_majors)) &
    (df['Gender'].isin(selected_genders)) &
    (df['PartTimeJob'].isin(selected_job_status)) &
    (df['ExtracurricularActivities'].isin(selected_activities)) &
    (df['GPA'] >= gpa_range[0]) &
    (df['GPA'] <= gpa_range[1]) &
    (df['StudyHoursPerWeek'] >= study_hours_range[0]) &
    (df['StudyHoursPerWeek'] <= study_hours_range[1]) &
    (df['AttendanceRate'] >= attendance_range[0]) &
    (df['AttendanceRate'] <= attendance_range[1])
]

# Key metrics
st.markdown("### Statistics Key Performance Indicators")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("Total Students", len(filtered_df))
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("Average GPA", f"{filtered_df['GPA'].mean():.2f}")
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("Average Study Hours", f"{filtered_df['StudyHoursPerWeek'].mean():.1f}")
    st.markdown('</div>', unsafe_allow_html=True)

with col4:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("Average Attendance", f"{filtered_df['AttendanceRate'].mean():.1f}%")
    st.markdown('</div>', unsafe_allow_html=True)

# Main tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Performance Overview", 
    " Academic Performance", 
    "Study Analysis", 
    "Demographic Student Details", 
    "Search Insights"
])

with tab1:
    st.markdown('<h3 class="section-header">Overall Performance Overview</h3>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Performance distribution
        fig_performance = px.pie(
            filtered_df, 
            names='Performance',
            title='Student Performance Distribution',
            color='Performance',
            color_discrete_map={
                'Needs Improvement': 'Red', 
                'Average': 'Neon green', 
                'Good': 'White', 
                'Excellent': 'Purple'
            }
        )
        st.plotly_chart(fig_performance, use_container_width=True)
        
        # Gender distribution
        fig_gender = px.pie(
            filtered_df, 
            names='Gender',
            title='Gender Distribution'
        )
        st.plotly_chart(fig_gender, use_container_width=True)
    
    with col2:
        # Major distribution
        fig_major = px.bar(
            filtered_df['Major'].value_counts().reset_index(),
            x='Major',
            y='count',
            title='Students by Major',
            labels={'count': 'Number of Students', 'Major': 'Field of Study'}
        )
        fig_major.update_layout(xaxis_tickangle=45)
        st.plotly_chart(fig_major, use_container_width=True)
        
        # Extracurricular activities
        fig_extra = px.bar(
            filtered_df['ExtracurricularActivities'].value_counts().reset_index(),
            x='ExtracurricularActivities',
            y='count',
            title='Extracurricular Activities Participation',
            color='ExtracurricularActivities'
        )
        st.plotly_chart(fig_extra, use_container_width=True)

with tab2:
    st.markdown('<h3 class="section-header">Academic Performance Analysis</h3>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # GPA by Major
        fig_gpa_major = px.box(
            filtered_df,
            x='Major',
            y='GPA',
            title='GPA Distribution by Major',
            color='Major'
        )
        fig_gpa_major.update_layout(xaxis_tickangle=45)
        st.plotly_chart(fig_gpa_major, use_container_width=True)
        
        # GPA by Gender
        fig_gpa_gender = px.box(
            filtered_df,
            x='Gender',
            y='GPA',
            title='GPA Distribution by Gender',
            color='Gender'
        )
        st.plotly_chart(fig_gpa_gender, use_container_width=True)
    
    with col2:
        # GPA by Part-time Job
        fig_gpa_job = px.box(
            filtered_df,
            x='PartTimeJob',
            y='GPA',
            title='GPA vs Part-time Job',
            color='PartTimeJob'
        )
        st.plotly_chart(fig_gpa_job, use_container_width=True)
        
        # GPA by Extracurricular Activities
        fig_gpa_extra = px.box(
            filtered_df,
            x='ExtracurricularActivities',
            y='GPA',
            title='GPA vs Extracurricular Activities',
            color='ExtracurricularActivities'
        )
        st.plotly_chart(fig_gpa_extra, use_container_width=True)

with tab3:
    st.markdown('<h3 class="section-header">Study Habits & Attendance Analysis</h3>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Study Hours vs GPA
        fig_study_gpa = px.scatter(
    filtered_df,
    x='AttendanceRate',      
    y='GPA',                    
    color='Performance',        
    title='Study Hours vs GPA',
    trendline='lowess',         
    size='StudyHoursPerWeek',      
    hover_data=['Major', 'Gender']  
)
        st.plotly_chart(fig_study_gpa, use_container_width=True)
        
        # Study hours distribution
        fig_study_dist = px.histogram(
            filtered_df,
            x='StudyHoursPerWeek',
            nbins=20,
            title='Study Hours Distribution',
            color_discrete_sequence=['orange']
        )
        st.plotly_chart(fig_study_dist, use_container_width=True)
    
    with col2:
        # Attendance vs GPA
        fig_attendance_gpa = px.scatter(
            filtered_df,
            x='AttendanceRate',
            y='GPA',
            color='Performance',
            title='Attendance Rate vs GPA',
            trendline='lowess',
            size='StudyHoursPerWeek',
            hover_data=['Major', 'Gender']
        )
        st.plotly_chart(fig_attendance_gpa, use_container_width=True)
        
        fig_attendance_gpa = px.scatter(
    filtered_df,
    x='AttendanceRate',         
    y='GPA',                    
    color='Performance',        
    title='Attendance Rate vs GPA',
    trendline='lowess',         
    size='StudyHoursPerWeek',   
    hover_data=['Major', 'Gender'] 
)
        
        # Attendance distribution
        fig_attendance_dist = px.histogram(
            filtered_df,
            x='AttendanceRate',
            nbins=20,
            title='Attendance Rate Distribution',
            color_discrete_sequence=['brown']
        )
        st.plotly_chart(fig_attendance_dist, use_container_width=True)

with tab4:
    st.markdown('<h3 class="section-header">Student Details & Data</h3>', unsafe_allow_html=True)
    
    # Search and sort options
    col1, col2, col3 = st.columns(3)
    
    with col1:
        search_student = st.text_input("Search by Student ID:")
    
    with col2:
        sort_by = st.selectbox(
            "Sort by:",
            ['GPA', 'StudyHoursPerWeek', 'AttendanceRate', 'Age', 'Major']
        )
    
    with col3:
        sort_order = st.radio("Order:", ['Descending', 'Ascending'])
    
    # Filter data based on search
    display_data = filtered_df.copy()
    
    if search_student:
        display_data = display_data[display_data['StudentID'].str.contains(search_student, case=False)]
    
    # Sort data
    ascending = sort_order == 'Ascending'
    display_data = display_data.sort_values(by=sort_by, ascending=ascending)
    
    # Display data table
    st.dataframe(
        display_data,
        use_container_width=True,
        height=400
    )
    
    # Download data
    st.markdown("### Export Data")
    csv = filtered_df.to_csv(index=False)
    st.download_button(
        label="Download Filtered Data as CSV",
        data=csv,
        file_name=f"student_performance_data_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )

with tab5:
    st.markdown('<h3 class="section-header">Key Insights & Correlations</h3>', unsafe_allow_html=True)
    
    if len(filtered_df) > 0:
        # Calculate correlations
        numeric_cols = ['Age', 'StudyHoursPerWeek', 'AttendanceRate', 'GPA']
        correlation_matrix = filtered_df[numeric_cols].corr()
        
        # Display correlation heatmap
        fig_corr = px.imshow(
            correlation_matrix,
            title='Correlation Matrix: Academic Factors',
            aspect='auto',
            color_continuous_scale='RdBu_r',
            zmin=-1, zmax=1
        )
        st.plotly_chart(fig_corr, use_container_width=True)
        
        # Key insights
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Performance Insights")
            
            # Study hours correlation
            study_corr = filtered_df['StudyHoursPerWeek'].corr(filtered_df['GPA'])
            st.info(f"**Study Hours Impact**: Correlation with GPA: {study_corr:.3f}")
            
            # Attendance correlation
            attendance_corr = filtered_df['AttendanceRate'].corr(filtered_df['GPA'])
            st.info(f"**Attendance Impact**: Correlation with GPA: {attendance_corr:.3f}")
            
            # Top performing major
            top_major = filtered_df.groupby('Major')['GPA'].mean().idxmax()
            top_major_gpa = filtered_df.groupby('Major')['GPA'].mean().max()
            st.success(f"**Top Performing Major**: {top_major} (Avg GPA: {top_major_gpa:.2f})")
        
        with col2:
            st.markdown("### Behavioral Insights")
            
            # Part-time job impact
            job_gpa = filtered_df.groupby('PartTimeJob')['GPA'].mean()
            job_impact = job_gpa['Yes'] - job_gpa['No']
            st.warning(f"**Part-time Job Impact**: GPA difference: {job_impact:+.2f}")
            
            # Extracurricular impact
            extra_gpa = filtered_df.groupby('ExtracurricularActivities')['GPA'].mean()
            if len(extra_gpa) > 1:
                extra_impact = extra_gpa.max() - extra_gpa.min()
                st.info(f"**Extracurricular Impact**: GPA range: {extra_impact:.2f}")
            
            # Study hours recomend
            avg_study_top = filtered_df[filtered_df['Performance'] == 'Excellent']['StudyHoursPerWeek'].mean()
            st.success(f"**Study Recommendation**: Top performers average {avg_study_top:.1f} study hours/week")
    
    else:
        st.warning("No data available with current filters. Please adjust your filter settings.")
        
 
# summary statistics in expander
with st.expander(" Dataset Summary"):
    st.write(f"**Total Students in Dataset:** {len(df)}")
    st.write(f"**Data Columns:** {', '.join(df.columns)}")
    st.write("**Sample Data:**")
    st.dataframe(df.head(10), use_container_width=True)       

# Footer
st.markdown("---")
st.markdown(" **Student Performance Analytics Dashboard** | Built with Streamlit • Pandas • Plotly")




   
