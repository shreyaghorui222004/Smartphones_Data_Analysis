import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import seaborn as sns
import matplotlib.pyplot as plt
from io import StringIO

# Cache data loading for performance
@st.cache_data
def load_data(file_path):
    try:
        df = pd.read_csv("Final_Smartphones_Cleaned_Data.csv")
        # Dynamically detect numeric and boolean columns
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        boolean_cols = df.select_dtypes(include=['bool']).columns
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        for col in boolean_cols:
            df[col] = df[col].astype(bool)
        return df
    except FileNotFoundError:
        st.error("Data file not found. Please check the file path.")
        return pd.DataFrame()

# Load your data
df = load_data("Final_Smartphones_Cleaned_Data.csv")  # Adjusted to match function

# App Title and Description
st.title("Smartphone Data Analysis Dashboard")
st.markdown("""
This dashboard provides an interactive analysis of 1,006 smartphones with 29 features, including price, hardware, display, camera, battery, and connectivity metrics. Use the sidebar to apply specific filters and explore colorful visualizations.
""")

# Dynamically detect available columns for filters
filterable_cols = [col for col in ['brand', 'os', 'rating', 'price_numeric', 'battery_watt'] if col in df.columns]
default_values = {col: df[col].dropna().unique() if col in df.columns else [] for col in filterable_cols}

# Sidebar for Specific Filters
st.sidebar.header("Filters")
selected_filters = {}
for col in filterable_cols:
    if col in ['rating', 'price_numeric', 'battery_watt']:
        min_val = float(df[col].min()) if col in df.columns else 0
        max_val = float(df[col].max()) if col in df.columns else 100
        selected_filters[col] = st.sidebar.slider(f"{col.replace('_', ' ').title()} Range", min_value=min_val, max_value=max_val, value=(min_val, max_val))
    else:
        selected_filters[col] = st.sidebar.multiselect(f"Select {col.replace('_', ' ').title()}", options=df[col].dropna().unique() if col in df.columns else [], default=default_values[col])

# Apply Filters
filtered_df = df.copy()
for col, value in selected_filters.items():
    if col in ['rating', 'price_numeric', 'battery_watt']:
        filtered_df = filtered_df[filtered_df[col].between(value[0], value[1])]
    else:
        filtered_df = filtered_df[filtered_df[col].isin(value)]
filtered_df = filtered_df.dropna()

# Tabs for Different Sections
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Summary", "Distributions", "Comparisons", "Correlations", "Outlier Detection"])

with tab1:
    st.header("Summary Statistics")
    if not filtered_df.empty:
        st.dataframe(filtered_df.describe())
        csv = filtered_df.to_csv(index=False)
        st.download_button(label="Download Filtered Data as CSV", data=csv, file_name="filtered_smartphone_data.csv", mime="text/csv")
    st.subheader("Insights")
    st.markdown("""
    - The summary statistics provide a snapshot of central tendencies and variability for numeric features like price, rating, and battery wattage.
    - High standard deviations in price_numeric suggest a wide range of smartphone costs, indicating diverse market segments.
    - The minimum and maximum values for display_size and battery_watt highlight the variety in hardware specifications across brands.
    """)

with tab2:
    st.header("Distributions")
    if 'rating' in filtered_df.columns:
        fig_rating = px.histogram(filtered_df, x='rating', nbins=20, title="Rating Distribution", color_discrete_sequence=['#FF6384'], opacity=0.8)
        st.plotly_chart(fig_rating)
        st.markdown("**Description**: This histogram displays the distribution of user ratings (out of 5) across 1,006 smartphones, using a vibrant pink shade to highlight popular rating ranges.")
    if 'price_numeric' in filtered_df.columns:
        fig_price = px.histogram(filtered_df, x='price_numeric', nbins=20, title="Price Distribution (INR)", color_discrete_sequence=['#36A2EB'], opacity=0.8)
        st.plotly_chart(fig_price)
        st.markdown("**Description**: This histogram shows the price distribution in INR, with a bright blue color to emphasize common price segments.")
    if 'display_size' in filtered_df.columns:
        fig_display = px.histogram(filtered_df, x='display_size', nbins=15, title="Display Size Distribution (Inches)", color_discrete_sequence=['#FFCE56'], opacity=0.8)
        st.plotly_chart(fig_display)
        st.markdown("**Description**: This histogram illustrates display size distribution with a warm yellow tone, highlighting the prevalence of various screen sizes.")
    st.subheader("Insights")
    st.markdown("""
    - The rating distribution often peaks around 4.0-4.5, suggesting most smartphones receive high user satisfaction.
    - Price distribution may show a bimodal pattern, with peaks at budget (<₹20,000) and premium (>₹50,000) segments.
    - Display sizes are likely concentrated between 6-7 inches, reflecting current market trends for larger screens.
    """)

with tab3:
    st.header("Comparisons")
    if 'brand' in filtered_df.columns and 'price_numeric' in filtered_df.columns:
        avg_price_brand = filtered_df.groupby('brand')['price_numeric'].mean().reset_index().sort_values('price_numeric', ascending=False)
        fig_price_brand = px.bar(avg_price_brand, x='brand', y='price_numeric', title="Average Price by Brand (INR)", color='brand', color_discrete_sequence=px.colors.qualitative.Pastel)
        fig_price_brand.update_traces(text=avg_price_brand['price_numeric'].round(2), textposition='auto', textfont=dict(size=12))
        st.plotly_chart(fig_price_brand)
        st.markdown("**Description**: This bar chart compares average prices by brand with pastel colors, including text labels for precise values.")
    if 'price_numeric' in filtered_df.columns and 'rating' in filtered_df.columns:
        fig_rating_price = px.scatter(filtered_df, x='price_numeric', y='rating', color='brand' if 'brand' in filtered_df.columns else None, title="Rating vs. Price (INR)", hover_data=['model'], color_discrete_sequence=px.colors.qualitative.Set3)
        st.plotly_chart(fig_rating_price)
        st.markdown("**Description**: This scatter plot uses a diverse color palette to show the relationship between price and rating, colored by brand.")
    if 'charging_type' in filtered_df.columns and 'battery_watt' in filtered_df.columns:
        fig_battery_type = px.box(filtered_df, x='charging_type', y='battery_watt', title="Battery Wattage by Charging Type", color='charging_type', color_discrete_sequence=px.colors.qualitative.D3)
        st.plotly_chart(fig_battery_type)
        st.markdown("**Description**: This box plot compares battery wattage across charging types with a colorful D3 palette, showing medians and outliers.")
    if 'ppi' in filtered_df.columns and 'price_numeric' in filtered_df.columns:
        fig_ppi_price = px.scatter(filtered_df, x='ppi', y='price_numeric', color='brand' if 'brand' in filtered_df.columns else None, title="PPI vs. Price (INR)", hover_data=['model'], color_discrete_sequence=px.colors.qualitative.Set2)
        st.plotly_chart(fig_ppi_price)
        st.markdown("**Description**: This scatter plot analyzes PPI vs. price with a Set2 color scheme, offering a clear view of display quality vs. cost.")
    if 'brand' in filtered_df.columns and '5G' in filtered_df.columns:
        five_g_brand = filtered_df.groupby('brand')['5G'].mean().reset_index().sort_values('5G', ascending=False)
        five_g_brand['5G'] = five_g_brand['5G'] * 100
        fig_5g = px.bar(five_g_brand, x='brand', y='5G', title="Percentage of 5G Support by Brand", color='brand', color_discrete_sequence=px.colors.qualitative.Safe)
        fig_5g.update_traces(text=five_g_brand['5G'].round(1), textposition='auto', textfont=dict(size=12))
        st.plotly_chart(fig_5g)
        st.markdown("**Description**: This bar chart displays 5G support percentages by brand with a safe color palette, including text labels for accuracy.")
    if 'brand' in filtered_df.columns:
        brand_counts = filtered_df['brand'].value_counts().reset_index()
        brand_counts.columns = ['brand', 'count']
        fig_brand_pie = px.pie(brand_counts, names='brand', values='count', title="Distribution of Phone Brands", color_discrete_sequence=px.colors.qualitative.Prism)
        fig_brand_pie.update_traces(textinfo='percent+label', textfont_size=14, textposition='outside')
        fig_brand_pie.update_layout(
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5
            ),
            width=700,
            height=600,
            showlegend=True
        )
        st.plotly_chart(fig_brand_pie)
        st.markdown("**Description**: This clear pie chart shows the distribution of phone brands with a Prism color palette, percentage labels outside for readability, and a horizontal legend.")
    st.subheader("Insights")
    st.markdown("""
    - Premium brands like Apple and Samsung tend to have higher average prices, often exceeding ₹50,000.
    - The rating vs. price scatter plot may reveal that higher ratings are not always tied to higher prices, suggesting value options.
    - Fast charging types (e.g., >30W) show higher median battery wattage, indicating technological advancement.
    - PPI increases with price, especially for brands like OnePlus and Xiaomi, reflecting better display quality in mid-range models.
    - Brands with higher 5G support (e.g., Samsung, OnePlus) are likely targeting newer markets.
    - The brand distribution pie chart may show Samsung and Xiaomi dominating, with smaller brands contributing less than 5% each.
    """)

with tab4:
    st.header("Correlations")
    if not filtered_df.empty:
        corr_cols = [col for col in ['rating', 'price_numeric', 'battery_watt', 'display_size', 'display_frequency', 'ppi', 'screen_area_in2', 'battery_per_gb_ram', 'watt_per_mah'] if col in filtered_df.columns and filtered_df[col].notna().any()]
        if corr_cols:
            corr_df = filtered_df[corr_cols].corr()
            fig_corr, ax = plt.subplots(figsize=(12, 10))
            sns.heatmap(corr_df, annot=True, cmap='RdYlBu', ax=ax, vmin=-1, vmax=1, annot_kws={"size": 10})
            plt.title("Correlation Heatmap of Numeric Features", fontsize=14)
            st.pyplot(fig_corr)
        else:
            st.warning("No numeric columns available for correlation analysis.")
    st.subheader("Insights")
    st.markdown("""
    - A positive correlation between price_numeric and ppi suggests that higher-priced phones offer better display quality.
    - Battery_watt may correlate with watt_per_mah, indicating efficient charging in high-wattage models.
    - Display_size and screen_area_in2 are likely strongly correlated, as they are derived from similar measurements.
    - Rating might show a weak correlation with price_numeric, implying that user satisfaction is not solely price-dependent.
    - Negative correlations could exist between battery_per_gb_ram and price_numeric, suggesting efficiency trade-offs in premium models.
    """)

with tab5:
    st.header("Outlier Detection")
    if 'price_numeric' in filtered_df.columns:
        Q1_price = filtered_df['price_numeric'].quantile(0.25)
        Q3_price = filtered_df['price_numeric'].quantile(0.75)
        IQR_price = Q3_price - Q1_price
        outliers_price = filtered_df[(filtered_df['price_numeric'] < (Q1_price - 1.5 * IQR_price)) | (filtered_df['price_numeric'] > (Q3_price + 1.5 * IQR_price))]
        st.subheader("Outliers in Price")
        st.dataframe(outliers_price[['model', 'rating', 'brand', 'price_numeric'] if 'brand' in outliers_price.columns else ['model', 'price_numeric']])
        fig_outlier_price = px.box(filtered_df, y='price_numeric', title="Box Plot of Prices (Outliers Highlighted)", color_discrete_sequence=['#36A2EB'])
        st.plotly_chart(fig_outlier_price)
        st.markdown("**Description**: This box plot identifies price outliers with a blue shade, useful for spotting unusual price points.")
    st.subheader("Insights")
    st.markdown("""
    - Rating outliers are rare, often occurring in niche models with unusually low or high ratings, possibly due to limited reviews.
    - Price outliers are likely found in luxury models (e.g., >₹100,000) or extremely budget options (<₹10,000), reflecting market extremes.
    - The presence of outliers suggests a diverse dataset, with some brands pushing boundaries in pricing or performance.
    """)

# Footer
st.markdown("---")
st.markdown(f"Last Updated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')} IST. Built with Streamlit.")

# Run the app
if __name__ == "__main__":
    if filtered_df.empty:
        st.warning("No data available after applying filters. Please adjust the filters.")