import streamlit as st
import pandas as pd

import plotly.express as px
import plotly.graph_objects as go

from sklearn.linear_model import LinearRegression

ROOT_PATH = 'data'


df = pd.read_csv(f'{ROOT_PATH}/forecast.csv')
df["Date"] = pd.to_datetime(df["Date"]).dt.date

df_cov = pd.read_csv(f'{ROOT_PATH}/df_cov.csv')
df_cov["Date"] = pd.to_datetime(df_cov["Date"])
df_cov.set_index("Date", inplace=True)

target = "Median Rent Price growth1"
features = ["Unemployment rate diff1", "NHPI growth1", "LNLR", ] 


# Set the title and favicon that appear in the Browser's tab bar.
st.set_page_config(
    page_title='Canadian property analytics',
    page_icon=':flag_ca:', # This is an emoji shortcode. Could be a URL too.
)


# -----------------------------------------------------------------------------
# Draw the actual page

# Set the title that appears at the top of the page.
'''
# Canadian housing market analytics

The Canadian housing market has been a topic of much discussion, particularly 
in recent years due to fluctuating economic conditions. This page focuses 
on forecasting various real estate trends such as Toronto's median rent prices 
and the MLS Composite Home Price Benchmark in Canada, which represents the price 
trajectory of a typical home in the country.
'''

# Add some spacing
''
''

st.header('Median Rent Price in Toronto', divider='gray')

''

min_value = df['Date'].min()
max_value = df['Date'].max()

# print(min_value, max_value)

from_year, to_year = st.slider(
    'Which dates are you interested in?',
    min_value=min_value,
    max_value=max_value,
    value=[df['Date'].iloc[-36], max_value])

# Filter the data
filtered_df = df[
    (df['Date'] <= to_year)
    & (from_year <= df['Date'])
]

st.line_chart(
    filtered_df,
    x='Date',
    y='Median Rent Price',
    color='Data Type',
)

st.dataframe(filtered_df.tail(12))



st.header("Rent Price and Leading Indicators", divider="gray")

# countries = gdp_df['Country Code'].unique()

# if not len(countries):
#     st.warning("Select at least one country")


selected_feature = st.selectbox("Choose a feature:", features)

# Lag Selection
max_lag = max(
    int(col.split("lag")[1].strip()) if "lag" in col else 0
    for col in df_cov.columns if selected_feature in col
)
selected_lag = st.slider("Choose a lag value:", 0, max_lag, 0)


def plot_line_chart(df, feature, lag, target):
    col_name = feature if lag == 0 else f"{feature} lag {lag}"
    
    if col_name not in df.columns:
        st.warning(f"Column '{col_name}' not found in the dataset.")
        return go.Figure()
    
    # Plotly Line Chart
    fig = go.Figure()

    # Add Median Rent Price Line
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df[target],
            mode="lines",
            name=target,
            line=dict(color="blue"),
        )
    )

    # Add Unemployment Rate Line
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df[col_name],
            mode="lines",
            name=col_name,
            line=dict(color="black"),
            yaxis="y2",  # Link to secondary y-axis
        )
    )

    # Layout for Dual Axes
    fig.update_layout(
        title= target + " and " + col_name,
        xaxis=dict(title="Date"),
        yaxis=dict(
            title=target,
            titlefont=dict(color="blue"),
            tickfont=dict(color="blue"),
        ),
        yaxis2=dict(
            title=feature,
            titlefont=dict(color="black"),
            tickfont=dict(color="black"),
            overlaying="y",  # Overlay on the first y-axis
            side="right",  # Position on the right
            showgrid=False,
        ),
        legend=dict(
        orientation="h",  # Horizontal legend
        yanchor="top",
        y=-0.3,  # Position below the x-axis
        xanchor="left",
        x=0,
        ),
        # legend=dict(x=0.1, y=1),
    )

    # Display Plotly Chart in Streamlit
    st.plotly_chart(fig, use_container_width=True)

plot_line_chart(df_cov, selected_feature, selected_lag, target)

# Define the function using Plotly
def plot_target_covariates(df, selected_feature, selected_lag, target):
    col_name = selected_feature if selected_lag == 0 else f"{selected_feature} lag {selected_lag}"
    
    if col_name not in df.columns:
        st.warning(f"Column '{col_name}' not found in the dataset.")
        return go.Figure()
    
    # Extract the data for the selected feature and target
    x = df[col_name].values.reshape(-1, 1)  # Feature values
    y = df[target].values  # Target values

    # Create scatter plot for the selected feature and lag
    fig = px.scatter(
        df,
        x=col_name,
        y=target,
        # title=f"Scatter Plot: {col_name} vs {target}",
        labels={col_name: col_name, target: target},
    )
    
    # Perform Linear Regression to get the regression line
    model = LinearRegression()
    model.fit(x, y)
    y_pred = model.predict(x)

    # Add regression line to the plot
    fig.add_trace(
        go.Scatter(
            x=df[col_name],
            y=y_pred,
            mode='lines',
            name='Regression Line',
            line=dict(color='red', width=2)
        )
    )

    # Customize layout
    fig.update_layout(
        template="plotly_white",
        height=600,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.2,
            xanchor="left",
            x=0.5,
        ),
    )
    return fig
# Sidebar Controls
# selected_feature2 = st.selectbox("Choose a feature:", features)


# Generate Plot
fig = plot_target_covariates(df_cov, selected_feature, selected_lag, target)

# Display Plot
st.plotly_chart(fig, use_container_width=True)
