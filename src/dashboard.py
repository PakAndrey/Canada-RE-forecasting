import streamlit as st
import pandas as pd
import altair as alt

import plotly.express as px
import plotly.graph_objects as go

from sklearn.linear_model import LinearRegression

def load_forecast(file_path):
    # Load the data
    df = pd.read_csv(file_path)
    df["Date"] = pd.to_datetime(df["Date"]).dt.date
    return df

def show_forecast(df, target):
    min_date = df['Date'].min()
    max_date = df['Date'].max()

    from_date, to_date = st.slider(
        'Which dates are you interested in?',
        min_value=min_date,
        max_value=max_date,
        value=[df['Date'].iloc[-36], max_date])

    # Filter the data
    filtered_df = df[(from_date <= df['Date']) & (df['Date'] <= to_date)].copy()

    # **Calculate Dynamic Y-Axis Limits**
    y_min = filtered_df[target].min() * 0.95  # Add margin
    y_max = filtered_df[target].max() * 1.05  # Add margin

    # **Base Line Chart with Forecast Legend**
    line_chart = alt.Chart(filtered_df).mark_line().encode(
        x=alt.X('Date:T', title="Date"),
        y=alt.Y(f'{target}:Q', title=target, scale=alt.Scale(domain=[y_min, y_max])),  # **Dynamic y-axis**
        color=alt.Color('Data Type:N', legend=alt.Legend(title="Legend",orient="bottom")),  # **Legend for Forecast**
        strokeDash=alt.condition(
            alt.datum["Data Type"] == "Forecast",  # Apply dashes only to Forecast
            alt.value([0]),  # Dashed line
            alt.value([0])  # Solid line
        )
    )

    # **Hover Effect**
    nearest = alt.selection_single(
        fields=['Date'], 
        nearest=True,  
        on='mouseover',
        empty='none'
    )

    hover_circles = alt.Chart(filtered_df).mark_circle(size=80).encode(
        x='Date:T',
        y=alt.Y(f'{target}:Q'),
        tooltip=['Date:T', f'{target}:Q'],
        opacity=alt.condition(nearest, alt.value(1), alt.value(0))
    ).add_selection(nearest)  

    # **Combine Layers**
    final_chart = alt.layer(line_chart, hover_circles)

    # Show Chart in Streamlit
    st.altair_chart(final_chart, use_container_width=True)

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
            # line=dict(color="blue"),
        )
    )

    # Add Unemployment Rate Line
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df[col_name],
            mode="lines",
            name=col_name,
            # line=dict(color="black"),
            yaxis="y2",  # Link to secondary y-axis
        )
    )

    # Layout for Dual Axes
    fig.update_layout(
        title= target + " and " + col_name,
        xaxis=dict(title="Date"),
        yaxis=dict(
            title=target,
            # titlefont=dict(color="blue"),
            # tickfont=dict(color="blue"),
        ),
        yaxis2=dict(
            title=feature,
            # titlefont=dict(color="black"),
            # tickfont=dict(color="black"),
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
        # height=600,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.2,
            xanchor="center",
            x=0.5,
        ),
    )

    # Display Plot
    st.plotly_chart(fig, use_container_width=True)
