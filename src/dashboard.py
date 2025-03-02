import streamlit as st
import pandas as pd

import plotly.express as px
import plotly.graph_objects as go

from sklearn.linear_model import LinearRegression


def load_forecast(file_path):
    # Load the data
    df = pd.read_csv(file_path)
    df["Date"] = pd.to_datetime(df["Date"]).dt.date
    return df


def show_forecast(df, target):
    min_date = df["Date"].min()
    max_date = df["Date"].max()

    from_date, to_date = st.slider(
        "Which dates are you interested in?",
        min_value=min_date,
        max_value=max_date,
        value=[df["Date"].iloc[-36], max_date],
        key=target,
    )

    # Filter the data
    filtered_df = df[(from_date <= df["Date"]) & (df["Date"] <= to_date)].copy()
    final_chart = plot_forecast(filtered_df, target)
    # Show Chart in Streamlit
    st.plotly_chart(final_chart, use_container_width=True)


def plot_forecast(filtered_df, target):
    # Calculate Dynamic Y-Axis Limits
    y_min = filtered_df[target].min() * 0.95  # Add margin
    y_max = filtered_df[target].max() * 1.05  # Add margin

    # Create figure
    fig = go.Figure()

    # Add main line trace
    fig.add_trace(
        go.Scatter(
            x=filtered_df["Date"],
            y=filtered_df[target],
            mode="lines",
            name=target,
            line=dict(color="#1f77b4"),
        )
    )

    # Add forecast background using Shapes
    forecast_df = filtered_df[filtered_df["Data Type"] == "Forecast"]
    if not forecast_df.empty:
        start_date = forecast_df["Date"].min()
        end_date = filtered_df["Date"].max()

        fig.add_shape(
            type="rect",
            x0=start_date,
            x1=end_date,
            y0=y_min,
            y1=y_max,
            fillcolor="rgba(169, 169, 169, 0.3)",  # Grey background
            line=dict(width=0),
        )

        # Add forecast label in the middle
        mid_date = start_date + (end_date - start_date) / 2
        fig.add_annotation(
            x=mid_date,
            y=y_max * 0.99,
            text="Forecast",
            showarrow=False,
            # font=dict(size=14, color="#000000", family="Arial"),
            xanchor="center",
            yanchor="middle",
        )

    # Update layout
    fig.update_layout(
        yaxis_title=target,
        yaxis=dict(range=[y_min, y_max], tickformat="$,.0f"),
        template="plotly_white",
    )

    return fig


def get_line_fig(df, col_name, target):

    # Plotly Line Chart
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df[target],
            mode="lines",
            name=target,
            line=dict(color="#3B68C5"),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df[col_name],
            mode="lines",
            name=col_name,
            yaxis="y2",  # Link to secondary y-axis
        )
    )

    # Layout for Dual Axes
    fig.update_layout(
        title=target + " and " + col_name,
        # xaxis=dict(title="Date"),
        yaxis=dict(
            title=target,
            # titlefont=dict(color="blue"),
            # tickfont=dict(color="blue"),
        ),
        yaxis2=dict(
            title=col_name,
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
    return fig


def plot_line_chart(df, feature, lag, target):
    col_name = feature if lag == 0 else f"{feature} lag {lag}"

    if col_name not in df.columns:
        st.warning(f"Column '{col_name}' not found in the dataset.")
        return go.Figure()
    fig = get_line_fig(df, col_name, target)
    # Display Plotly Chart in Streamlit
    st.plotly_chart(fig, use_container_width=True)


# Define the function using Plotly
def plot_target_covariates(df, selected_feature, selected_lag, target):
    col_name = (
        selected_feature
        if selected_lag == 0
        else f"{selected_feature} lag {selected_lag}"
    )

    if col_name not in df.columns:
        st.warning(f"Column '{col_name}' not found in the dataset.")
        return go.Figure()
    fig = get_scatter_fig(df, col_name, target)
    # Display Plot
    st.plotly_chart(fig, use_container_width=True)


def get_scatter_fig(df, col_name, target):
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
            mode="lines",
            name="Regression Line",
            line=dict(color="red", width=2),
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
    return fig


def add_slider(df, col_name, target):
    max_lag = max(
        int(col.split("lag")[1].strip()) if "lag" in col else 0
        for col in df.columns
        if col_name in col
    )
    # Create figure
    fig = go.Figure()

    # Add traces for each lag
    for lag in range(max_lag + 1):
        lag_col = f"{col_name} lag {lag}" if lag > 0 else col_name

        # Generate the scatter plot with regression line
        scatter_fig = get_scatter_fig(df, lag_col, target)

        # Extract traces and add them to the main figure
        for trace in scatter_fig.data:
            trace.visible = True if lag == 0 else False  # Show only lag 0 initially
            fig.add_trace(trace)

    # Define slider steps
    steps = []
    for i, lag in enumerate(range(max_lag + 1)):
        step = {
            "method": "update",
            "args": [
                {
                    "visible": [j // 2 == i for j in range(2 * (max_lag + 1))]
                },  # Toggle traces
                {"title": f"Scatter Plot: {col_name} vs {target} (Lag {lag})"},
            ],
            "label": f"{lag}",
        }
        steps.append(step)

    # Add slider to layout
    fig.update_layout(
        sliders=[
            {
                "active": 0,
                "currentvalue": {"prefix": "Selected Lag: "},
                "pad": {"t": 50},
                "steps": steps,
            }
        ],
        title=f"Scatter Plot with Lags for {col_name}",
    )

    return fig
