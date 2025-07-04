from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

st.set_page_config(page_title="simply-estimate", page_icon=":rocket:", layout="wide")

# Initialize session state for tasks
if "tasks" not in st.session_state:
    st.session_state.tasks = []


def calculate_pert_estimates(optimistic, pessimistic, most_likely):
    """Calculate PERT estimates using the formula: (O + 4M + P) / 6"""
    expected_time = (optimistic + 4 * most_likely + pessimistic) / 6
    variance = ((pessimistic - optimistic) / 6) ** 2
    std_dev = np.sqrt(variance)
    return expected_time, variance, std_dev


def find_critical_path(tasks_df):
    """Find the critical path using network analysis"""
    if tasks_df.empty:
        return [], {}, 0

    # Create a directed graph
    G = nx.DiGraph()

    # Add nodes with expected times
    for _, task in tasks_df.iterrows():
        expected_time, _, _ = calculate_pert_estimates(
            task["min_time"], task["max_time"], task["most_likely_time"]
        )
        G.add_node(task["name"], duration=expected_time)

    # Add edges based on dependencies
    for _, task in tasks_df.iterrows():
        if task["dependencies"]:
            deps = [
                dep.strip() for dep in task["dependencies"].split(",") if dep.strip()
            ]
            for dep in deps:
                if dep in [t["name"] for _, t in tasks_df.iterrows()]:
                    G.add_edge(dep, task["name"])

    try:
        # Calculate earliest start and finish times
        earliest_start = {}
        earliest_finish = {}

        # Topological sort to process nodes in correct order
        for node in nx.topological_sort(G):
            duration = G.nodes[node]["duration"]

            # Find maximum earliest finish of predecessors
            predecessors = list(G.predecessors(node))
            if not predecessors:
                earliest_start[node] = 0
            else:
                earliest_start[node] = max(
                    earliest_finish[pred] for pred in predecessors
                )

            earliest_finish[node] = earliest_start[node] + duration

        # Calculate latest start and finish times (backward pass)
        latest_start = {}
        latest_finish = {}

        # Project completion time
        project_duration = max(earliest_finish.values()) if earliest_finish else 0

        # Initialize latest finish times
        for node in G.nodes():
            successors = list(G.successors(node))
            if not successors:  # End nodes
                latest_finish[node] = earliest_finish[node]

        # Work backwards
        for node in reversed(list(nx.topological_sort(G))):
            duration = G.nodes[node]["duration"]
            successors = list(G.successors(node))

            if not successors:
                latest_finish[node] = earliest_finish[node]
            else:
                latest_finish[node] = min(latest_start[succ] for succ in successors)

            latest_start[node] = latest_finish[node] - duration

        # Find critical path (nodes with zero slack)
        critical_nodes = []
        slack_times = {}
        for node in G.nodes():
            slack = latest_start[node] - earliest_start[node]
            slack_times[node] = slack
            if abs(slack) < 0.001:  # Account for floating point precision
                critical_nodes.append(node)

        return critical_nodes, slack_times, project_duration

    except:
        return [], {}, 0


def create_gantt_chart(tasks_df, critical_path):
    """Create a Gantt chart visualization"""
    if tasks_df.empty:
        return go.Figure()

    fig = go.Figure()

    for i, (_, task) in enumerate(tasks_df.iterrows()):
        expected_time, _, _ = calculate_pert_estimates(
            task["min_time"], task["max_time"], task["most_likely_time"]
        )

        color = "red" if task["name"] in critical_path else "blue"

        fig.add_trace(
            go.Bar(
                y=[task["name"]],
                x=[expected_time],
                orientation="h",
                name=task["name"],
                marker_color=color,
                text=f"{expected_time:.1f} days",
                textposition="outside",
            )
        )

    fig.update_layout(
        title="Project Timeline (Gantt Chart)",
        xaxis_title="Duration (Days)",
        yaxis_title="Tasks",
        showlegend=False,
        height=max(400, len(tasks_df) * 40),
    )

    return fig


def create_network_graph(tasks_df):
    """Create a network graph showing task dependencies"""
    if tasks_df.empty:
        return go.Figure()

    G = nx.DiGraph()

    # Add nodes
    for _, task in tasks_df.iterrows():
        expected_time, _, _ = calculate_pert_estimates(
            task["min_time"], task["max_time"], task["most_likely_time"]
        )
        G.add_node(task["name"], duration=expected_time)

    # Add edges
    for _, task in tasks_df.iterrows():
        if task["dependencies"]:
            deps = [
                dep.strip() for dep in task["dependencies"].split(",") if dep.strip()
            ]
            for dep in deps:
                if dep in [t["name"] for _, t in tasks_df.iterrows()]:
                    G.add_edge(dep, task["name"])

    # Create layout
    pos = nx.spring_layout(G, k=3, iterations=50)

    # Create edge traces
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=2, color="gray"),
        hoverinfo="none",
        mode="lines",
    )

    # Create node traces
    node_x = []
    node_y = []
    node_text = []
    node_info = []

    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)
        duration = G.nodes[node]["duration"]
        node_info.append(f"{node}<br>Duration: {duration:.1f} days")

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        hoverinfo="text",
        text=node_text,
        hovertext=node_info,
        textposition="middle center",
        marker=dict(size=50, color="lightblue", line=dict(width=2, color="black")),
    )

    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title=dict(text="Task Dependency Network", font=dict(size=16)),
            showlegend=False,
            hovermode="closest",
            margin=dict(b=20, l=5, r=5, t=40),
            annotations=[
                dict(
                    text="Hover over nodes for details",
                    showarrow=False,
                    xref="paper",
                    yref="paper",
                    x=0.005,
                    y=-0.002,
                    xanchor="left",
                    yanchor="bottom",
                    font=dict(color="gray", size=12),
                )
            ],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        ),
    )

    return fig


# Main app
st.title("🛀 Simply Estimate")
st.markdown("Plan your projects with PERT analysis and critical path identification")

# Section 1: Task Input
st.header("📝 Section 1: Task Management")

with st.expander("➕ Add New Task", expanded=True):
    col1, col2 = st.columns(2)

    with col1:
        task_name = st.text_input("Task Name*", key="task_name")
        task_owner = st.text_input("Task Owner*", key="task_owner")
        task_description = st.text_area("Task Description", key="task_description")
        task_dod = st.text_area("Definition of Done", key="task_dod")

    with col2:
        min_time = st.number_input(
            "Optimistic Time (days)*",
            min_value=0.1,
            value=1.0,
            step=0.1,
            key="min_time",
        )
        max_time = st.number_input(
            "Pessimistic Time (days)*",
            min_value=0.1,
            value=5.0,
            step=0.1,
            key="max_time",
        )
        most_likely_time = st.number_input(
            "Most Likely Time (days)*",
            min_value=0.1,
            value=3.0,
            step=0.1,
            key="most_likely_time",
        )
        dependencies = st.text_input(
            "Dependencies (comma-separated task names)", key="dependencies"
        )

    if st.button("Add Task", type="primary"):
        if task_name and task_owner and min_time and max_time and most_likely_time:
            if max_time >= most_likely_time >= min_time:
                new_task = {
                    "name": task_name,
                    "owner": task_owner,
                    "description": task_description,
                    "definition_of_done": task_dod,
                    "min_time": min_time,
                    "max_time": max_time,
                    "most_likely_time": most_likely_time,
                    "dependencies": dependencies,
                }
                st.session_state.tasks.append(new_task)
                st.success(f"Task '{task_name}' added successfully!")
                st.rerun()
            else:
                st.error("Please ensure: Optimistic ≤ Most Likely ≤ Pessimistic time")
        else:
            st.error("Please fill in all required fields (*)")

# Display existing tasks
if st.session_state.tasks:
    st.subheader("📋 Current Tasks")

    tasks_df = pd.DataFrame(st.session_state.tasks)

    # Add PERT calculations to the display
    for i, task in enumerate(st.session_state.tasks):
        expected_time, variance, std_dev = calculate_pert_estimates(
            task["min_time"], task["max_time"], task["most_likely_time"]
        )

        with st.expander(
            f"**{task['name']}** (Owner: {task['owner']}) - Expected: {expected_time:.1f} days"
        ):
            col1, col2, col3 = st.columns(3)

            with col1:
                st.write(f"**Description:** {task['description']}")
                st.write(f"**Definition of Done:** {task['definition_of_done']}")

            with col2:
                st.write(f"**Time Estimates:**")
                st.write(f"- Optimistic: {task['min_time']} days")
                st.write(f"- Most Likely: {task['most_likely_time']} days")
                st.write(f"- Pessimistic: {task['max_time']} days")
                st.write(f"- **Expected: {expected_time:.1f} days**")

            with col3:
                st.write(
                    f"**Dependencies:** {task['dependencies'] if task['dependencies'] else 'None'}"
                )
                st.write(f"**Standard Deviation:** {std_dev:.1f} days")

                if st.button(f"Delete", key=f"delete_{i}"):
                    st.session_state.tasks.pop(i)
                    st.rerun()

    # Section 2: Analysis
    st.header("📊 Section 2: Project Analysis")

    if len(st.session_state.tasks) > 0:
        critical_path, slack_times, project_duration = find_critical_path(tasks_df)

        # Project Summary
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Total Tasks", len(st.session_state.tasks))

        with col2:
            st.metric("Project Duration", f"{project_duration:.1f} days")

        with col3:
            st.metric("Critical Path Length", len(critical_path))

        # Critical Path Analysis
        st.subheader("🎯 Critical Path Analysis")

        if critical_path:
            st.success(f"**Critical Path:** {' → '.join(critical_path)}")
            st.write(
                "Tasks on the critical path have zero slack time and determine the project duration."
            )

            # Slack times table
            st.subheader("⏱️ Task Slack Times")
            slack_df = pd.DataFrame(
                [
                    {
                        "Task": task,
                        "Slack Time (days)": f"{slack:.1f}",
                        "Status": "Critical" if slack < 0.001 else "Non-Critical",
                    }
                    for task, slack in slack_times.items()
                ]
            )
            st.dataframe(slack_df, use_container_width=True)
        else:
            st.warning(
                "Could not determine critical path. Please check task dependencies."
            )

        # Visualizations
        st.subheader("📈 Project Visualizations")

        tab1, tab2 = st.tabs(["Gantt Chart", "Dependency Network"])

        with tab1:
            gantt_fig = create_gantt_chart(tasks_df, critical_path)
            st.plotly_chart(gantt_fig, use_container_width=True)
            st.caption(
                "Red bars indicate critical path tasks, blue bars indicate non-critical tasks."
            )

        with tab2:
            network_fig = create_network_graph(tasks_df)
            st.plotly_chart(network_fig, use_container_width=True)
            st.caption("Network diagram showing task dependencies and relationships.")

        # PERT Analysis Summary
        st.subheader("📊 PERT Analysis Summary")

        total_expected = sum(
            [
                calculate_pert_estimates(
                    task["min_time"], task["max_time"], task["most_likely_time"]
                )[0]
                for task in st.session_state.tasks
            ]
        )

        total_variance = sum(
            [
                calculate_pert_estimates(
                    task["min_time"], task["max_time"], task["most_likely_time"]
                )[1]
                for task in st.session_state.tasks
                if task["name"] in critical_path
            ]
        )

        project_std_dev = np.sqrt(total_variance)

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Sum of All Task Times", f"{total_expected:.1f} days")
            st.metric("Critical Path Duration", f"{project_duration:.1f} days")

        with col2:
            st.metric("Project Standard Deviation", f"{project_std_dev:.1f} days")
            st.metric(
                "68% Confidence Range",
                f"{project_duration - project_std_dev:.1f} - {project_duration + project_std_dev:.1f} days",
            )

        # Export functionality
        st.subheader("💾 Export Data")

        if st.button("Download Project Data as CSV"):
            # Create comprehensive export data
            export_data = []
            for task in st.session_state.tasks:
                expected_time, variance, std_dev = calculate_pert_estimates(
                    task["min_time"], task["max_time"], task["most_likely_time"]
                )
                slack = slack_times.get(task["name"], 0)
                is_critical = task["name"] in critical_path

                export_data.append(
                    {
                        "Task Name": task["name"],
                        "Owner": task["owner"],
                        "Description": task["description"],
                        "Definition of Done": task["definition_of_done"],
                        "Optimistic Time": task["min_time"],
                        "Most Likely Time": task["most_likely_time"],
                        "Pessimistic Time": task["max_time"],
                        "Expected Time": expected_time,
                        "Standard Deviation": std_dev,
                        "Dependencies": task["dependencies"],
                        "Slack Time": slack,
                        "Is Critical": is_critical,
                    }
                )

            export_df = pd.DataFrame(export_data)
            csv = export_df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"project_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
            )
    else:
        st.info("Add some tasks above to see the project analysis!")

else:
    st.info("👆 Start by adding your first task above!")

# Clear all tasks button
if st.session_state.tasks:
    if st.button("🗑️ Clear All Tasks", type="secondary"):
        st.session_state.tasks = []
        st.rerun()
        st.rerun()
