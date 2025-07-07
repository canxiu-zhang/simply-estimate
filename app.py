import math
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# Set page config
st.set_page_config(page_title="Simply Estimate", page_icon="📊", layout="wide")


def parse_dependencies(dependency_str):
    """Parse dependency string and return list of task IDs"""
    if (
        pd.isna(dependency_str)
        or dependency_str == ""
        or str(dependency_str).strip() == ""
    ):
        return []
    # Handle both string and numeric task IDs, ensure they're converted to strings
    deps = [
        str(dep).strip() for dep in str(dependency_str).split(",") if str(dep).strip()
    ]
    return deps


def calculate_pert_estimates(optimistic, nominal, pessimistic):
    """Calculate PERT estimates (Expected time and Standard Deviation)"""
    expected = (optimistic + 4 * nominal + pessimistic) / 6
    std_dev = (pessimistic - optimistic) / 6
    return expected, std_dev


def topological_sort(tasks_df):
    """Perform topological sort to determine task execution order"""
    # Create a directed graph
    G = nx.DiGraph()

    # Convert Task ID column to string to ensure consistency
    tasks_df["Task ID"] = tasks_df["Task ID"].astype(str)

    # Add all tasks as nodes
    for task_id in tasks_df["Task ID"]:
        G.add_node(str(task_id))

    # Add edges for dependencies
    for _, row in tasks_df.iterrows():
        dependencies = parse_dependencies(row["Dependency"])
        for dep in dependencies:
            dep_str = str(dep).strip()
            task_id_str = str(row["Task ID"]).strip()
            if dep_str in tasks_df["Task ID"].values:
                G.add_edge(dep_str, task_id_str)

    # Check for cycles
    if not nx.is_directed_acyclic_graph(G):
        st.error("Circular dependencies detected! Please check your task dependencies.")
        return None, None

    # Get topological order
    try:
        topo_order = list(nx.topological_sort(G))
        return topo_order, G
    except nx.NetworkXError:
        st.error("Error in dependency analysis. Please check your task dependencies.")
        return None, None


def calculate_project_timeline(tasks_df, topo_order):
    """Calculate project timeline considering dependencies and resource constraints"""
    # Create timeline dictionary
    timeline = {}
    resource_schedule = {}  # Track when each person is available

    # Convert Task ID column to string to ensure consistency
    tasks_df["Task ID"] = tasks_df["Task ID"].astype(str)

    for task_id in topo_order:
        task = tasks_df[tasks_df["Task ID"] == str(task_id)].iloc[0]

        # Get dependencies
        dependencies = parse_dependencies(task["Dependency"])

        # Calculate earliest start time based on dependencies
        earliest_start = 0
        for dep in dependencies:
            dep_str = str(dep).strip()
            if dep_str in timeline:
                earliest_start = max(earliest_start, timeline[dep_str]["end_time"])

        # Check resource availability
        owner = task["Owner"]
        if owner in resource_schedule:
            earliest_start = max(earliest_start, resource_schedule[owner])

        # Calculate task duration (using expected time)
        duration = task["Expected"]

        # Set start and end times
        start_time = earliest_start
        end_time = start_time + duration

        timeline[str(task_id)] = {
            "task": task["Task"],
            "owner": owner,
            "start_time": start_time,
            "end_time": end_time,
            "duration": duration,
            "optimistic": task["Optimistic"],
            "nominal": task["Nominal"],
            "pessimistic": task["Pessimistic"],
            "expected": task["Expected"],
            "std_dev": task["Standard Deviation"],
        }

        # Update resource schedule
        resource_schedule[owner] = end_time

    return timeline


def create_gantt_chart(timeline, project_name):
    """Create Gantt chart visualization"""
    # Prepare data for Gantt chart
    gantt_data = []
    base_date = datetime.now().date()  # Use current date as project start

    for task_id, info in timeline.items():
        start_date = base_date + timedelta(days=info["start_time"])
        end_date = base_date + timedelta(days=info["end_time"])

        gantt_data.append(
            {
                "Task": f"{task_id}: {info['task'][:30]}...",
                "Start": start_date,
                "Finish": end_date,
                "Owner": info["owner"],
                "Duration": info["duration"],
            }
        )

    gantt_df = pd.DataFrame(gantt_data)

    # Create Gantt chart
    fig = px.timeline(
        gantt_df,
        x_start="Start",
        x_end="Finish",
        y="Task",
        color="Owner",
        title=f"Project Timeline - {project_name}",
    )

    fig.update_layout(
        xaxis_title="Time (days)",
        yaxis_title="Tasks",
        height=max(400, len(gantt_data) * 30),
        showlegend=True,
    )

    return fig


def create_dependency_graph(tasks_df, G):
    """Create dependency graph visualization"""
    # Set up the plot
    fig, ax = plt.subplots(figsize=(12, 8))

    # Calculate layout
    pos = nx.spring_layout(G, k=3, iterations=50)

    # Draw nodes
    nx.draw_networkx_nodes(
        G, pos, node_color="lightblue", node_size=1500, alpha=0.7, ax=ax
    )

    # Draw edges
    nx.draw_networkx_edges(G, pos, edge_color="gray", arrows=True, arrowsize=20, ax=ax)

    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight="bold", ax=ax)

    ax.set_title("Task Dependencies Graph", fontsize=16, fontweight="bold")
    ax.axis("off")

    return fig


def analyze_workload(timeline):
    """Analyze workload distribution by owner"""
    workload_data = {}

    for task_id, info in timeline.items():
        owner = info["owner"]
        if owner not in workload_data:
            workload_data[owner] = {"total_tasks": 0, "total_duration": 0, "tasks": []}

        workload_data[owner]["total_tasks"] += 1
        workload_data[owner]["total_duration"] += info["duration"]
        workload_data[owner]["tasks"].append(
            {
                "task_id": task_id,
                "task": info["task"],
                "duration": info["duration"],
                "start_time": info["start_time"],
                "end_time": info["end_time"],
            }
        )

    return workload_data


def calculate_confidence_intervals(timeline):
    """Calculate confidence intervals for project completion"""
    total_variance = 0
    critical_path_tasks = []

    # For this simplified version, we'll consider all tasks as potentially on critical path
    for task_id, info in timeline.items():
        total_variance += info["std_dev"] ** 2
        critical_path_tasks.append(task_id)

    total_std_dev = math.sqrt(total_variance)

    # Calculate project completion time
    project_end = max(info["end_time"] for info in timeline.values())

    # Calculate confidence intervals
    confidence_intervals = {
        "1_sd": (project_end - total_std_dev, project_end + total_std_dev),
        "2_sd": (project_end - 2 * total_std_dev, project_end + 2 * total_std_dev),
        "3_sd": (project_end - 3 * total_std_dev, project_end + 3 * total_std_dev),
    }

    return confidence_intervals, project_end, total_std_dev


def main():
    st.title("📊 Simply Estimate - Project Planning Tool")
    st.markdown(
        "*Using PERT (Program Evaluation And Review Technique) and CPM (Critical Path Method)*"
    )

    # Sidebar for file upload
    st.sidebar.header("Upload Project Data")
    uploaded_file = st.sidebar.file_uploader(
        "Choose an Excel file", type=["xlsx", "xls"]
    )

    if uploaded_file is not None:
        try:
            # Read Excel file
            df = pd.read_excel(uploaded_file)

            # Validate required columns
            required_columns = [
                "Project",
                "Task ID",
                "Task",
                "Description",
                "DoD",
                "Dependency",
                "Owner",
                "Optimistic",
                "Nominal",
                "Pessimistic",
            ]

            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                st.error(f"Missing required columns: {missing_columns}")
                st.stop()

            # Calculate PERT estimates if not provided
            if "Expected" not in df.columns or "Standard Deviation" not in df.columns:
                df["Expected"], df["Standard Deviation"] = zip(
                    *df.apply(
                        lambda row: calculate_pert_estimates(
                            row["Optimistic"], row["Nominal"], row["Pessimistic"]
                        ),
                        axis=1,
                    )
                )

            # Calculate confidence intervals if not provided
            for col in ["1 sd", "2 sd", "3 sd"]:
                if col not in df.columns:
                    if col == "1 sd":
                        df[col] = df["Expected"] + df["Standard Deviation"]
                    elif col == "2 sd":
                        df[col] = df["Expected"] + 2 * df["Standard Deviation"]
                    elif col == "3 sd":
                        df[col] = df["Expected"] + 3 * df["Standard Deviation"]

            # Ensure proper data types for display
            df["Task ID"] = df["Task ID"].astype(str)
            df["Dependency"] = df["Dependency"].fillna("").astype(str)

            # Display raw data
            st.header("📋 Project Data Overview")
            st.dataframe(df)

            # Get unique projects
            projects = df["Project"].unique()

            for project in projects:
                st.header(f"🎯 Project: {project}")

                # Filter data for current project
                project_df = df[df["Project"] == project].copy()

                # Perform topological sort
                topo_order, G = topological_sort(project_df)
                if topo_order is None:
                    continue

                # Calculate project timeline
                timeline = calculate_project_timeline(project_df, topo_order)

                # Add debug information
                with st.expander("🔍 Debug Information"):
                    st.write("**Task Order (Topological Sort):**")
                    st.write(topo_order)

                    st.write("**Dependencies:**")
                    for _, row in project_df.iterrows():
                        deps = parse_dependencies(row["Dependency"])
                        if deps:
                            st.write(f"Task {row['Task ID']}: depends on {deps}")

                    st.write("**Timeline Calculation:**")
                    for task_id, info in timeline.items():
                        st.write(
                            f"Task {task_id}: Start={info['start_time']:.1f}, End={info['end_time']:.1f}, Duration={info['duration']:.1f}"
                        )

                # Create tabs for different views
                tab1, tab2, tab3, tab4 = st.tabs(
                    ["📊 Timeline", "🔗 Dependencies", "👥 Workload", "📈 Analysis"]
                )

                with tab1:
                    st.subheader("Gantt Chart")
                    gantt_fig = create_gantt_chart(timeline, project)
                    st.plotly_chart(gantt_fig, use_container_width=True)

                    # Project summary
                    project_end = max(info["end_time"] for info in timeline.values())
                    st.metric("Project Duration", f"{project_end:.1f} days")

                with tab2:
                    st.subheader("Task Dependencies")
                    if len(G.edges()) > 0:
                        dep_fig = create_dependency_graph(project_df, G)
                        st.pyplot(dep_fig)
                    else:
                        st.info("No dependencies found for this project.")

                with tab3:
                    st.subheader("Workload Distribution")
                    workload_data = analyze_workload(timeline)

                    # Create workload chart
                    workload_df = pd.DataFrame(
                        [
                            {
                                "Owner": owner,
                                "Total Tasks": data["total_tasks"],
                                "Total Duration": data["total_duration"],
                            }
                            for owner, data in workload_data.items()
                        ]
                    )

                    col1, col2 = st.columns(2)

                    with col1:
                        fig_tasks = px.bar(
                            workload_df,
                            x="Owner",
                            y="Total Tasks",
                            title="Number of Tasks per Person",
                        )
                        st.plotly_chart(fig_tasks, use_container_width=True)

                    with col2:
                        fig_duration = px.bar(
                            workload_df,
                            x="Owner",
                            y="Total Duration",
                            title="Total Duration per Person (days)",
                        )
                        st.plotly_chart(fig_duration, use_container_width=True)

                    # Detailed workload breakdown
                    st.subheader("Detailed Workload Breakdown")
                    for owner, data in workload_data.items():
                        with st.expander(
                            f"{owner} - {data['total_tasks']} tasks, {data['total_duration']:.1f} days"
                        ):
                            for task in data["tasks"]:
                                st.write(
                                    f"**{task['task_id']}**: {task['task']} ({task['duration']:.1f} days)"
                                )

                with tab4:
                    st.subheader("Project Analysis")

                    # Calculate confidence intervals
                    confidence_intervals, project_end, total_std_dev = (
                        calculate_confidence_intervals(timeline)
                    )

                    # Display metrics
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric("Expected Duration", f"{project_end:.1f} days")

                    with col2:
                        st.metric("Standard Deviation", f"{total_std_dev:.1f} days")

                    with col3:
                        st.metric("Total Tasks", len(timeline))

                    # Confidence intervals
                    st.subheader("Confidence Intervals")
                    ci_data = []
                    for level, (lower, upper) in confidence_intervals.items():
                        ci_data.append(
                            {
                                "Confidence Level": level.replace("_", " ").upper(),
                                "Lower Bound": f"{lower:.1f} days",
                                "Upper Bound": f"{upper:.1f} days",
                                "Range": f"{upper - lower:.1f} days",
                            }
                        )

                    ci_df = pd.DataFrame(ci_data)
                    st.dataframe(ci_df)

                    # Risk analysis
                    st.subheader("Risk Analysis")
                    st.write("**Interpretation:**")
                    st.write(
                        "- **68% confidence**: Project will complete between {:.1f} and {:.1f} days".format(
                            confidence_intervals["1_sd"][0],
                            confidence_intervals["1_sd"][1],
                        )
                    )
                    st.write(
                        "- **95% confidence**: Project will complete between {:.1f} and {:.1f} days".format(
                            confidence_intervals["2_sd"][0],
                            confidence_intervals["2_sd"][1],
                        )
                    )
                    st.write(
                        "- **99.7% confidence**: Project will complete between {:.1f} and {:.1f} days".format(
                            confidence_intervals["3_sd"][0],
                            confidence_intervals["3_sd"][1],
                        )
                    )

        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            st.info(
                "Please ensure your Excel file contains all required columns and is properly formatted."
            )

    else:
        st.info("Please upload an Excel file to get started.")

        # Show sample data format
        st.header("📝 Expected File Format")
        st.write("Your Excel file should contain the following columns:")

        sample_data = {
            "Project": ["Project Alpha", "Project Alpha", "Project Alpha"],
            "Task ID": ["001", "002", "003"],
            "Task": ["Design UI", "Backend API", "Integration"],
            "Description": [
                "Create user interface",
                "Develop REST API",
                "Integrate frontend with backend",
            ],
            "DoD": [
                "UI mockups approved",
                "API endpoints tested",
                "End-to-end testing complete",
            ],
            "Dependency": ["", "001", "001,002"],
            "Owner": ["Alice", "Bob", "Charlie"],
            "Optimistic": [3, 5, 2],
            "Nominal": [5, 8, 4],
            "Pessimistic": [8, 12, 7],
        }

        sample_df = pd.DataFrame(sample_data)
        st.dataframe(sample_df)

        st.markdown(
            """
        **Column Descriptions:**
        - **Project**: Name of the project
        - **Task ID**: Unique identifier for each task (e.g., 001, 002, 003)
        - **Task**: Task name/title
        - **Description**: Detailed description of the task
        - **DoD**: Definition of Done
        - **Dependency**: Task IDs that must be completed before this task (comma-separated for multiple dependencies)
        - **Owner**: Person responsible for the task
        - **Optimistic**: Best-case estimate (in days)
        - **Nominal**: Most likely estimate (in days)
        - **Pessimistic**: Worst-case estimate (in days)
        
        *Note: The app will automatically calculate Expected time, Standard Deviation, and confidence intervals using PERT formulas.*
        """
        )


if __name__ == "__main__":
    main()
