import math
from datetime import datetime, timedelta

import matplotlib.patches as mpatches
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


def create_gantt_chart(timeline, project_name, start_date=None):
    """Create Gantt chart visualization"""
    # Prepare data for Gantt chart
    gantt_data = []
    base_date = (
        start_date if start_date else datetime.now().date()
    )  # Use provided start date or current date

    for task_id, info in timeline.items():
        start_date_calc = add_business_days(base_date, info["start_time"])
        end_date_calc = add_business_days(base_date, info["end_time"])

        gantt_data.append(
            {
                "Task": f"{task_id}: {info['task'][:30]}...",
                "Start": start_date_calc,
                "Finish": end_date_calc,
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


def create_dependency_graph(tasks_df, G, critical_path=None):
    """Create dependency graph visualization with optional critical path highlighting"""
    # Set up the plot
    fig, ax = plt.subplots(figsize=(12, 8))

    # Calculate layout
    pos = nx.spring_layout(G, k=3, iterations=50)

    # Define node colors
    node_colors = []
    for node in G.nodes():
        if critical_path and node in critical_path:
            node_colors.append("red")  # Critical path nodes in red
        else:
            node_colors.append("lightblue")  # Regular nodes in light blue

    # Draw nodes
    nx.draw_networkx_nodes(
        G, pos, node_color=node_colors, node_size=1500, alpha=0.7, ax=ax
    )

    # Define edge colors
    edge_colors = []
    edge_widths = []
    if critical_path:
        for edge in G.edges():
            # Check if this edge is part of the critical path
            if (
                edge[0] in critical_path
                and edge[1] in critical_path
                and critical_path.index(edge[1]) == critical_path.index(edge[0]) + 1
            ):
                edge_colors.append("red")
                edge_widths.append(3.0)
            else:
                edge_colors.append("gray")
                edge_widths.append(1.0)
    else:
        edge_colors = ["gray"] * len(G.edges())
        edge_widths = [1.0] * len(G.edges())

    # Draw edges
    nx.draw_networkx_edges(
        G,
        pos,
        edge_color=edge_colors,
        arrows=True,
        arrowsize=20,
        width=edge_widths,
        ax=ax,
    )

    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight="bold", ax=ax)

    # Add legend if critical path exists
    if critical_path:
        legend_elements = [
            mpatches.Patch(facecolor="red", alpha=0.7, label="Critical Path"),
            mpatches.Patch(facecolor="lightblue", alpha=0.7, label="Regular Tasks"),
        ]
        ax.legend(handles=legend_elements, loc="upper right")

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


def calculate_critical_path(timeline, G):
    """Calculate the critical path through the project"""
    # Find the task that ends last (project end)
    project_end_time = max(info["end_time"] for info in timeline.values())
    end_tasks = [
        task_id
        for task_id, info in timeline.items()
        if info["end_time"] == project_end_time
    ]

    critical_path = []
    critical_path_duration = 0

    # For each potential end task, trace back to find the longest path
    for end_task in end_tasks:
        path = []
        current_task = end_task
        path_duration = 0

        # Trace back through dependencies to find the longest path
        while current_task:
            path.insert(0, current_task)
            path_duration += timeline[current_task]["duration"]

            # Find the predecessor that ends latest (on critical path)
            predecessors = [edge[0] for edge in G.edges() if edge[1] == current_task]
            if not predecessors:
                break

            # Among predecessors, find the one that ends latest
            latest_end = -1
            next_task = None
            for pred in predecessors:
                if timeline[pred]["end_time"] > latest_end:
                    latest_end = timeline[pred]["end_time"]
                    next_task = pred

            current_task = next_task

        # Keep the longest path as critical path
        if path_duration > critical_path_duration:
            critical_path = path
            critical_path_duration = path_duration

    return critical_path, critical_path_duration


def add_business_days(start_date, business_days):
    """Add business days to a start date, skipping weekends (Saturday=5, Sunday=6)"""
    current_date = start_date
    days_added = 0

    while days_added < business_days:
        current_date += timedelta(days=1)
        # Skip weekends (Monday=0, Sunday=6)
        if current_date.weekday() < 5:  # Monday to Friday
            days_added += 1

    return current_date


def main():
    st.title("📊 Simply Estimate")
    st.markdown(
        "*Using PERT (Program Evaluation And Review Technique) and CPM (Critical Path Method) to plan projects.*"
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
                with st.expander(f"🎯 Project: {project}", expanded=False):
                    # Filter data for current project
                    project_df = df[df["Project"] == project].copy()

                    # Perform topological sort
                    topo_order, G = topological_sort(project_df)
                    if topo_order is None:
                        continue  # Calculate project timeline
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
                        ["📈 Analysis", "📊 Timeline", "🔗 Dependencies", "👥 Workload"]
                    )

                    with tab1:
                        st.subheader("Project Analysis")

                        # Calculate confidence intervals
                        confidence_intervals, project_end, total_std_dev = (
                            calculate_confidence_intervals(timeline)
                        )

                        # Display metrics
                        col1, col2, col3 = st.columns(3)

                        with col1:
                            st.metric(
                                "Expected Duration", f"{project_end:.1f} business days"
                            )

                        with col2:
                            st.metric(
                                "Standard Deviation",
                                f"{total_std_dev:.1f} business days",
                            )

                        with col3:
                            st.metric("Total Tasks", len(timeline))

                        # Confidence intervals
                        st.subheader("Confidence Intervals")
                        ci_data = []
                        for level, (lower, upper) in confidence_intervals.items():
                            ci_data.append(
                                {
                                    "Confidence Level": level.replace("_", " ").upper(),
                                    "Lower Bound": f"{lower:.1f} business days",
                                    "Upper Bound": f"{upper:.1f} business days",
                                    "Range": f"{upper - lower:.1f} business days",
                                }
                            )

                        ci_df = pd.DataFrame(ci_data)
                        st.dataframe(ci_df)

                        # Risk analysis
                        st.subheader("Risk Analysis")
                        st.write("**Interpretation:**")
                        st.write(
                            "- **68% confidence**: Project will complete between {:.1f} and {:.1f} business days".format(
                                confidence_intervals["1_sd"][0],
                                confidence_intervals["1_sd"][1],
                            )
                        )
                        st.write(
                            "- **95% confidence**: Project will complete between {:.1f} and {:.1f} business days".format(
                                confidence_intervals["2_sd"][0],
                                confidence_intervals["2_sd"][1],
                            )
                        )
                        st.write(
                            "- **99.7% confidence**: Project will complete between {:.1f} and {:.1f} business days".format(
                                confidence_intervals["3_sd"][0],
                                confidence_intervals["3_sd"][1],
                            )
                        )

                        # Critical path analysis
                        st.subheader("Critical Path Analysis")
                        critical_path, critical_path_duration = calculate_critical_path(
                            timeline, G
                        )

                        st.write("**Critical Path:**")
                        st.write(" ➜ ".join(critical_path))

                        st.write(
                            f"**Critical Path Duration:** {critical_path_duration:.1f} business days"
                        )

                    with tab2:
                        st.subheader("Gantt Chart")

                        # Add date picker for project start date
                        col1, col2 = st.columns([1, 2])
                        with col1:
                            start_date = st.date_input(
                                "Project Start Date",
                                value=datetime.now().date(),
                                help="Select the date when the project will begin",
                                key=f"start_date_{project}",
                            )

                        gantt_fig = create_gantt_chart(timeline, project, start_date)
                        st.plotly_chart(gantt_fig, use_container_width=True)

                        # Project summary
                        project_end = max(
                            info["end_time"] for info in timeline.values()
                        )
                        with col2:
                            st.metric(
                                "Project Duration", f"{project_end:.1f} business days"
                            )
                            # Calculate and display project end date (business days only)
                            end_date = add_business_days(start_date, project_end)
                            st.metric("Project End Date", end_date.strftime("%Y-%m-%d"))

                            # Also show calendar days for reference
                            calendar_days = (end_date - start_date).days
                            st.metric("Calendar Days", f"{calendar_days} days")

                    with tab3:
                        st.subheader("Task Dependencies")
                        if len(G.edges()) > 0:
                            # Calculate critical path first
                            critical_path, critical_path_duration = (
                                calculate_critical_path(timeline, G)
                            )

                            # Create dependency graph with critical path highlighting
                            dep_fig = create_dependency_graph(
                                project_df, G, critical_path
                            )
                            st.pyplot(dep_fig)

                            # Display critical path analysis
                            st.subheader("🎯 Critical Path Analysis")

                            if critical_path:
                                st.write("**Critical Path:**")
                                path_text = " → ".join(
                                    [
                                        f"{task_id} ({timeline[task_id]['task'][:20]}...)"
                                        for task_id in critical_path
                                    ]
                                )
                                st.write(path_text)

                                col1, col2 = st.columns(2)
                                with col1:
                                    st.metric(
                                        "Critical Path Duration",
                                        f"{critical_path_duration:.1f} business days",
                                    )
                                with col2:
                                    st.metric(
                                        "Number of Critical Tasks", len(critical_path)
                                    )

                                # Show critical path details in a table
                                st.subheader("Critical Path Task Details")
                                critical_path_data = []
                                for i, task_id in enumerate(critical_path):
                                    task_info = timeline[task_id]
                                    critical_path_data.append(
                                        {
                                            "Sequence": i + 1,
                                            "Task ID": task_id,
                                            "Task": task_info["task"],
                                            "Owner": task_info["owner"],
                                            "Duration": f"{task_info['duration']:.1f} business days",
                                            "Start": f"{task_info['start_time']:.1f}",
                                            "End": f"{task_info['end_time']:.1f}",
                                        }
                                    )

                                critical_df = pd.DataFrame(critical_path_data)
                                st.dataframe(critical_df, use_container_width=True)

                                st.info(
                                    "💡 **Critical Path**: These tasks directly impact the project completion date. Any delay in these tasks will delay the entire project."
                                )
                            else:
                                st.warning("Could not determine critical path.")
                        else:
                            st.info("No dependencies found for this project.")

                    with tab4:
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
                                title="Total Duration per Person (business days)",
                            )
                            st.plotly_chart(fig_duration, use_container_width=True)

                        # Detailed workload breakdown
                        st.subheader("Detailed Workload Breakdown")
                        for owner, data in workload_data.items():
                            with st.expander(
                                f"{owner} - {data['total_tasks']} tasks, {data['total_duration']:.1f} business days"
                            ):
                                for task in data["tasks"]:
                                    st.write(
                                        f"**{task['task_id']}**: {task['task']} ({task['duration']:.1f} business days)"
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
        - **Optimistic**: Best-case estimate (in business days)
        - **Nominal**: Most likely estimate (in business days)
        - **Pessimistic**: Worst-case estimate (in business days)
        
        *Note: The app will automatically calculate Expected time, Standard Deviation, and confidence intervals using PERT formulas. All durations are in business days (weekends are excluded from calculations).*
        """
        )


if __name__ == "__main__":
    main()
