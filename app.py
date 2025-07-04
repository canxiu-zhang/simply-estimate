from datetime import datetime, timedelta

import networkx as nx
import numpy as np
import pandas as pd
import streamlit as st

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


def find_critical_path(tasks_df, resource_constrained=True):
    """Find the critical path using network analysis with optional resource constraints"""
    if tasks_df.empty:
        return [], {}, 0

    # Create a directed graph
    G = nx.DiGraph()

    # Add nodes with expected times and owner information
    for _, task in tasks_df.iterrows():
        expected_time, _, _ = calculate_pert_estimates(
            task["min_time"], task["max_time"], task["most_likely_time"]
        )
        G.add_node(
            task["task_id"],
            duration=expected_time,
            name=task["name"],
            owner=task["owner"],
        )

    # Add edges based on dependencies
    for _, task in tasks_df.iterrows():
        if task["dependencies"]:
            deps = [
                dep.strip() for dep in task["dependencies"].split(",") if dep.strip()
            ]
            for dep in deps:
                if dep in [t["task_id"] for _, t in tasks_df.iterrows()]:
                    G.add_edge(dep, task["task_id"])

    # Add resource constraints: tasks with same owner cannot run in parallel
    if resource_constrained:
        tasks_by_owner = {}
        for _, task in tasks_df.iterrows():
            owner = task["owner"]
            if owner not in tasks_by_owner:
                tasks_by_owner[owner] = []
            tasks_by_owner[owner].append(task["task_id"])

        # For each owner, add dependencies between tasks that have no explicit dependencies
        for owner, task_ids in tasks_by_owner.items():
            if len(task_ids) > 1:
                # Get tasks for this owner that have no dependencies
                independent_tasks = []
                for task_id in task_ids:
                    if not list(G.predecessors(task_id)):  # No predecessors
                        independent_tasks.append(task_id)

                # Create a chain of independent tasks for the same owner
                for i in range(len(independent_tasks) - 1):
                    G.add_edge(independent_tasks[i], independent_tasks[i + 1])

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

        # Initialize latest finish times for all end nodes
        for node in G.nodes():
            successors = list(G.successors(node))
            if not successors:  # End nodes
                latest_finish[node] = project_duration

        # Work backwards
        for node in reversed(list(nx.topological_sort(G))):
            duration = G.nodes[node]["duration"]
            successors = list(G.successors(node))

            if not successors:
                latest_finish[node] = project_duration
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


# Main app
st.title("🛀 Simply Estimate")
st.markdown("Plan your projects with PERT analysis and critical path identification")

# Create two columns for side-by-side layout
col_left, col_right = st.columns([1, 1])

# Section 1: Task Input (Left Column)
with col_left:
    st.header("📝 Section 1: Task Management")

    st.markdown(
        "Enter all your project tasks in the table below. You can add or remove rows as needed."
    )

    # Initialize the data editor with sample data if no tasks exist
    if not st.session_state.tasks:
        initial_data = pd.DataFrame(
            {
                "Task ID": ["T001", "T002", "T003"],
                "Task Name": ["Task 1", "Task 2", "Task 3"],
                "Owner": ["John Doe", "Jane Smith", "Bob Johnson"],
                "Description": [
                    "Description for task 1",
                    "Description for task 2",
                    "Description for task 3",
                ],
                "Definition of Done": [
                    "DOD for task 1",
                    "DOD for task 2",
                    "DOD for task 3",
                ],
                "Optimistic Time (days)": [1.0, 2.0, 1.5],
                "Most Likely Time (days)": [3.0, 4.0, 3.5],
                "Pessimistic Time (days)": [5.0, 6.0, 5.5],
                "Dependencies": ["", "T001", ""],
            }
        )
    else:
        # Convert existing tasks to DataFrame for editing
        initial_data = pd.DataFrame(
            [
                {
                    "Task ID": task["task_id"],
                    "Task Name": task["name"],
                    "Owner": task["owner"],
                    "Description": task["description"],
                    "Definition of Done": task["definition_of_done"],
                    "Optimistic Time (days)": task["min_time"],
                    "Most Likely Time (days)": task["most_likely_time"],
                    "Pessimistic Time (days)": task["max_time"],
                    "Dependencies": task["dependencies"],
                }
                for task in st.session_state.tasks
            ]
        )

    # Configure column types for the data editor
    column_config = {
        "Task ID": st.column_config.TextColumn(
            "Task ID*",
            required=True,
            max_chars=10,
            help="Unique identifier for the task (e.g., T001, T002)",
        ),
        "Task Name": st.column_config.TextColumn(
            "Task Name*", required=True, max_chars=100
        ),
        "Owner": st.column_config.TextColumn("Owner*", required=True, max_chars=50),
        "Description": st.column_config.TextColumn("Description", max_chars=200),
        "Definition of Done": st.column_config.TextColumn(
            "Definition of Done", max_chars=200
        ),
        "Optimistic Time (days)": st.column_config.NumberColumn(
            "Optimistic Time (days)*",
            min_value=0.1,
            max_value=1000.0,
            step=0.1,
            format="%.1f",
            required=True,
        ),
        "Most Likely Time (days)": st.column_config.NumberColumn(
            "Most Likely Time (days)*",
            min_value=0.1,
            max_value=1000.0,
            step=0.1,
            format="%.1f",
            required=True,
        ),
        "Pessimistic Time (days)": st.column_config.NumberColumn(
            "Pessimistic Time (days)*",
            min_value=0.1,
            max_value=1000.0,
            step=0.1,
            format="%.1f",
            required=True,
        ),
        "Dependencies": st.column_config.TextColumn(
            "Dependencies",
            help="Comma-separated Task IDs that must be completed before this task (e.g., T001,T002)",
            max_chars=200,
        ),
    }

    # Data editor for tasks
    edited_data = st.data_editor(
        initial_data,
        column_config=column_config,
        num_rows="dynamic",
        use_container_width=True,
        key="tasks_editor",
    )

    button_col1, button_col2 = st.columns([1, 1])

    with button_col1:
        if st.button("💾 Save All Tasks", type="primary"):
            # Validate and save tasks
            valid_tasks = []
            errors = []
            task_ids = set()

            for idx, row in edited_data.iterrows():
                # Check required fields
                if pd.isna(row["Task ID"]) or row["Task ID"].strip() == "":
                    errors.append(f"Row {idx + 1}: Task ID is required")
                    continue
                if pd.isna(row["Task Name"]) or row["Task Name"].strip() == "":
                    errors.append(f"Row {idx + 1}: Task Name is required")
                    continue
                if pd.isna(row["Owner"]) or row["Owner"].strip() == "":
                    errors.append(f"Row {idx + 1}: Owner is required")
                    continue
                if (
                    pd.isna(row["Optimistic Time (days)"])
                    or pd.isna(row["Most Likely Time (days)"])
                    or pd.isna(row["Pessimistic Time (days)"])
                ):
                    errors.append(f"Row {idx + 1}: All time estimates are required")
                    continue

                # Check for duplicate Task IDs
                task_id = row["Task ID"].strip().upper()
                if task_id in task_ids:
                    errors.append(f"Row {idx + 1}: Task ID '{task_id}' is duplicated")
                    continue
                task_ids.add(task_id)

                # Validate time estimates
                opt_time = row["Optimistic Time (days)"]
                likely_time = row["Most Likely Time (days)"]
                pess_time = row["Pessimistic Time (days)"]

                if not (opt_time <= likely_time <= pess_time):
                    errors.append(
                        f"Row {idx + 1}: Time estimates must satisfy Optimistic ≤ Most Likely ≤ Pessimistic"
                    )
                    continue

                # Create task object
                task = {
                    "task_id": task_id,
                    "name": row["Task Name"].strip(),
                    "owner": row["Owner"].strip(),
                    "description": (
                        row["Description"] if pd.notna(row["Description"]) else ""
                    ),
                    "definition_of_done": (
                        row["Definition of Done"]
                        if pd.notna(row["Definition of Done"])
                        else ""
                    ),
                    "min_time": opt_time,
                    "most_likely_time": likely_time,
                    "max_time": pess_time,
                    "dependencies": (
                        row["Dependencies"].strip().upper()
                        if pd.notna(row["Dependencies"])
                        else ""
                    ),
                }
                valid_tasks.append(task)

            # Validate dependencies after all tasks are processed
            if not errors:
                all_task_ids = {task["task_id"] for task in valid_tasks}
                for idx, task in enumerate(valid_tasks):
                    if task["dependencies"]:
                        deps = [
                            dep.strip()
                            for dep in task["dependencies"].split(",")
                            if dep.strip()
                        ]
                        for dep in deps:
                            if dep not in all_task_ids:
                                errors.append(
                                    f"Task {task['task_id']}: Dependency '{dep}' does not exist"
                                )

            if errors:
                st.error(
                    "Please fix the following errors:\n"
                    + "\n".join(f"• {error}" for error in errors)
                )
            else:
                st.session_state.tasks = valid_tasks
                st.success(f"Successfully saved {len(valid_tasks)} tasks!")
                st.rerun()

    with button_col2:
        if st.session_state.tasks:
            if st.button("🗑️ Clear All Tasks", type="secondary"):
                st.session_state.tasks = []
                st.rerun()

# Section 2: Analysis (Right Column)
with col_right:

    # Section 2: Analysis
    st.header("📊 Section 2: Project Analysis")

    # Resource constraints toggle
    st.subheader("⚙️ Resource Settings")
    resource_constrained = st.checkbox(
        "Enable Resource Constraints",
        value=True,
        help="When enabled, tasks with the same owner cannot run in parallel (realistic for single-person or limited teams)",
    )

    if len(st.session_state.tasks) > 0:
        tasks_df = pd.DataFrame(st.session_state.tasks)
        critical_path, slack_times, project_duration = find_critical_path(
            tasks_df, resource_constrained
        )

        # Critical Path Analysis
        st.subheader("🎯 Critical Path Analysis")

        if critical_path:
            # Create a mapping of task_id to task_name for better display
            task_id_to_name = {
                task["task_id"]: task["name"] for task in st.session_state.tasks
            }
            critical_path_display = [
                f"{task_id} ({task_id_to_name[task_id]})" for task_id in critical_path
            ]
            st.success(f"**Critical Path:** {' → '.join(critical_path_display)}")
        else:
            st.warning(
                "Could not determine critical path. Please check task dependencies."
            )

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
                if task["task_id"] in critical_path
            ]
        )

        project_std_dev = np.sqrt(total_variance)

        # Calculate comparison with unlimited resources
        if resource_constrained:
            _, _, unlimited_duration = find_critical_path(
                tasks_df, resource_constrained=False
            )
            time_difference = project_duration - unlimited_duration
        else:
            unlimited_duration = project_duration
            time_difference = 0

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Total Work Effort",
                f"{total_expected:.1f} days",
                help="Sum of all individual task durations",
            )
            st.metric("Actual Project Duration", f"{project_duration:.1f} days")

        with col2:
            st.metric("Project Standard Deviation", f"{project_std_dev:.1f} days")
            if resource_constrained:
                st.metric(
                    "Resource Impact",
                    (
                        f"+{time_difference:.1f} days"
                        if time_difference > 0
                        else f"{time_difference:.1f} days"
                    ),
                    help="Additional time due to resource constraints vs unlimited resources",
                )
            else:
                st.metric(
                    "68% Confidence Range",
                    f"{project_duration - project_std_dev:.1f} - {project_duration + project_std_dev:.1f} days",
                )

        # Explain the difference between the metrics
        if resource_constrained:
            if time_difference > 0:
                st.info(
                    f"📋 **Resource-Constrained Scheduling**: Tasks with the same owner are scheduled sequentially. "
                    f"This adds **{time_difference:.1f} days** compared to unlimited resources scenario."
                )
            else:
                st.info(
                    "📋 **Resource-Constrained Scheduling**: Tasks with the same owner are scheduled sequentially. "
                    "No additional time needed as tasks don't conflict."
                )

            # Show breakdown by owner
            if time_difference > 0:
                st.subheader("👥 Resource Utilization")
                owner_breakdown = {}
                for task in st.session_state.tasks:
                    owner = task["owner"]
                    expected_time, _, _ = calculate_pert_estimates(
                        task["min_time"], task["max_time"], task["most_likely_time"]
                    )
                    if owner not in owner_breakdown:
                        owner_breakdown[owner] = {"tasks": 0, "total_time": 0}
                    owner_breakdown[owner]["tasks"] += 1
                    owner_breakdown[owner]["total_time"] += expected_time

                for owner, info in owner_breakdown.items():
                    if info["tasks"] > 1:
                        st.write(
                            f"**{owner}**: {info['tasks']} tasks, {info['total_time']:.1f} days total work"
                        )
                    else:
                        st.write(
                            f"**{owner}**: {info['tasks']} task, {info['total_time']:.1f} days"
                        )
        else:
            st.warning(
                "⚠️ **Unlimited Resources Mode**: All tasks without dependencies start simultaneously, "
                "assuming unlimited resources are available."
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
                slack = slack_times.get(task["task_id"], 0)
                is_critical = task["task_id"] in critical_path

                export_data.append(
                    {
                        "Task ID": task["task_id"],
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
