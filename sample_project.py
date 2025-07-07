#!/usr/bin/env python3
"""
Script to generate a sample Excel file for testing the Simply Estimate app
"""

import numpy as np
import pandas as pd


def create_sample_project():
    """Create sample project data"""

    # Sample data for a software development project
    sample_data = {
        "Project": ["E-Commerce Platform"] * 12,
        "Task ID": [
            "001",
            "002",
            "003",
            "004",
            "005",
            "006",
            "007",
            "008",
            "009",
            "010",
            "011",
            "012",
        ],
        "Task": [
            "Requirements Analysis",
            "Database Design",
            "UI/UX Design",
            "User Authentication",
            "Product Catalog",
            "Shopping Cart",
            "Payment Integration",
            "Order Management",
            "Testing Framework",
            "Unit Testing",
            "Integration Testing",
            "Deployment",
        ],
        "Description": [
            "Gather and analyze business requirements",
            "Design database schema and relationships",
            "Create user interface mockups and prototypes",
            "Implement user login and registration system",
            "Build product listing and search functionality",
            "Develop shopping cart and wishlist features",
            "Integrate payment gateway (Stripe/PayPal)",
            "Create order processing and tracking system",
            "Set up automated testing infrastructure",
            "Write and execute unit tests for all components",
            "Perform end-to-end integration testing",
            "Deploy application to production environment",
        ],
        "DoD": [
            "Requirements document approved by stakeholders",
            "Database schema reviewed and approved",
            "UI mockups approved by design team",
            "Authentication system passes security review",
            "Product catalog supports 1000+ items",
            "Shopping cart handles concurrent users",
            "Payment processing is PCI compliant",
            "Order system handles 100+ orders/day",
            "Testing framework covers 90% code coverage",
            "All unit tests pass with 95% coverage",
            "Integration tests pass without critical bugs",
            "Application is live and accessible",
        ],
        "Dependency": [
            "",  # Requirements Analysis - no dependencies
            "001",  # Database Design depends on Requirements
            "001",  # UI/UX Design depends on Requirements
            "002",  # User Auth depends on Database Design
            "002",  # Product Catalog depends on Database Design
            "002,004",  # Shopping Cart depends on Database and Auth
            "004",  # Payment depends on User Auth
            "005,006",  # Order Management depends on Catalog and Cart
            "001",  # Testing Framework depends on Requirements
            "003,004,005,006,007,008",  # Unit Testing depends on main features
            "009,010",  # Integration Testing depends on Testing Framework and Unit Tests
            "011",  # Deployment depends on Integration Testing
        ],
        "Owner": [
            "Sarah (BA)",
            "Mike (DBA)",
            "Lisa (Designer)",
            "John (Backend)",
            "John (Backend)",
            "Alice (Frontend)",
            "John (Backend)",
            "Bob (Backend)",
            "Charlie (QA)",
            "Charlie (QA)",
            "Charlie (QA)",
            "David (DevOps)",
        ],
        "Optimistic": [3, 2, 4, 3, 5, 4, 3, 4, 2, 6, 4, 2],
        "Nominal": [5, 4, 7, 5, 8, 6, 5, 7, 3, 10, 6, 3],
        "Pessimistic": [8, 7, 12, 8, 12, 10, 8, 12, 5, 15, 10, 5],
    }

    return pd.DataFrame(sample_data)


def create_multi_project_sample():
    """Create sample data with multiple projects"""

    # First project data
    project1 = create_sample_project()

    # Second project data - Mobile App
    project2_data = {
        "Project": ["Mobile App"] * 8,
        "Task ID": ["M001", "M002", "M003", "M004", "M005", "M006", "M007", "M008"],
        "Task": [
            "Market Research",
            "App Architecture",
            "Backend API",
            "iOS Development",
            "Android Development",
            "API Integration",
            "Testing",
            "App Store Release",
        ],
        "Description": [
            "Research market and competitor analysis",
            "Design mobile app architecture",
            "Develop REST API for mobile app",
            "Build iOS native application",
            "Build Android native application",
            "Integrate mobile apps with backend API",
            "Comprehensive mobile app testing",
            "Submit to App Store and Google Play",
        ],
        "DoD": [
            "Market research report completed",
            "Architecture document approved",
            "API documented and tested",
            "iOS app ready for testing",
            "Android app ready for testing",
            "Apps successfully communicate with API",
            "Apps pass QA testing",
            "Apps available in stores",
        ],
        "Dependency": [
            "",  # Market Research - no dependencies
            "M001",  # App Architecture depends on Market Research
            "M002",  # Backend API depends on Architecture
            "M002",  # iOS Development depends on Architecture
            "M002",  # Android Development depends on Architecture
            "M003,M004,M005",  # API Integration depends on API and both apps
            "M006",  # Testing depends on Integration
            "M007",  # Release depends on Testing
        ],
        "Owner": [
            "Emma (PM)",
            "Alex (Architect)",
            "John (Backend)",
            "iOS Team",
            "Android Team",
            "Full Stack Team",
            "QA Team",
            "DevOps Team",
        ],
        "Optimistic": [2, 3, 5, 8, 8, 3, 4, 1],
        "Nominal": [4, 5, 8, 12, 12, 5, 6, 2],
        "Pessimistic": [7, 8, 12, 18, 18, 8, 10, 4],
    }

    project2 = pd.DataFrame(project2_data)

    # Combine both projects
    combined_df = pd.concat([project1, project2], ignore_index=True)

    return combined_df


if __name__ == "__main__":
    # Create sample data
    df = create_multi_project_sample()

    # Save to Excel file
    output_file = "sample_projects.xlsx"
    df.to_excel(output_file, index=False)

    print(f"Sample Excel file created: {output_file}")
    print(f"Total tasks: {len(df)}")
    print(f"Projects: {df['Project'].unique()}")
    print("\nSample data preview:")
    print(df.head())
    print(df.head())
