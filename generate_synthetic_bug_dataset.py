"""Generate a synthetic bug report dataset with controlled severity signal.

This script creates a balanced synthetic dataset for bug category, severity, and
suggested fix prediction. Severity is correlated with category, environment,
and error codes to make the dataset meaningful for model training.
"""

import random
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = BASE_DIR / 'bug_dataset_50k_cpy.csv'

BUG_CATEGORIES = [
    'API Bug', 'Memory Leak', 'Cloud Configuration Bug', 'Authentication Bug', 'Logging Bug',
    'Frontend Routing Bug', 'CI/CD Bug', 'UI Bug', 'Concurrency Bug', 'Monitoring Bug',
    'Performance Bug', 'Authorization Bug', 'Deployment Bug', 'Backend Logic Bug', 'Database Bug',
    'Security Vulnerability'
]
BUG_DOMAINS = ['Mobile', 'Data', 'DevOps', 'Backend Systems', 'Web Development', 'Cloud']
TECH_STACKS = ['Spring Boot', 'GCP', 'Django', 'Flask', 'Laravel', 'AWS', 'PostgreSQL', 'MySQL',
               'Kubernetes', 'Vue', 'Node.js', 'Docker', 'MongoDB', 'Angular', 'React', 'Azure']
SEVERITIES = ['Critical', 'High', 'Medium', 'Low']
ENVIRONMENTS = ['Development', 'Production', 'Staging']
DEVELOPER_ROLES = ['Full-Stack Developer', 'Backend Developer', 'Mobile Developer', 'Frontend Developer',
                   'DevOps Engineer', 'Data Engineer', 'Cloud Engineer', 'QA Engineer', 'Security Engineer']
ROOT_CAUSE_TEMPLATES = {
    'API Bug': 'Misconfiguration or logic issue related to api bug',
    'Memory Leak': 'Misconfiguration or logic issue related to memory leak',
    'Cloud Configuration Bug': 'Misconfiguration or logic issue related to cloud configuration bug',
    'Authentication Bug': 'Misconfiguration or logic issue related to authentication bug',
    'Logging Bug': 'Misconfiguration or logic issue related to logging bug',
    'Frontend Routing Bug': 'Misconfiguration or logic issue related to frontend routing bug',
    'CI/CD Bug': 'Misconfiguration or logic issue related to ci/cd bug',
    'UI Bug': 'Misconfiguration or logic issue related to ui bug',
    'Concurrency Bug': 'Misconfiguration or logic issue related to concurrency bug',
    'Monitoring Bug': 'Misconfiguration or logic issue related to monitoring bug',
    'Performance Bug': 'Misconfiguration or logic issue related to performance bug',
    'Authorization Bug': 'Misconfiguration or logic issue related to authorization bug',
    'Deployment Bug': 'Misconfiguration or logic issue related to deployment bug',
    'Backend Logic Bug': 'Misconfiguration or logic issue related to backend logic bug',
    'Database Bug': 'Misconfiguration or logic issue related to database bug',
    'Security Vulnerability': 'Misconfiguration or logic issue related to security vulnerability',
}
SUGGESTED_FIX_TEMPLATES = {
    category: f'Review and fix the {category.lower()} according to best practices'
    for category in BUG_CATEGORIES
}

CATEGORY_SEVERITY_SCORES = {
    'Security Vulnerability': 4,
    'Performance Bug': 4,
    'Database Bug': 4,
    'Authorization Bug': 4,
    'Authentication Bug': 3,
    'Monitoring Bug': 3,
    'Memory Leak': 3,
    'CI/CD Bug': 3,
    'Backend Logic Bug': 2,
    'Cloud Configuration Bug': 2,
    'API Bug': 2,
    'Concurrency Bug': 2,
    'Deployment Bug': 2,
    'Logging Bug': 1,
    'UI Bug': 1,
    'Frontend Routing Bug': 1,
}

ENVIRONMENT_SCORES = {
    'Production': 4,
    'Staging': 3,
    'Development': 2,
}

ERROR_CODE_SCORES = {
    500: 4,
    502: 4,
    503: 4,
    400: 2,
    401: 3,
    403: 3,
    404: 1,
}

DESCRIPTION_TEMPLATES = [
    'This issue relates to a {category} that occurs in a {domain} service built with {tech}.',
    'The {category} is impacting the {domain} platform and is visible in {environment}.',
    'A {category} has been reported in a {tech} application running on {environment}.',
    'The bug affects {domain} workflows in the {environment} environment and involves {tech}.',
]

EXPLANATION_TEMPLATES = [
    'This bug needs attention from a {role} due to how it affects the system.',
    'A {role} should review the root cause and implement a fix.',
    'The issue is best addressed by a {role} familiar with this stack.',
    'A {role} is required to resolve this problem effectively.',
]


SEVERITY_GENERATORS = {
    'Critical': {
        'categories': ['Security Vulnerability', 'Performance Bug', 'Database Bug', 'Authorization Bug', 'Authentication Bug'],
        'environments': ['Production'],
        'error_codes': [500, 502, 503],
    },
    'High': {
        'categories': ['Monitoring Bug', 'CI/CD Bug', 'Backend Logic Bug', 'Cloud Configuration Bug'],
        'environments': ['Production', 'Staging'],
        'error_codes': [500, 502, 403],
    },
    'Medium': {
        'categories': ['API Bug', 'Concurrency Bug', 'Deployment Bug', 'Memory Leak'],
        'environments': ['Staging', 'Development'],
        'error_codes': [400, 401, 403, 404],
    },
    'Low': {
        'categories': ['Logging Bug', 'UI Bug', 'Frontend Routing Bug'],
        'environments': ['Development', 'Staging'],
        'error_codes': [400, 404],
    },
}


def generate_row(index: int, severity: str) -> dict:
    """Create one synthetic bug report row with controlled severity correlation."""
    category = random.choice(SEVERITY_GENERATORS[severity]['categories'])
    domain = random.choice(BUG_DOMAINS)
    tech = random.choice(TECH_STACKS)
    environment = random.choice(SEVERITY_GENERATORS[severity]['environments'])
    role = random.choice(DEVELOPER_ROLES)
    error_code = random.choice(SEVERITY_GENERATORS[severity]['error_codes'])

    # Add a small amount of label noise to make the dataset more realistic.
    if random.random() < 0.08:
        severity = random.choice(SEVERITIES)

    title = f'{category} detected in system'
    description = random.choice(DESCRIPTION_TEMPLATES).format(
        category=category.lower(), domain=domain.lower(), tech=tech, environment=environment.lower()
    )
    explanation = random.choice(EXPLANATION_TEMPLATES).format(role=role)
    root_cause = ROOT_CAUSE_TEMPLATES[category]
    suggested_fix = SUGGESTED_FIX_TEMPLATES[category]
    created_at = (datetime(2024, 1, 1) + timedelta(days=random.randint(0, 730))).strftime('%Y-%m-%d')

    return {
        'bug_id': f'BUG_{index:06d}',
        'title': title,
        'description': description,
        'error_code': float(error_code),
        'bug_category': category,
        'bug_domain': domain,
        'tech_stack': tech,
        'severity': severity,
        'environment': environment,
        'developer_role': role,
        'root_cause': root_cause,
        'suggested_fix': suggested_fix,
        'explanation': explanation,
        'created_at': created_at,
    }


def main():
    """Generate the full synthetic dataset and save it to disk."""
    random.seed(42)
    rows = []
    per_class = 12500
    index = 1

    # Generate a controlled number of rows for each severity level.
    for severity in SEVERITIES:
        for _ in range(per_class):
            rows.append(generate_row(index, severity))
            index += 1

    # Shuffle the dataset to avoid ordering by severity.
    random.shuffle(rows)
    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f'Generated {len(df)} rows to {OUTPUT_PATH}')


if __name__ == '__main__':
    main()
