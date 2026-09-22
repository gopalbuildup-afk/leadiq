# LeadIQ Data Analysis

## Overall Admission Rate

| Total enquiries | Total admissions | Overall admission rate |
|---:|---:|---:|
| 9,649 | 658 | 6.82% |


## Admission Rate by Source

| Source | Enquiries | Admissions | Admission rate |
|---|---:|---:|---:|
| Click-to-WhatsApp Ad | 3,424 | 195 | 5.70% |
| Facebook Lead Form | 1,943 | 81 | 4.17% |
| Instagram Lead Form | 1,598 | 47 | 2.94% |
| Referral | 677 | 103 | 15.21% |
| Walk-in | 762 | 170 | 22.31% |
| Website Form | 1,169 | 61 | 5.22% |
| YouTube Ad | 76 | 1 | 1.32% |

## Admission Rate by Branch

| Branch | Enquiries | Admissions | Admission rate |
|---|---:|---:|---:|
| Central | 3,775 | 245 | 6.49% |
| Lakeview | 3,372 | 225 | 6.67% |
| Riverside | 2,502 | 188 | 7.51% |

## Admission Rate by Course

| Course interest | Enquiries | Admissions | Admission rate |
|---|---:|---:|---:|
| Cloud & DevOps | 751 | 58 | 7.72% |
| Data Analytics | 1,605 | 117 | 7.29% |
| Data Science & AI | 1,739 | 151 | 8.68% |
| Digital Marketing | 1,329 | 71 | 5.34% |
| Full Stack Development | 2,278 | 138 | 6.06% |
| Python Programming | 999 | 61 | 6.11% |
| UI/UX Design | 948 | 62 | 6.54% |




## First-Call Response Time

**Question:** How long does a first call usually take, and does it vary over the year?

**Overall mean:** 5.10 hours

"While response times remain stable for most of the year (median ~1.9 hrs, mean ~4.0 hrs), the April–May admission peak created a severe operational bottleneck. Average wait times tripled to 12.0 hours in May, with half of all leads waiting longer than 5.5 hours to receive their first call."


## Enquiries With No Messages

| Total enquiries | Leads that never sent a message | Percentage with zero messages |
|---:|---:|---:|
| 9,649 | 2,332 | 24.17% |


## Enquiry Column Categories

| Category | Column count | Columns |
|---|---:|---|
| Creation-time (static) | 11 | `lead_id`, `created_at`, `source`, `branch`, `course_interest`, `full_name`, `phone`, `email`, `education_status`, `age`, `home_locality` |
| Export-time (dynamic) | 6 | `assigned_counselor_id`, `current_stage`, `updated_at`, `last_contacted_at`, `legacy_score`, `legacy_score_version` |



## Duplicate Records and Phone Formats

### Phone Format Distribution

| Phone format | Count |
|---|---:|
| Country code (+91 / 91-prefixed) | 3,427 |
| Plain 10-digit (`XXXXXXXXXX`) | 2,548 |
| Contains spaces (for example, `XXXXX XXXXX`) | 1,739 |
| Leading zero (`0XXXXXXXXXX`) | 978 |
| Hyphenated (`XXXXX-XXXXX`) | 957 |

| Duplicate identifier | Distinct values | Enquiry records |
|---|---:|---:|
| Phone numbers | 691 | 1,434 |
| Email addresses | 301 | 615 |
| Name + course combinations | 1,912 | 4,478 |
## Timestamp and Timezone Audit

| Dataset | Timestamp column | Sample record | Native timezone | Format |
|---|---|---|---|---|
| `leads.csv` | `created_at` | `2025-09-01 00:58:10` | Asia/Kolkata (implicit naive) | `YYYY-MM-DD HH:MM:SS` |
| `calls.csv` | `started_at_ms` | `1756700776264` | UTC (Unix epoch ms) | Numeric epoch milliseconds |
| `messages.csv` | `sent_at` | `2025-08-31T19:28:31Z` | UTC (ISO 8601 Z) | `YYYY-MM-DDTHH:MM:SSZ` |
| `stage_history.csv` | `changed_at` | `2026-04-09T06:36:51+05:30` | Asia/Kolkata (+05:30 offset) | `YYYY-MM-DDTHH:MM:SS+05:30` |

**Conclusion:** No. The timestamps do not use a consistent timezone or format.
