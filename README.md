Overall Admission Rate:
   total_enquiries  total_admissions overall_admission_rate
0             9649               658                  6.82%


Admission Rate by Source:
                 source  number_of_enquiries  admissions admission_rate
0  Click-to-WhatsApp Ad                 3424         195          5.70%
1    Facebook Lead Form                 1943          81          4.17%
2   Instagram Lead Form                 1598          47          2.94%
3              Referral                  677         103         15.21%
4               Walk-in                  762         170         22.31%
5          Website Form                 1169          61          5.22%
6            YouTube Ad                   76           1          1.32%

Admission Rate by Branch:
      branch  number_of_enquiries  admissions admission_rate
0    Central                 3775         245          6.49%
1   Lakeview                 3372         225          6.67%
2  Riverside                 2502         188          7.51%

Admission Rate by Course:
          course_interest  number_of_enquiries  admissions admission_rate
0          Cloud & DevOps                  751          58          7.72%
1          Data Analytics                 1605         117          7.29%
2       Data Science & AI                 1739         151          8.68%
3       Digital Marketing                 1329          71          5.34%
4  Full Stack Development                 2278         138          6.06%
5      Python Programming                  999          61          6.11%
6            UI/UX Design                  948          62          6.54%




How long does a first call usually take, and does it vary over the year?
Overall mean : 5.104104(in hours)
"While response times remain stable for most of the year (median ~1.9 hrs, mean ~4.0 hrs), the April–May admission peak created a severe operational bottleneck. Average wait times tripled to 12.0 hours in May, with half of all leads waiting longer than 5.5 hours to receive their first call."


How many enquiries never send a single message?
   total_enquiries  leads_never_sent_message zero_message_percentage
0             9649                      2332                  24.17%


Which columns describe the enquiry at creation, and which describe it at export?
       Column Category  Column Count                                                                                                             Columns
Creation-Time (Static)            11 lead_id, created_at, source, branch, course_interest, full_name, phone, email, education_status, age, home_locality
 Export-Time (Dynamic)             6             assigned_counselor_id, current_stage, updated_at, last_contacted_at, legacy_score, legacy_score_version



How many phone formats exist, and how many people seem to appear more than once?
=== Detailed Phone Format Distribution ===
                       phone_format  count
   Country Code (+91 / 91 Prefixed)   3427
        Plain 10-Digit (XXXXXXXXXX)   2548
Contains Spaces (e.g., XXXXX XXXXX)   1739
         Leading Zero (0XXXXXXXXXX)    978
           Hyphenated (XXXXX-XXXXX)    957

Distinct Phone Numbers Appearing >1 Time: 691 (accounting for 1434 total enquiry records)
Distinct Email Addresses Appearing >1 Time: 301 (accounting for 615 total enquiry records)
Distinct (Name + Course) Combinations Appearing >1 Time: 1912 (accounting for 4478 total enquiry records)



Are all timestamps in the same timezone and format?
=== Timestamp & Timezone Audit ===
          Dataset Timestamp Column             Sample Record               Native Timezone                     Format
        leads.csv       created_at       2025-09-01 00:58:10 Asia/Kolkata (Implicit Naive)        YYYY-MM-DD HH:MM:SS
        calls.csv    started_at_ms             1756700776264           UTC (Unix Epoch ms) Numeric Epoch Milliseconds
     messages.csv          sent_at      2025-08-31T19:28:31Z              UTC (ISO 8601 Z)       YYYY-MM-DDTHH:MM:SSZ
stage_history.csv       changed_at 2026-04-09T06:36:51+05:30  Asia/Kolkata (+05:30 offset)  YYYY-MM-DDTHH:MM:SS+05:30

No , not all timestamps in the same timezone and formate
