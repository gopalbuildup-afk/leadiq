# AI_USAGE.md — LeadIQ Project

## Tools Used
* **ChatGPT**
* **Antigravity**
* **Trea Code**
* **OpenCode**
* **GitHub Copilot**

---

## Real Prompts

### Prompt 1
i want you to explain me entire task from attached document and break it into sections

### Prompt 2
> *"i have provided you pdf describing about entire task , i have completed all 5 module of task with testing , now i want you to complete module 6 and remaining task , in case of deployment at last you can guide me changes should be made in project . also each module is created independent , so don't try to change existing . in extereme case only change , and at time of analysis if you are finding any kind of code or part that is not as per the task described then you can tell me at last instead of directly modifying it , and i also provided you image that shows connection between modules"*

### Prompt 3
> *"I'm designing the M1 and M2 modules for our lead scoring pipeline. The core requirement is to enforce a strict point-in-time scoring moment $t$ exactly 24 hours after an enquiry arrives, ensuring zero data leakage from future events like stage changes or later calls. We need to ingest raw CSV files, parse timestamps explicitly into Asia/Kolkata, build out at least 15 features, and pipe them into a LightGBM model with isotonic calibration."*



## Error Encountered & How It Was Caught

### The Issue
During architectural flow analysis of our time-based feature calculations, an AI-generated test definition was caught attempting to validate exact scoring moments using mismatched time delta logic. Specifically, the test was structured as follows:

```python
def test_exact_scoring_moment_boundary(base_sample_data):
    #     (
    #         leads_df,
    #         _,
    #         _,
    #         _,
    #     ) = base_sample_data

    #     created_at = leads_df[
    #         "created_at"
    #     ].iloc[0]

    #     expected_t = (
    #         created_at
    #         + pd.Timedelta(hours=24)
    #     )

    #     assert expected_t - created_at == pd.Timedelta(
    #         hours=24
    #     )