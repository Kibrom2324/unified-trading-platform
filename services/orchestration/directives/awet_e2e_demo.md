# AWET End-to-End Demo (SuperAGI)

## Goal
Run a full AWET demo pipeline and verify system health.

## Steps
1. Call `awet_run_demo` to push synthetic data through Kafka.
2. Call `awet_check_pipeline_health` and summarize the status.
3. Stop after the demo is complete.

## Success Criteria
- Demo completes without errors.
- Health check reports all agents healthy.

## Notes
- Paper trading only.
- Do not change risk limits or database configuration.
