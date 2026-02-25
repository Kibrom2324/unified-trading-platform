# SuperAGI Orchestration Directive

## Goal
Provide minimal SuperAGI definitions to trigger local HTTP endpoints for each agent.

## Success Criteria
- Agent definitions exist for all pipeline agents
- Tool definitions call local /health endpoints

## Inputs
- Agent host/ports from config/app.yaml

## Tools/Services
- superagi/agents/*.yaml
- superagi/tools/*.yaml

## Outputs
- SuperAGI can register agents and tools without external dependencies

## Steps
1. Register each agent definition.
2. Register tool definitions pointing to local HTTP endpoints.
3. Verify /health endpoints respond.
